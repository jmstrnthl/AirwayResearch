"""
Large-deflection (elastica) shooting solver for an endotracheal tube
bent between laryngeal supports and the anterior tracheal wall.

Geometry is 2D (sagittal plane), in mm. Forces in N, moments in N*mm.

Points (from CT, in any consistent 2D frame):
    A : glottis contact
    B : arytenoid contact
    C : anterior tracheal wall contact (the load point, at the tube's free end)

Modes:
    "cantilever" : tube clamped at B with tangent angle phi_clamp (A unused)
    "pinned"     : pin at A, frictionless/frictional contact at B, load at C
    "fixed"      : clamp at A with tangent angle phi_clamp, contact at B, load at C

Method:
    The tube is integrated *backward* from C (where the bending moment is zero)
    toward A. Contact forces act normal to the tube at the contact point
    (plus optional Coulomb friction opposing advancement), so their directions
    follow the deformed shape automatically. Unknowns are found with fsolve.

    Unknowns   cantilever: [phi_C, P, s_CB]
               pinned/fixed: [phi_C, P, s_CB, R_B, s_BA]
    Conditions cantilever: tube passes through B, angle at B = phi_clamp
               pinned: passes through B and A, moment at A = 0
               fixed:  passes through B and A, angle at A = phi_clamp

Sign conventions:
    Angles are measured counterclockwise from +x, in radians internally.
    Arc length s runs from A (proximal) to C (distal).
    Contact force normal components (P, R_B) are signed along the tube's LEFT
    normal (-sin(phi), cos(phi)) at the contact. Check reported force vectors
    against which side the tissue is on: a contact that would have to PULL
    the tube means that contact has been lost and the configuration is invalid.
    kappa0 is the tube's natural (pre-)curvature, 1/R, positive = curving
    counterclockwise as s increases.

Force components reported by solve() (all in N):
    Trachea (C), two frames:
      C_wall_normal  normal force on the tracheal wall (pressure-type load)
      C_wall_shear   friction shear on the tracheal wall
      C_bending      component perpendicular to the tube tip (bends the tube)
      C_axial        component along the tube tip, + = compresses the tube
    Supports:
      A_normal       glottic contact force (perpendicular to the tube at A)
      B_normal       arytenoid contact force (perpendicular to the tube at B)
      B_tangential   arytenoid friction (along the tube at B)
      push           axial force that must be supplied proximally (operator,
                     via the tube/stylet), + = pushing the tube distally
    C_contact_ok     False if the tracheal wall would have to pull the tube
                     (contact not physical); see the check in solve()
    In cantilever mode, the clamp at B supplies both the sideways reaction
    (B_normal) and the push; A_normal is None.
    P, R_A, R_B remain the resultant magnitudes.
"""

import numpy as np
from scipy.optimize import fsolve


# ----------------------------------------------------------------------------
# Tube properties
# ----------------------------------------------------------------------------
def tube_EI(E_MPa, D_mm, d_mm):
    """Bending stiffness of a hollow circular tube, in N*mm^2."""
    I = np.pi * (D_mm**4 - d_mm**4) / 64.0  # mm^4
    return E_MPa * I  # MPa = N/mm^2


# ----------------------------------------------------------------------------
# Core integration
# ----------------------------------------------------------------------------
def _rhs(state, n, EI, kappa0):
    """Derivatives with respect to u = arc length measured from C toward A.
    state = [x, y, phi, M]; n = internal force (sum of external forces distal
    to the current point)."""
    x, y, phi, M = state
    return np.array([
        -np.cos(phi),
        -np.sin(phi),
        -(M / EI + kappa0),
        -(n[0] * np.sin(phi) - n[1] * np.cos(phi)),
    ])


def _integrate(state, length, n, EI, kappa0, steps, record=None):
    """Fixed-step RK4 (fixed steps keep the residual smooth for the root finder)."""
    h = length / steps
    for _ in range(steps):
        k1 = _rhs(state, n, EI, kappa0)
        k2 = _rhs(state + 0.5 * h * k1, n, EI, kappa0)
        k3 = _rhs(state + 0.5 * h * k2, n, EI, kappa0)
        k4 = _rhs(state + h * k3, n, EI, kappa0)
        state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if record is not None:
            record.append(state.copy())
    return state


def _contact_force(N, phi, mu, eps=1e-9):
    """Force on the tube from a contact: normal component N along the left
    normal, plus friction of magnitude mu*|N| opposing distal advancement."""
    t = np.array([np.cos(phi), np.sin(phi)])
    nh = np.array([-np.sin(phi), np.cos(phi)])
    return N * nh - mu * np.sqrt(N * N + eps * eps) * t


def _load_at_C(P, phiC, mu, wall_angle, load_angle):
    """Force on the tube at the tracheal contact C.
    load_angle : fixed load direction (rad), signed magnitude P, no friction.
    wall_angle : tracheal wall tangent direction at C, pointing distally (rad);
                 force is normal to the WALL, friction along the wall.
    neither    : force normal to the TUBE at C (side of tube resting on wall)."""
    if load_angle is not None:
        return P * np.array([np.cos(load_angle), np.sin(load_angle)])
    if wall_angle is not None:
        return _contact_force(P, wall_angle, mu)
    return _contact_force(P, phiC, mu)


def _split(f, phi):
    """Split force f into (|perpendicular|, signed along) relative to direction phi."""
    t = np.array([np.cos(phi), np.sin(phi)])
    n = np.array([-np.sin(phi), np.cos(phi)])
    return abs(f @ n), f @ t


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _shoot(z, mode, A, B, C, phi_clamp, EI, kappa0, mu, steps, Lref,
           wall_angle=None, load_angle=None, record=False):
    """
    z = [phi_c, P, sCB, RB, sBA]
    """
    phiC, P, sCB = z[0], z[1], z[2]
    state = np.array([C[0], C[1], phiC, 0.0])
    fC = _load_at_C(P, phiC, mu, wall_angle, load_angle)
    n = fC.copy()
    rec = [state.copy()] if record else None

    state = _integrate(state, sCB, n, EI, kappa0, steps, rec)
    res = [(state[0] - B[0]) / Lref, (state[1] - B[1]) / Lref]

    out = {"fC": fC, "phiC": phiC, "phiB": state[2], "MB": state[3], "sCB": sCB}

    if mode == "cantilever":
        res.append(_wrap(state[2] - phi_clamp))
        # clamp at B: reaction force and moment reported under the B / A keys
        out.update(fB=-fC, fA=np.zeros(2), MA=state[3], phiA=None, sBA=0.0)
    else:
        RB, sBA = z[3], z[4]
        fB = _contact_force(RB, state[2], mu)
        n = n + fB
        state = _integrate(state, sBA, n, EI, kappa0, steps, rec)
        res += [(state[0] - A[0]) / Lref, (state[1] - A[1]) / Lref]
        if mode == "pinned":
            res.append(state[3] * Lref / EI)
        else:  # fixed
            res.append(_wrap(state[2] - phi_clamp))
        out.update(fB=fB, fA=-(fC + fB), MA=state[3], phiA=state[2], sBA=sBA)

    if record:
        out["shape"] = np.array(rec)[::-1]  # ordered proximal -> distal
    return np.array(res), out


# ----------------------------------------------------------------------------
# Public solver
# ----------------------------------------------------------------------------
def solve(A, B, C, mode, EI, phi_clamp=None, kappa0=0.0, mu=0.0,
          wall_angle=None, load_angle=None, n_ramp=20, steps=300, start=None):
    """
    Solve one configuration.

    A, B, C   : (x, y) in mm. A may be None for mode="cantilever".
    mode      : "cantilever", "pinned" or "fixed"
    EI        : N*mm^2 (see tube_EI)
    phi_clamp : clamp tangent angle in DEGREES (cantilever: at B; fixed: at A).
                Defaults to the direction A->B.
    kappa0    : natural curvature, 1/mm (e.g. 1/140 for a 140 mm radius tube)
    mu        : Coulomb friction coefficient at B and C
    wall_angle: DEGREES, direction of the tracheal wall at C pointing distally.
                If given, the tracheal force acts normal to the wall (tube tip
                impinging on the wall). If None, it acts normal to the tube
                (tube's side resting on the wall).
    load_angle: DEGREES, fixed direction for the tracheal force (validation
                against textbook cases; ignores friction at C).
    n_ramp    : steps used to walk from the unloaded state to the target
                (tip position, pre-curve, wall angle and friction are ramped together)
    start     : previous result dict, to continue from it (used by sweep)

    Returns a dict with forces, moments, angles, arc lengths and the deformed shape.
    """
    A = None if A is None else np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    if mode not in ("cantilever", "pinned", "fixed"):
        raise ValueError("mode must be 'cantilever', 'pinned' or 'fixed'")
    if mode != "cantilever" and A is None:
        raise ValueError("A is required for pinned/fixed modes")

    if phi_clamp is None:
        if A is None:
            raise ValueError("phi_clamp is required for cantilever mode without A")
        phi_clamp_rad = np.arctan2(*(B - A)[::-1])
    else:
        phi_clamp_rad = np.radians(phi_clamp)

    # Unloaded direction at B, used to build a zero-load starting state
    if mode == "pinned":
        phi0 = np.arctan2(*(B - A)[::-1])
    elif mode == "fixed":
        phi0 = phi_clamp_rad
    else:
        phi0 = phi_clamp_rad

    Lref = np.linalg.norm(C - B)
    wa = None if wall_angle is None else np.radians(wall_angle)
    la = None if load_angle is None else np.radians(load_angle)

    # Continuation: start from an exactly known solution and walk to the target.
    # The tip target, pre-curve, wall angle and friction are all ramped together,
    # so every intermediate problem is physically sensible. (Holding the wall
    # angle and friction at their final values while the tube is still straight
    # can create impossible, jammed intermediate states.)
    if start is None:
        # Unloaded straight tube: zero forces, straight-line lengths, and a wall
        # parallel to the tube (zero impingement) with no friction.
        # z = [phi_c, P, sCB, RB, sBA]
        C_from = B + Lref * np.array([np.cos(phi0), np.sin(phi0)])
        k_from = 0.0
        w_from = phi0
        mu_from = 0.0
        z = [phi0, 0.0, Lref]
        if mode != "cantilever":
            z += [0.0, np.linalg.norm(B - A)]
        z = np.array(z)
    else:
        C_from = np.asarray(start["C"], float)
        k_from = start["kappa0"]
        mu_from = start["mu"]
        # previous wall direction; for side contact that is the tube's tip angle
        w_from = start["wall_rad"] if start["wall_rad"] is not None else start["phiC"]
        z = start["z"].copy()

    for k in range(1, n_ramp + 1):
        f = k / n_ramp
        Ck = C_from + f * (C - C_from)
        kk = k_from + f * (kappa0 - k_from)
        muk = mu_from + f * (mu - mu_from)
        wk = None if wa is None else w_from + f * _wrap(wa - w_from)
        
        # lambda function, fun(zz), allows all other inputs to _shoot to be defined and 
        # ensure that the _shoot()[0] result are considered as the error
        fun = lambda zz: _shoot(zz, mode, A, B, Ck, phi_clamp_rad, EI, kk, muk,
                                steps, Lref, wk, la)[0]
        z, info, ier, msg = fsolve(fun, z, full_output=True, xtol=1e-12)
        if ier != 1:
            raise RuntimeError(
                f"Solver did not converge at ramp step {k}/{n_ramp}: {msg}\n"
                "With wall_angle and friction, persistent failure can mean the tip "
                "is genuinely jammed (impingement + friction angle >= 90 deg).")

    res, out = _shoot(z, mode, A, B, C, phi_clamp_rad, EI, kappa0, mu, steps, Lref, wa, la, record=True)
    out.update(
        mode=mode, z=z, C=C, kappa0=kappa0, mu=mu, wall_rad=wa,
        residual=np.max(np.abs(res)),
        P_normal=z[1],
        P=np.linalg.norm(out["fC"]),
        R_B=np.linalg.norm(out["fB"]),
        R_A=np.linalg.norm(out["fA"]),
        tip_angle_deg=np.degrees(out["phiC"]),
        impingement_deg=(None if wall_angle is None
                         else np.degrees(_wrap(wa - out["phiC"]))),
        arc_A_to_C=out["sBA"] + out["sCB"],
    )

    # Trachea, tube frame: bending part and axial part (+ = compressive)
    bend, along = _split(out["fC"], out["phiC"])
    out["C_bending"], out["C_axial"] = bend, -along
    # Trachea, wall frame. Without wall_angle the tube's side lies on the wall,
    # so the wall direction is the tube direction at C.
    if load_angle is None:
        wdir = wa if wall_angle is not None else out["phiC"]
        wn, wt = _split(out["fC"], wdir)
        out["C_wall_normal"], out["C_wall_shear"] = wn, abs(wt)
    else:
        out["C_wall_normal"] = out["C_wall_shear"] = None

    # Contact check at the trachea: the wall can only PUSH the tube away from
    # the tissue. The tissue side is taken as the side of the wall that the
    # straight, unbent tube would run into (the premise of the whole problem).
    # If the solved force points into the tissue, the tube would actually miss
    # the wall (e.g. its pre-curve already carries it clear) and the result is
    # not physical.
    if load_angle is None:
        tw = np.array([np.cos(wdir), np.sin(wdir)])
        ahead = B + 2 * Lref * np.array([np.cos(phi0), np.sin(phi0)])
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        tissue_side = np.sign(cross(tw, ahead - C))
        force_side = np.sign(cross(tw, out["fC"]))
        out["C_contact_ok"] = bool(tissue_side != 0 and force_side == -tissue_side)
    else:
        out["C_contact_ok"] = None

    # Supports
    bn, bt = _split(out["fB"], out["phiB"])
    out["B_normal"], out["B_tangential"] = bn, abs(bt)
    if mode == "cantilever":
        out["A_normal"] = None
        out["push"] = bt          # clamp at B supplies the axial force
    else:
        an, at = _split(out["fA"], out["phiA"])
        out["A_normal"], out["push"] = an, at
    return out


def sweep(A, B, C_list, mode, EI, wall_angles=None, **kw):
    """Solve a sequence of contact points (e.g. advancing insertion depth),
    continuing each solution from the previous one.

    wall_angles : wall direction in DEGREES (distal-pointing, same frame as the
                  points) for each point in C_list. Accepts
                    - None: no wall angle (side-contact mode) at every point
                    - a single number: the same wall angle at every point
                    - a sequence the same length as C_list; an entry may be None
                      to use side contact at that point
    Any other keyword (mu, kappa0, phi_clamp, n_ramp, ...) is passed to solve()
    and applies to every point.
    """
    if "wall_angle" in kw:
        raise ValueError("sweep() takes 'wall_angles' (one per point), not 'wall_angle'")
    n = len(C_list)
    if wall_angles is None or np.isscalar(wall_angles):
        wall_angles = [wall_angles] * n
    else:
        wall_angles = list(wall_angles)
        if len(wall_angles) != n:
            raise ValueError(f"wall_angles has {len(wall_angles)} entries but "
                             f"C_list has {n} points")

    results, prev = [], None
    for i, (C, w) in enumerate(zip(C_list, wall_angles)):
        try:
            r = solve(A, B, C, mode, EI, wall_angle=w, start=prev, **kw)
        except RuntimeError as e:
            raise RuntimeError(f"sweep failed at point {i} (C = {tuple(C)}, "
                               f"wall_angle = {w}): {e}") from e
        results.append(r)
        prev = r
    return results


def report(r):
    """Two-line summary: trachea, then supports."""
    if r["C_wall_normal"] is not None:
        tr = (f"Trachea: wall normal = {r['C_wall_normal']:5.2f} N, wall shear = "
              f"{r['C_wall_shear']:5.2f} N | tube bending = {r['C_bending']:5.2f} N, "
              f"axial = {r['C_axial']:5.2f} N")
    else:
        tr = (f"Trachea: resultant = {r['P']:5.2f} N | tube bending = "
              f"{r['C_bending']:5.2f} N, axial = {r['C_axial']:5.2f} N")
    tr += f" | tip angle = {r['tip_angle_deg']:6.1f} deg"
    if r.get("impingement_deg") is not None:
        tr += f", impingement = {r['impingement_deg']:5.1f} deg"
    if r["mode"] == "cantilever":
        sp = (f"Clamp at B: sideways = {r['B_normal']:5.2f} N, moment = "
              f"{abs(r['MA']):6.1f} N*mm | push = {r['push']:5.2f} N")
    else:
        sp = (f"Glottis A = {r['A_normal']:5.2f} N | arytenoid B = {r['B_normal']:5.2f} N"
              f" (friction {r['B_tangential']:4.2f}) | push = {r['push']:5.2f} N")
        if r["mode"] == "fixed":
            sp += f" | M_A = {abs(r['MA']):5.1f} N*mm"
    sp += f" | arc B->C = {r['sCB']:5.1f} mm"
    out = tr + "\n      " + sp
    if r.get("C_contact_ok") is False:
        out += ("\n      WARNING: tracheal force points into the tissue (wall would have "
                "to pull). The tube likely misses this contact point; result not physical.")
    return out


# ----------------------------------------------------------------------------
# Results as a pandas DataFrame
# ----------------------------------------------------------------------------
def results_to_df(results, include_shape=False):
    """Flatten a list of solve() results (e.g. from sweep()) into a DataFrame,
    one row per solved point. Also accepts a dict {mode: [results]}, in which
    case the per-mode tables are stacked into one.

    Units: lengths mm, forces N, moments N*mm, angles degrees.

    Columns
      identity    : mode, point (index within the sweep)
      inputs      : C_x, C_y, wall_angle_deg (NaN if none), mu, kappa0
      geometry    : s_BC, s_AB, s_AC (arc lengths), tip_angle_deg,
                    impingement_deg, phi_A_deg, phi_B_deg
      trachea     : P (resultant), C_wall_normal, C_wall_shear (tissue frame),
                    C_bending, C_axial (tube frame), fC_x, fC_y
      supports    : A_normal (glottis), B_normal, B_tangential, push,
                    R_A, R_B (resultant magnitudes), fA_x, fA_y, fB_x, fB_y
      moments     : M_prox = moment at the proximal end of the modelled tube
                    (at A for pinned/fixed, at the clamp B for cantilever)
      checks      : C_contact_ok, residual
    Entries that don't apply to a mode (e.g. A_normal for the cantilever) are NaN.
    include_shape=True adds a 'shape' column holding each deformed shape (N x 4
    array: x, y, angle, moment) so you can plot or post-process individual rows.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("results_to_df needs pandas: pip install pandas") from e

    # Also accept {mode: [results]} (the by_mode dict used for plotting), so
    # sweeps that were already run can be tabulated without solving again.
    if isinstance(results, dict):
        frames = [results_to_df(rs, include_shape=include_shape) for rs in results.values()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    nan = float("nan")
    f = lambda v: nan if v is None else float(v)
    rows = []
    for i, r in enumerate(results):
        row = {
            "mode": r["mode"], "point": i,
            "C_x": float(r["C"][0]), "C_y": float(r["C"][1]),
            "wall_angle_deg": nan if r["wall_rad"] is None else float(np.degrees(r["wall_rad"])),
            "mu": float(r["mu"]), "kappa0": float(r["kappa0"]),
            "s_BC": float(r["sCB"]), "s_AB": float(r["sBA"]), "s_AC": float(r["arc_A_to_C"]),
            "tip_angle_deg": float(r["tip_angle_deg"]),
            "impingement_deg": f(r["impingement_deg"]),
            "phi_A_deg": nan if r["phiA"] is None else float(np.degrees(r["phiA"])),
            "phi_B_deg": float(np.degrees(r["phiB"])),
            "P": float(r["P"]), "C_wall_normal": f(r["C_wall_normal"]),
            "C_wall_shear": f(r["C_wall_shear"]),
            "C_bending": float(r["C_bending"]), "C_axial": float(r["C_axial"]),
            "fC_x": float(r["fC"][0]), "fC_y": float(r["fC"][1]),
            "A_normal": f(r["A_normal"]), "B_normal": float(r["B_normal"]),
            "B_tangential": float(r["B_tangential"]), "push": float(r["push"]),
            "R_A": float(r["R_A"]), "R_B": float(r["R_B"]),
            "fA_x": float(r["fA"][0]), "fA_y": float(r["fA"][1]),
            "fB_x": float(r["fB"][0]), "fB_y": float(r["fB"][1]),
            "M_prox": float(r["MA"]),
            "C_contact_ok": r["C_contact_ok"], "residual": float(r["residual"]),
        }
        if include_shape:
            row["shape"] = r["shape"]
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_df(A, B, C_list, EI, modes=("cantilever", "pinned", "fixed"),
             include_shape=False, **kw):
    """Run sweep() for each mode and return one combined DataFrame.

    Takes the same keyword arguments as sweep() (wall_angles, mu, kappa0,
    phi_clamp, n_ramp, ...). If a mode fails to converge, it is skipped with a
    warning rather than losing the other modes' results.
    """
    import warnings
    import pandas as pd
    frames = []
    for m in modes:
        try:
            frames.append(results_to_df(sweep(A, B, C_list, m, EI, **kw),
                                        include_shape=include_shape))
        except RuntimeError as e:
            warnings.warn(f"mode '{m}' skipped: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
MODES = ("cantilever", "pinned", "fixed")


def plot_shapes(results_by_mode, A, B, filename="ett_shapes.png", wall_len=8.0):
    """Deformed tube shapes, one panel per mode, one curve per contact point.

    results_by_mode : {mode: [result, ...]} as returned by sweep()
    A, B            : glottis and arytenoid points (A may be None)
    wall_len        : length (mm) of the little line drawn along the wall at each C

    A short line through each contact point C shows the wall direction used for
    that point. Solid = the wall_angle you supplied. Dotted = no wall_angle was
    given, so the wall is assumed to lie along the tube's tip direction.
    C is the tube's centerline point, so the real wall surface sits about one
    tube radius (D/2) beyond this line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [m for m in MODES if m in results_by_mode]
    # fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4.5),
    #                          sharey=True, squeeze=False)
    fig, axes = plt.subplots(1, len(modes), figsize=(3 * len(modes), 4.5),
                             sharex = True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, mode in zip(axes, modes):
        for r in results_by_mode[mode]:
            C = r["C"]
            sh = r["shape"]
            (line,) = ax.plot(sh[:, 0], sh[:, 1], lw=1.5,
                              label=f"s_BC={r['sCB']:.0f} mm: P={r['P']:.1f} N")
            ax.plot(*C, "x", color="k")
            if r["wall_rad"] is not None:
                w, ls = r["wall_rad"], "-"
            else:
                w, ls = r["phiC"], ":"
            d = 0.5 * wall_len * np.array([np.cos(w), np.sin(w)])
            ax.plot([C[0] - d[0], C[0] + d[0]], [C[1] - d[1], C[1] + d[1]],
                    ls, color=line.get_color(), lw=2.5, solid_capstyle="round")
        ax.plot(*B, "o", color="tab:red", label="B (arytenoid)")
        if mode != "cantilever" and A is not None:
            ax.plot(*A, "s", color="tab:purple", label="A (glottis)")
        
        ax.set_title(mode)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (mm)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.05,1), loc='upper left')
        ax.set_xmargin(0.5)
    
    axes[0].set_ylabel("y (mm)")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    # plt.close(fig)
    return filename, fig, axes


def plot_force_vs_arc(results_by_mode, filename="ett_force_vs_arc.png"):
    """Tracheal force vs arc length from B to C (s_BC), one line per mode.

    Left : resultant tracheal force P.
    Right: wall normal force (solid) and wall shear (dashed), i.e. what the
           tissue feels. Normal/shear use the wall direction if a wall_angle was
           given, otherwise the tube's tip direction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"cantilever": "tab:blue", "pinned": "tab:orange", "fixed": "tab:green"}
    fig, ax  = plt.subplots(2, 2, figsize=(11, 11))
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]
    for mode in MODES:
        if mode not in results_by_mode:
            continue
        rs = results_by_mode[mode]
        s = [r["sCB"] for r in rs]
        c = colors[mode]
        ax1.plot(s, [r["P"] for r in rs], "o-", color=c, label=mode)
        if all(r["C_wall_normal"] is not None for r in rs):
            ax2.plot(s, [r["C_wall_normal"] for r in rs], "o-", color=c,
                     label=f"{mode}: normal")
            ax2.plot(s, [r["C_wall_shear"] for r in rs], "s--", color=c,
                     label=f"{mode}: shear")
        ax3.plot(s,[r["R_B"] for r in rs], "o-", color=c, label = mode)
        ax4.plot(s,[r["B_normal"] for r in rs], "o-", color=c, label=f'"{mode}: normal')
        ax4.plot(s,[r["B_tangential"] for r in rs], "s--", color=c, label=f'"{mode}: shear')
    ax1.set_title("Resultant Tracheal Force")
    ax2.set_title("Force on Tracheal Wall")
    ax3.set_title("Resultant Arytenoid Force")
    ax4.set_title("Force on Arytenoids")
    # for ax in (ax1, ax2, ax3, ax4):
    for ax in (ax1,ax3):
        ax.set_ylabel("force (N)")
    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in (ax3, ax4):
        ax.set_xlabel("arc length B to C, s_BC (mm)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        # ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    # plt.close(fig)
    axes = [ax1, ax2, ax3, ax4]
    return filename, fig, axes


# ----------------------------------------------------------------------------
# Demo and validation
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    EI = tube_EI(14.5, 10.2, 7.5)  # 7.5 mm ID tube, ~5454 N*mm^2
    span = 18.2                   # glottis -> arytenoid, mm

    # Your current depth table: a = distance beyond arytenoid, delta = offset.
    # Here they're placed in a frame with B at the origin and the unloaded
    # tube along +x, with the trachea deflecting the tip toward -y.
    # Replace with real CT coordinates when you have them.
    # rows = [(17.3, -1.43), (19.0, -2.0), (28.4, -6.9), (42.0, -18.3), (50.0, -29.0)]
    rows = [(-10.8, -18.39),(-11.8, -20.91),(-12.3, -30.25),(-8.7, -45.53),(-2, -54.58)]
    A = (-span, 0.0)
    A = (10.2, 15)
    B = (0.0, 0.0)
    Cs = [(a, d) for a, d in rows]

    # # --- Validation 1: cantilever at B vs. closed-form elastica integral ---
    # from scipy.integrate import quad
    # from scipy.optimize import brentq

    # def closed_form(X, Y):
    #     def J(th, f):
    #         g = lambda t: (2 * t * f(th - t * t) / np.sqrt(np.sin(th) - np.sin(th - t * t))
    #                        if t > 0 else 2 * f(th) / np.sqrt(np.cos(th)))
    #         return quad(g, 0, np.sqrt(th))[0]
    #     th = brentq(lambda t: J(t, np.sin) / (2 * np.sqrt(np.sin(t))) - Y / X, 1e-6, np.radians(89.9))
    #     return 2 * EI * np.sin(th) / X**2

    # print("Validation 1: cantilever clamped at arytenoid, fixed vertical load,")
    # print("              vs closed-form elastica integral")
    # for C in Cs:
    #     r = solve(None, B, C, "cantilever", EI, phi_clamp=0.0, load_angle=90.0)
    #     print(f"  a={C[0]:5.1f}, d={-C[1]:5.2f}: shooting {r['P']:.3f} N, "
    #           f"closed form {closed_form(C[0], -C[1]):.3f} N")

    # # --- Validation 2: small deflection vs linear beam formulas ---
    # print("\nValidation 2: small deflection (d = 0.1 mm) vs linear formulas")
    # a, d = 28.4, 0.1
    # lin = {
    #     "cantilever": 3 * EI * d / a**3,
    #     "pinned": 3 * EI * d / (a**2 * (span + a)),
    #     "fixed": 12 * EI * d / (a**2 * (3 * span + 4 * a)),
    # }
    # for mode in ("cantilever", "pinned", "fixed"):
    #     r = solve(A, B, (a, -d), mode, EI, phi_clamp=0.0)
    #     print(f"  {mode:>10s}: shooting {r['P']:.4f} N, linear {lin[mode]:.4f} N")

    # # --- Results for the depth table ---
    # print("\nDepth table (frictionless, straight tube):")
    # for mode in ("cantilever", "pinned", "fixed"):
    #     print(f"\n{mode}:")
    #     for C, r in zip(Cs, sweep(A, B, Cs, mode, EI, phi_clamp=0.0)):
    #         print(f"  a={C[0]:5.1f} d={-C[1]:5.2f}\n    " + report(r))

    # --- Example: tip impinging on the wall, with friction and pre-curve ---
    # print("\nExample: pinned, tip pressing on a wall at -40 deg, R = 140 mm, friction sweep")
    # for mu in (0.0, 0.1, 0.3, 0.5):
    #     r = solve(A, B, (28.4, -6.9), "pinned", EI, wall_angle=-40.0, mu=mu, kappa0=-1/140)
    #     print(f"  mu = {mu}\n    " + report(r))
    
    # --- Example: increasing friction
    mu = [0.1, 0.3, 0.5, 0.7]
    wall_angle = -87.1
    A = (10.2, 15)
    B = (0.0, 0.0)
    C = (-12.3, -30.25)
    res = []
    for m in mu:
        r = solve(A, B, C, "fixed", EI, wall_angle=wall_angle, mu=m, kappa0=-1/140)
        res.append(r)
        
    nan = float("nan")
    f = lambda v: nan if v is None else float(v)
    rows = []
    for i, r in enumerate(res):
        row = {
            "mode": r["mode"], "point": i,
            "C_x": float(r["C"][0]), "C_y": float(r["C"][1]),
            "wall_angle_deg": nan if r["wall_rad"] is None else float(np.degrees(r["wall_rad"])),
            "mu": float(r["mu"]), "kappa0": float(r["kappa0"]),
            "s_BC": float(r["sCB"]), "s_AB": float(r["sBA"]), "s_AC": float(r["arc_A_to_C"]),
            "tip_angle_deg": float(r["tip_angle_deg"]),
            "impingement_deg": f(r["impingement_deg"]),
            "phi_A_deg": nan if r["phiA"] is None else float(np.degrees(r["phiA"])),
            "phi_B_deg": float(np.degrees(r["phiB"])),
            "P": float(r["P"]), "C_wall_normal": f(r["C_wall_normal"]),
            "C_wall_shear": f(r["C_wall_shear"]),
            "C_bending": float(r["C_bending"]), "C_axial": float(r["C_axial"]),
            "fC_x": float(r["fC"][0]), "fC_y": float(r["fC"][1]),
            "A_normal": f(r["A_normal"]), "B_normal": float(r["B_normal"]),
            "B_tangential": float(r["B_tangential"]), "push": float(r["push"]),
            "R_A": float(r["R_A"]), "R_B": float(r["R_B"]),
            "fA_x": float(r["fA"][0]), "fA_y": float(r["fA"][1]),
            "fB_x": float(r["fB"][0]), "fB_y": float(r["fB"][1]),
            "M_prox": float(r["MA"]),
            "C_contact_ok": r["C_contact_ok"], "residual": float(r["residual"]),
        }
        rows.append(row)
        
    

    # --- Plots (optional) ---
    # Wall directions traced from CT, one per contact point (degrees,
    # pointing distally, same frame as the points).
    wall_angles = [-21.3, -24.0, -32.8, -47.1, -56.4]
    wall_angles = [-112, -103, -87.1, -69.5, -67]
    try:
        by_mode = {m: sweep(A, B, Cs, m, EI, wall_angles=wall_angles, mu=0.4,
                            phi_clamp=-124, kappa0=-1/140) for m in MODES}
        df = results_to_df(by_mode)   # reuse the sweeps above; no second solve
        df.to_csv("ett_sweep_results.csv", index=False)
        print("\nSaved ett_sweep_results.csv")
        print(df[["mode", "point", "s_BC", "impingement_deg", "P", "C_wall_normal",
                  "C_wall_shear", "push", "A_normal", "B_normal"]].round(2).to_string())
        
        filename, fig, ax = plot_shapes(by_mode, A, B)
        filename2, fig2, ax2 = plot_force_vs_arc(by_mode)
        print("\nSaved" + filename)
        print("Saved" + filename2)
        
    except ImportError:
        pass
    
    import matplotlib.pyplot as plt
    fig3, ax3 = plt.subplots()
    for line in ax[1].get_lines():
        ax3.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker = line.get_marker(), label = line.get_label())
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.4)
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title('Pinned')
    # ax3.legend(bbox_to_anchor=(0.5,0.7),loc='upper left')
    ax3.set_xmargin(0.5)
    