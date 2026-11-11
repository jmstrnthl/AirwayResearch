# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 03:13:10 2026

@author: jmstr
"""

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

Spring at B (optional, kB in N/mm; None = rigid support):
    B may translate only perpendicular to the ORIGINAL A->B line (for a cantilever
    with no A, perpendicular to the clamp direction). With fB the force the support
    exerts on the tube and nB the unit vector perpendicular to A->B, the spring gives
        u = -(fB . nB) / kB        (B moves away from the tube's push)
        B_final = B + u * nB
    and the two position residuals at B compare the integrated tube position with
    B_final instead of with the fixed B. For the cantilever the clamp is on a
    translational spring only (its angle is still phi_clamp).

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

#%% Core Functions
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
           wall_angle=None, load_angle=None, kB=None, nB=None, record=False):
    """
    z = [phi_c, P, sCB, RB, sBA]

    kB, nB : optional translational spring at B. kB is the stiffness (N/mm) and
             nB the unit vector B may move along (perpendicular to the original
             A->B line). kB=None means a rigid support at B.
    """
    phiC, P, sCB = z[0], z[1], z[2]
    state = np.array([C[0], C[1], phiC, 0.0])
    fC = _load_at_C(P, phiC, mu, wall_angle, load_angle)
    n = fC.copy()
    rec = [state.copy()] if record else None

    state = _integrate(state, sCB, n, EI, kappa0, steps, rec)

    # Force the support at B exerts on the tube. It is known before the position
    # residual is formed, because it depends only on the guesses and on the tube
    # angle at B reached by the integration.
    if mode == "cantilever":
        fB = -fC
    else:
        fB = _contact_force(z[3], state[2], mu)

    # Where B actually is: its original position, or displaced along nB by the
    # spring (the tube pushes B with -fB, so B moves in the direction of -fB).
    if kB is None:
        B_t, uB = B, 0.0
    else:
        uB = -(fB @ nB) / kB
        B_t = B + uB * nB

    res = [(state[0] - B_t[0]) / Lref, (state[1] - B_t[1]) / Lref]

    out = {"fC": fC, "phiC": phiC, "phiB": state[2], "MB": state[3], "sCB": sCB,
           "B_final": B_t, "u_B": uB}

    if mode == "cantilever":
        res.append(_wrap(state[2] - phi_clamp))
        # clamp at B: reaction force and moment reported under the B / A keys
        out.update(fB=fB, fA=np.zeros(2), MA=state[3], phiA=None, sBA=0.0)
    else:
        sBA = z[4]
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

#%% Public Solver
# ----------------------------------------------------------------------------
# Public solver
# ----------------------------------------------------------------------------
def solve(A, B, C, mode, EI, phi_clamp=None, kappa0=0.0, mu=0.0,
          wall_angle=None, load_angle=None, kB=None, n_ramp=20, steps=50, start=None):
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
    kB        : stiffness (N/mm) of a translational spring at B, or None (default)
                for a rigid support. B moves only perpendicular to the original
                A->B line (perpendicular to the clamp direction if A is None).
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

    # Spring at B: direction B may move along = perpendicular to the original A->B line
    if kB is not None and not np.isfinite(kB):
        kB = None                      # infinite stiffness = rigid
    if kB is not None:
        if kB <= 0:
            raise ValueError("kB must be positive (N/mm), or None for a rigid support")
        if A is not None:
            tAB = B - A
        else:
            tAB = np.array([np.cos(phi_clamp_rad), np.sin(phi_clamp_rad)])
        tAB = tAB / np.linalg.norm(tAB)
        nB = np.array([-tAB[1], tAB[0]])
    else:
        nB = None

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
                                steps, Lref, wk, la, kB, nB)[0]
        z, info, ier, msg = fsolve(fun, z, full_output=True, xtol=1e-12)
        if ier != 1:
            raise RuntimeError(
                f"Solver did not converge at ramp step {k}/{n_ramp}: {msg}\n"
                "With wall_angle and friction, persistent failure can mean the tip "
                "is genuinely jammed (impingement + friction angle >= 90 deg).")

    res, out = _shoot(z, mode, A, B, C, phi_clamp_rad, EI, kappa0, mu, steps, Lref, wa, la,
                      kB, nB, record=True)
    out.update(
        mode=mode, z=z, C=C, kappa0=kappa0, mu=mu, wall_rad=wa,
        B0=B, kB=kB, nB=nB,
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


def sweep_path(A, B, C_list, mode, EI, wall_angles=None, n_steps=8,
               n_ramp=5, n_ramp_inc=1, **kw):
    """Advance the tip through the contact points in small increments and keep
    EVERY solve (full result dictionaries, including the deformed shapes).

    C_list      : contact points from the CT, in order of advancing depth (>= 2)
    wall_angles : wall direction (DEGREES) at each point; None, a single number,
                  or one per point (all numbers, or all None)
    n_steps     : increments between consecutive contact points. The tip position
                  and wall angle are interpolated linearly, i.e. the tip is assumed
                  to travel in a straight line between the CT points.
    n_ramp      : continuation steps for the first solve (starts from the unloaded tube)
    n_ramp_inc  : continuation steps for every later solve; each one starts from the
                  previous solution, so a small value is enough
    **kw        : passed to solve() (mu, kappa0, phi_clamp, kB, steps, ...)

    Returns a list of result dictionaries, one per increment. Each also carries
        step      : increment index along the dense path
        is_key    : True for the original CT contact points
        tip_path  : distance (mm) the tip has travelled since the first point
    Use results_to_df(results) to tabulate them. An increment that fails to
    converge is skipped with a warning (the next one continues from the last
    good solution).
    """
    import warnings
    if "wall_angle" in kw:
        raise ValueError("sweep_path() takes 'wall_angles', not 'wall_angle'")
    C_arr = [np.asarray(C, float) for C in C_list]
    n = len(C_arr)
    if n < 2:
        raise ValueError("sweep_path needs at least two contact points")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    if wall_angles is None:
        w_list = [None] * n
    elif np.isscalar(wall_angles):
        w_list = [float(wall_angles)] * n
    else:
        w_list = list(wall_angles)
        if len(w_list) != n:
            raise ValueError(f"wall_angles has {len(w_list)} entries but C_list has {n} points")
        if any(w is None for w in w_list) and not all(w is None for w in w_list):
            raise ValueError("wall_angles must be all numbers or all None")

    # Dense path: (tip position, wall angle, is_key, segment index)
    dense = []
    for j in range(n - 1):
        dw = None if w_list[j] is None else (w_list[j + 1] - w_list[j] + 180.0) % 360.0 - 180.0
        for k in range(n_steps):
            t = k / n_steps
            C = (1 - t) * C_arr[j] + t * C_arr[j + 1]
            w = None if dw is None else w_list[j] + t * dw
            dense.append((C, w, k == 0, j))
    dense.append((C_arr[-1], w_list[-1], True, n - 1))

    results, prev, prevC, tip_path = [], None, None, 0.0
    for idx, (C, w, is_key, seg) in enumerate(dense):
        if prevC is not None:
            tip_path += float(np.linalg.norm(C - prevC))
        prevC = C
        try:
            r = solve(A, B, C, mode, EI, wall_angle=w, start=prev,
                      n_ramp=(n_ramp if prev is None else n_ramp_inc), **kw)
        except RuntimeError as e:
            warnings.warn(f"sweep_path: increment {idx} (C = {tuple(np.round(C, 2))}, "
                          f"wall_angle = {w}) did not converge and was skipped: {e}")
            continue
        r.update(step=idx, is_key=is_key, key_segment=seg, tip_path=tip_path)
        results.append(r)
        prev = r
    if not results:
        raise RuntimeError("sweep_path: no increment converged")
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
    if r.get("kB") is not None:
        sp += f" | B moved {r['u_B']:+.2f} mm (kB = {r['kB']:g} N/mm)"
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
      identity    : mode, point (row index within the sweep), step (increment index
                    along a sweep_path; equals point otherwise)
      inputs      : C_x, C_y, wall_angle_deg (NaN if none), mu, kappa0
      geometry    : s_BC, s_AB, s_AC (arc lengths), tip_angle_deg,
                    impingement_deg, phi_A_deg, phi_B_deg
      trachea     : P (resultant), C_wall_normal, C_wall_shear (tissue frame),
                    C_bending, C_axial (tube frame), fC_x, fC_y
      supports    : A_normal (glottis), B_normal, B_tangential, push,
                    R_A, R_B (resultant magnitudes), fA_x, fA_y, fB_x, fB_y
      B spring    : kB (N/mm, NaN = rigid), B_disp (signed displacement of B along
                    the unit normal to the original A->B line, mm), B_final_x/y
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
            "mode": r["mode"], "point": i, "step": int(r.get("step", i)),
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
            "is_key": bool(r.get("is_key", False)), "tip_path": f(r.get("tip_path")),
            "kB": nan if r["kB"] is None else float(r["kB"]),
            "B_disp": float(r["u_B"]),
            "B_final_x": float(r["B_final"][0]), "B_final_y": float(r["B_final"][1]),
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

#%% Plotting Helpers
# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
MODES = ("cantilever", "pinned", "fixed")

def plot_setup(A, B, Cs, wall_angles, wall_len=10.0, degrees=True, phi_clamp=None,
               kappa0=None, ax=None):
    """
    Plot the supports, contact points and wall directions before a run, and
    check that every direction points the way the solver expects.
 
    Parameters
    ----------
    A, B : (x, y) in mm. A = glottis/stylet tip, B = arytenoid. A may be None
           (cantilever); then phi_clamp is needed to know the tube direction.
    Cs : list of (x, y) in mm. Prescribed tip/contact points, in order of depth.
    wall_angles : one per C, measured counter-clockwise from +x, POINTING
           DISTALLY (the way the tube travels). None = no wall at that point.
    wall_len : length (mm) of the wall arrow drawn at each C.
    degrees : True (default) if wall_angles and phi_clamp are in degrees.
    phi_clamp : optional clamp angle, same units, as you will pass to solve().
           Drawn at A (or at B if A is None) and checked against A->B.
    kappa0 : optional natural (pre-)curvature in 1/mm, same sign convention as
           solve(): positive = curving counter-clockwise as the tube advances.
           0 draws a straight tube; None (default) draws nothing.
    ax : optional matplotlib axes to draw on.
 
    What is drawn
    -------------
    - Grey arrow A->B: the tube's direction of travel through the larynx.
    - Dotted line B->C: chord to each contact point (approximate tube direction).
    - Solid arrow at C: the wall direction you supplied. It should point
      roughly the same way as the chord (distally).
    - Dashed arrow at C: wall normal, pointing into the lumen (the way the
      wall pushes the tube).
    - Orange arrow: phi_clamp, if given.
    - Dashed brown curve: the tube's unloaded shape for kappa0, starting at A
      (B if A is None) along phi_clamp if given, otherwise along A->B. This is
      where the tube would go with no tissue in the way, so the gap between
      it and each C is the deflection the tissue has to impose. A good sign
      check: the R = 140 mm pre-curve should bend the way the manufactured
      tube bends in the airway.
 
    Checks (shown in red on the plot and issued as warnings)
    --------------------------------------------------------
    - A wall arrow pointing back toward the mouth (more than 90 deg from the
      chord). Fix: subtract 180 deg.
    - Approximate impingement (wall angle minus chord angle) of 60 deg or more,
      where the tip is close to jamming. The real impingement uses the solved
      tip angle, so treat this as a rough screen.
    - phi_clamp more than 90 deg from the A->B direction.
 
    Returns fig, ax, problems (a list of strings; empty if everything passed).
    """
    import matplotlib.pyplot as plt
    import warnings
 
    to_rad = (lambda v: np.radians(v)) if degrees else (lambda v: float(v))
    from_rad = (lambda v: np.degrees(v)) if degrees else (lambda v: v)
    unit_s = "°" if degrees else " rad"
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    unit = lambda a: np.array([np.cos(a), np.sin(a)])
 
    Cs = [np.asarray(c, dtype=float) for c in Cs]
    if len(wall_angles) != len(Cs):
        raise ValueError(f"{len(Cs)} contact points but {len(wall_angles)} wall angles")
    A = None if A is None else np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A is None and phi_clamp is None:
        raise ValueError("Pass phi_clamp when A is None (cantilever).")
 
    # Direction the tube travels through the larynx
    travel = np.arctan2(*(B - A)[::-1]) if A is not None else to_rad(phi_clamp)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
    else:
        fig = ax.figure
    problems = []
 
    def arrow(p, v, **kw):
        ax.annotate("", xy=p + v, xytext=p, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, **kw))
 
    # Supports and direction of travel
    ax.plot(*B, "k^", ms=10, zorder=5)
    ax.annotate("B", B, xytext=(6, 8), textcoords="offset points", fontweight="bold")
    if A is not None:
        ax.plot(*A, "ks", ms=9, zorder=5)
        ax.annotate("A", A, xytext=(6, 8), textcoords="offset points", fontweight="bold")
        arrow(A, B - A, color="0.6", lw=2.5)
        ax.annotate(f"travel {from_rad(travel):.1f}{unit_s}", (A + B) / 2,
                    xytext=(0, 8), textcoords="offset points", color="0.4", fontsize=8)
 
    # Clamp direction
    if phi_clamp is not None:
        pc = to_rad(phi_clamp)
        at = A if A is not None else B
        arrow(at, unit(pc) * wall_len * 1.5, color="tab:orange", lw=2)
        msg = f"clamp {phi_clamp:g}{unit_s}"
        off = np.degrees(abs(wrap(pc - travel)))
        if off > 90:
            msg += "  ✗ points backwards"
            problems.append(f"phi_clamp = {phi_clamp:g}{unit_s} is {off:.0f}° from the "
                            f"A->B direction ({from_rad(travel):.1f}{unit_s}). Use "
                            "phi_clamp=None to clamp along A->B.")
        ax.annotate(msg, at, xytext=(6, -14), textcoords="offset points",
                    color="red" if off > 90 else "tab:orange", fontsize=8,
                    fontweight="bold")
 
    # Unloaded (pre-curved) tube shape
    if kappa0 is not None:
        start = A if A is not None else B
        th0 = to_rad(phi_clamp) if phi_clamp is not None else travel
        reach = max(np.linalg.norm(C - start) for C in Cs) if Cs else 50.0
        s_arc = np.linspace(0.0, 1.15 * reach, 300)
        th = th0 + kappa0 * s_arc
        ds = np.diff(s_arc, prepend=0.0)
        xy = start + np.cumsum(np.column_stack([np.cos(th), np.sin(th)]) * ds[:, None], axis=0)
        ax.plot(xy[:, 0], xy[:, 1], "--", color="saddlebrown", lw=1.8, zorder=2)
        if kappa0 == 0:
            txt = "unloaded tube: straight"
        else:
            turn = "CCW" if kappa0 > 0 else "CW"
            txt = f"unloaded tube: R = {1 / abs(kappa0):.0f} mm, curves {turn}"
        ax.annotate(txt, xy[-1], xytext=(6, 6), textcoords="offset points",
                    color="saddlebrown", fontsize=8, fontweight="bold")
 
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(Cs)))
    for i, (C, ang, col) in enumerate(zip(Cs, wall_angles, colors)):
        chord = np.arctan2(*(C - B)[::-1])
        ax.plot([B[0], C[0]], [B[1], C[1]], ":", color=col, lw=1)
        ax.plot(*C, "o", color=col, ms=7, zorder=5)
        label = f"C{i + 1}"
 
        if ang is None:
            ax.annotate(label + ": no wall", C, xytext=(8, -12),
                        textcoords="offset points", fontsize=8, color=col)
            continue
 
        a = to_rad(ang)
        t = unit(a)
        imp = np.degrees(wrap(a - chord))
        bad_dir = abs(imp) > 90
        near_jam = not bad_dir and abs(imp) >= 60
        c_wall = "red" if (bad_dir or near_jam) else col
 
        # wall drawn as an arrow centred on C, pointing in the declared direction
        arrow(C - t * wall_len / 2, t * wall_len, color=c_wall, lw=2.5)
 
        # wall normal toward the lumen (the side the supports are on)
        n = np.array([-t[1], t[0]])
        if np.dot(n, B - C) < 0:
            n = -n
        ax.annotate("", xy=C + n * wall_len * 0.6, xytext=C, zorder=4,
                    arrowprops=dict(arrowstyle="->", color=col, ls="--"))
 
        label += f": wall {ang:g}{unit_s}, imp ≈ {imp:.0f}°"
        if bad_dir:
            fix = from_rad(wrap(a + np.pi))
            label += "  ✗ points proximally"
            problems.append(f"C{i + 1}: wall_angle {ang:g}{unit_s} points back toward "
                            f"the mouth ({abs(imp):.0f}° from the tube direction). "
                            f"Use {fix:.1f}{unit_s}.")
        elif near_jam:
            label += "  ! near jamming"
            problems.append(f"C{i + 1}: impingement ≈ {abs(imp):.0f}°; the tip jams once "
                            "impingement + atan(mu) reaches 90°.")
        ax.annotate(label, C, xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=c_wall)
 
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if problems:
        ax.set_title(f"Setup check: {len(problems)} problem(s), see red labels",
                     color="red")
        for p in problems:
            warnings.warn("plot_setup: " + p, stacklevel=2)
    else:
        ax.set_title("Setup check: directions OK")
    ax.margins(0.15)
    return fig, ax, problems

def plot_shapes(results_by_mode, A, B, filename="ett_shapes.png", wall_len=8.0,
                orientation="horizontal"):
    """Deformed tube shapes, one panel per mode, one curve per contact point.
 
    results_by_mode : {mode: [result, ...]} as returned by sweep()
    A, B            : glottis and arytenoid points (A may be None); B is the
                      ORIGINAL (unloaded) position of the arytenoid support
    wall_len        : length (mm) of the little line drawn along the wall at each C
    orientation     : how the model panels are arranged
                      "horizontal" (default) = side by side in a row
                      "vertical"             = stacked in a column
                      The returned axes are a flat array (cantilever, pinned, fixed
                      order) in both cases.
 
    If the results were solved with a spring at B (kB), B's original position is
    drawn as a hollow red circle and its final (displaced) position for each contact
    point as a diamond in that curve's colour, joined to the original by a thin line.
 
    A short line through each contact point C shows the wall direction used for
    that point. Solid = the wall_angle you supplied. Dotted = no wall_angle was
    given, so the wall is assumed to lie along the tube's tip direction.
    C is the tube's centerline point, so the real wall surface sits about one
    tube radius (D/2) beyond this line.
    """
    # import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    modes = [m for m in MODES if m in results_by_mode]
    # fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4.5),
    #                          sharey=True, squeeze=False)
    n = len(modes)
    orientation = str(orientation).lower()
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    vertical = orientation == "vertical"
    if vertical:
        nrows, ncols, figsize = n, 1, (10, 3 * n)
    else:
        nrows, ncols, figsize = 1, n, (3 * n, 4.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, mode in zip(axes, modes):
        spring = any(r.get("kB") is not None for r in results_by_mode[mode])
        shown_B_final = False
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
            if r.get("kB") is not None:
                Bf = r["B_final"]
                ax.plot([B[0], Bf[0]], [B[1], Bf[1]], "-", color=line.get_color(),
                        lw=0.8, alpha=0.7)
                ax.plot(*Bf, "D", color=line.get_color(), mec="k", ms=5,
                        label="B final" if not shown_B_final else None)
                shown_B_final = True
        if spring:
            ax.plot(*B, "o", mfc="none", mec="tab:red", mew=1.8, ms=8, label="B initial")
        else:
            ax.plot(*B, "o", color="tab:red", label="B (arytenoid)")
        if mode != "cantilever" and A is not None:
            ax.plot(*A, "s", color="tab:purple", label="A (glottis)")
        
        ax.set_title(mode)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (mm)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.05,1), loc='upper left')
        ax.set_xmargin(0.5)
    
    if vertical:
        for ax in axes:
            ax.set_ylabel("y (mm)")          # every row has its own y axis
        for ax in axes[:-1]:
            ax.set_xlabel("")                # x label only on the bottom panel
    else:
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
        ax4.plot(s,[r["B_normal"] for r in rs], "o-", color=c, label=f"{mode}: normal")
        ax4.plot(s,[r["B_tangential"] for r in rs], "s--", color=c, label=f"{mode}: shear")
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


def plot_friction_sweep(results_by_mode, filename="ett_friction_sweep.png"):
    """Tracheal forces vs friction coefficient mu, one line per mode.

    results_by_mode : {mode: [result, ...]} where each result was solved at a
                      different mu (same contact point, wall angle and curvature).
    Panels: resultant tracheal force P, wall normal force, wall shear force,
    and the component along the tube (axial, + = compressive).
    """
    import matplotlib.pyplot as plt

    colors = {"cantilever": "tab:blue", "pinned": "tab:orange", "fixed": "tab:green"}
    panels = [("P", "Resultant tracheal force"),
              ("C_wall_normal", "Normal force on tracheal wall"),
              ("C_wall_shear", "Shear (friction) force on tracheal wall"),
              ("C_axial", "Axial force along tube tip (+ = compressive)")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, (key, title) in zip(axs.flat, panels):
        for mode in MODES:
            if mode not in results_by_mode:
                continue
            rs = results_by_mode[mode]
            ax.plot([r["mu"] for r in rs], [r[key] for r in rs], "o-",
                    color=colors[mode], label=mode)
        ax.set_title(title)
        ax.set_ylabel("force (N)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("friction coefficient, mu")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    return filename, fig, axs


def plot_amplification_vs_mu(results_by_mode, filename="ett_amplification_vs_mu.png"):
    """Force amplification vs friction coefficient mu, one line per mode.

    Amplification = wall normal force / force that bends the tube. It is how
    much harder the wall has to push because the tip meets it at an angle and
    friction tilts the wall force. Dots are the solver results; dashed lines are
    the closed-form relation 1 / (cos(psi) - mu*sin(psi)) using each point's
    own impingement angle psi, which should land on the dots.
    The amplification diverges as psi + arctan(mu) approaches 90 degrees (jamming).
    """
    import matplotlib.pyplot as plt

    colors = {"cantilever": "tab:blue", "pinned": "tab:orange", "fixed": "tab:green"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mode in MODES:
        if mode not in results_by_mode:
            continue
        rs = results_by_mode[mode]
        mus = [r["mu"] for r in rs]
        amp = [r["C_wall_normal"] / r["C_bending"] for r in rs]
        psi = [np.radians(abs(r["impingement_deg"])) for r in rs]
        theory = [1.0 / (np.cos(p) - m * np.sin(p)) for p, m in zip(psi, mus)]
        ax.plot(mus, amp, "o", color=colors[mode], label=f"{mode} (solver)")
        ax.plot(mus, theory, "--", color=colors[mode], lw=1,
                label=f"{mode}: 1/(cos psi - mu sin psi)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("friction coefficient, mu")
    ax.set_ylabel("amplification (wall normal / bending force)")
    ax.set_title("Force amplification vs friction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    return filename, fig, ax


def _select_mode_results(results, mode):
    """Accept a list of results (one model) or {mode: [results]} plus a mode name."""
    if isinstance(results, dict):
        if mode is None:
            if len(results) == 1:
                mode = next(iter(results))
            else:
                raise ValueError("results holds several models; pass "
                                 "mode='cantilever', 'pinned' or 'fixed'")
        if mode not in results:
            raise KeyError(f"mode '{mode}' not in results (have {list(results)})")
        return results[mode], mode
    return results, (mode or results[0]["mode"])


def _advance_figure():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])
    axs = (fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.09,
                        wspace=0.22, hspace=0.38)
    return fig, axs


def _advance_setup(results, A, B, mode, on_tissue, show_A, arrow_frac=0.2):
    """Fixed axis limits and arrow scale for every frame of an advance plot/GIF."""
    sgn = -1.0 if on_tissue else 1.0
    use_A = show_A and A is not None and mode != "cantilever"
    geo = [np.atleast_2d(B)]
    if A is not None:
        geo.append(np.atleast_2d(A))
    for r in results:
        geo += [r["shape"][:, :2], np.atleast_2d(r["C"]), np.atleast_2d(r["B_final"])]
    G = np.vstack(geo)
    span = float((G.max(0) - G.min(0)).max())
    fmax = max(max(np.linalg.norm(r["fC"]), np.linalg.norm(r["fB"]),
                   np.linalg.norm(r["fA"]) if use_A else 0.0) for r in results) or 1.0
    scale = arrow_frac * span / fmax              # mm of arrow per N
    tips = []
    for r in results:
        tips += [r["C"] + sgn * scale * r["fC"], r["B_final"] + sgn * scale * r["fB"]]
        if use_A:
            tips.append(np.asarray(A, float) + sgn * scale * r["fA"])
    G = np.vstack([G, np.array(tips)])
    pad = 0.07 * span
    lo, hi = G.min(0) - pad, G.max(0) + pad
    return dict(sgn=sgn, scale=scale, use_A=use_A,
                xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))


def _arrow(ax, tail, vec, color, label):
    """Arrow from `tail` along `vec` (data units), labelled just beyond its tip."""
    tail = np.asarray(tail, float)
    vec = np.asarray(vec, float)
    tip = tail + vec
    ax.annotate("", xy=tip, xytext=tail, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                shrinkA=0, shrinkB=0))
    L = float(np.linalg.norm(vec))
    u = vec / L if L > 0 else np.array([1.0, 0.0])
    ax.annotate(label, xy=tip, xytext=(9 * u[0], 9 * u[1]), textcoords="offset points",
                ha="left" if u[0] >= 0 else "right", va="bottom" if u[1] >= 0 else "top",
                color=color, fontsize=8, fontweight="bold", zorder=7)


def _draw_advance_frame(axs, results, i, A, B, mode, setup, wall_len=8.0):
    """Draw frame i: beam shape with force arrows, plus force traces vs s_BC."""
    ax_s, ax_t, ax_a = axs
    for ax in axs:
        ax.clear()
    r = results[i]
    sgn, scale = setup["sgn"], setup["scale"]
    spring = r.get("kB") is not None
    c_tr, c_ar, c_gl = "#c0392b", "#1e8449", "#7d3c98"

    # ---------------- beam shape panel ----------------
    for rr in results:                                  # faint history of all shapes
        ax_s.plot(rr["shape"][:, 0], rr["shape"][:, 1], color="0.85", lw=0.8, zorder=1)
    Cp = np.array([rr["C"] for rr in results])
    ax_s.plot(Cp[:, 0], Cp[:, 1], ":", color="0.55", lw=1, zorder=1)
    keys = np.array([rr["C"] for rr in results if rr.get("is_key")])
    if len(keys):
        ax_s.plot(keys[:, 0], keys[:, 1], "o", mfc="none", mec="0.45", ms=4, zorder=2)

    sh = r["shape"]
    ax_s.plot(sh[:, 0], sh[:, 1], color="tab:blue", lw=2.6, zorder=4)
    C = np.asarray(r["C"], float)
    w, ls = (r["wall_rad"], "-") if r["wall_rad"] is not None else (r["phiC"], ":")
    d = 0.5 * wall_len * np.array([np.cos(w), np.sin(w)])
    ax_s.plot([C[0] - d[0], C[0] + d[0]], [C[1] - d[1], C[1] + d[1]], ls,
              color="k", lw=2.2, solid_capstyle="round", zorder=3)
    ax_s.plot(*C, "x", color="k", zorder=5)

    Bf = np.asarray(r["B_final"], float)
    if spring:
        ax_s.plot(*B, "o", mfc="none", mec="tab:red", mew=1.8, ms=9, zorder=3)
        ax_s.plot([B[0], Bf[0]], [B[1], Bf[1]], "-", color="tab:red", lw=0.8, zorder=3)
        ax_s.plot(*Bf, "D", color="tab:red", mec="k", ms=6, zorder=5)
    else:
        ax_s.plot(*B, "o", color="tab:red", ms=7, zorder=5)
    if A is not None and mode != "cantilever":
        ax_s.plot(*A, "s", color="tab:purple", ms=7, zorder=5)

    _arrow(ax_s, C, sgn * scale * r["fC"], c_tr, f"{np.linalg.norm(r['fC']):.1f} N")
    _arrow(ax_s, Bf, sgn * scale * r["fB"], c_ar, f"{np.linalg.norm(r['fB']):.1f} N")
    if setup["use_A"]:
        _arrow(ax_s, A, sgn * scale * r["fA"], c_gl, f"{np.linalg.norm(r['fA']):.1f} N")

    ax_s.set_xlim(*setup["xlim"]); ax_s.set_ylim(*setup["ylim"])
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.grid(alpha=0.3)
    ax_s.set_xlabel("x (mm)"); ax_s.set_ylabel("y (mm)")
    who = "tube on tissue" if sgn < 0 else "tissue on tube"
    ax_s.text(0.02, 0.99, f"arrows: force of {who}\n(10 N = {10 * scale:.0f} mm)\n"
              "red = trachea, green = arytenoid", transform=ax_s.transAxes,
              fontsize=7, va="top", ha="left",
              bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2))
    title = f"{mode}  |  s_BC = {r['sCB']:.1f} mm"
    if spring:
        title += f"  |  B moved {r['u_B']:+.2f} mm"
    if r.get("C_contact_ok") is False:
        title += "\n(tracheal contact not physical)"
    ax_s.set_title(title, fontsize=10)

    # ---------------- force traces ----------------
    x = np.array([rr["sCB"] for rr in results])
    def trace(ax, title, rows):
        ymax = 0.0
        for key, label, color, ls_ in rows:
            vals = [rr[key] for rr in results]
            if any(v is None for v in vals):
                continue
            y = np.array(vals, float)
            ymax = max(ymax, float(y.max()))
            ax.plot(x, y, ls_, color=color, lw=1, alpha=0.3)           # whole run
            ax.plot(x[:i + 1], y[:i + 1], ls_, color=color, lw=2.2, label=label)
            ax.plot(x[i], y[i], "o", color=color, ms=6)
        ax.axvline(x[i], color="0.6", lw=0.8)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(0, 1.12 * ymax if ymax > 0 else 1)
        ax.set_ylabel("force (N)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    trace(ax_t, "Force on trachea",
          [("P", "resultant", "k", "-"),
           ("C_wall_normal", "normal to wall", c_tr, "-"),
           ("C_wall_shear", "shear (friction)", c_tr, "--")])
    trace(ax_a, "Force on arytenoid (B)",
          [("R_B", "resultant", "k", "-"),
           ("B_normal", "perpendicular to tube", c_ar, "-"),
           ("B_tangential", "along tube", c_ar, "--")])
    ax_a.set_xlabel("arc length B to C, s_BC (mm)")


def plot_advance(results, A, B, idx=-1, mode=None, filename="ett_advance_frame.png",
                 on_tissue=True, show_A=False):
    """One frame of the advance: beam shape with trachea/arytenoid force arrows,
    and the force traces vs s_BC up to that frame.

    results : list from sweep_path() (one model), or {mode: list} with mode= to choose
    idx     : which solve to show (default: the last)
    on_tissue : True draws the force the tube exerts on the tissue; False the
                force the tissue exerts on the tube
    show_A  : also draw the glottis (A) force arrow (pinned/fixed only)
    In cantilever mode the arytenoid force is the clamp force at B, and its
    'along tube' component is the axial clamp reaction (the operator's push), not friction.
    """
    import matplotlib.pyplot as plt
    results, mode = _select_mode_results(results, mode)
    idx = idx % len(results)
    fig, axs = _advance_figure()
    setup = _advance_setup(results, A, B, mode, on_tissue, show_A)
    _draw_advance_frame(axs, results, idx, A, B, mode, setup)
    fig.savefig(filename, dpi=130)
    return filename, fig, axs


def animate_advance(results, A, B, filename="ett_advance.gif", mode=None, fps=6,
                    on_tissue=True, show_A=False, hold_last=6, dpi=90,
                    last_png=True, png_dpi=130):
    """GIF of the tube advancing down the trachea: beam shape and force arrows on
    the left, trachea and arytenoid force traces (with a moving cursor) on the right.

    results   : list from sweep_path() (one model), or {mode: list} with mode= to choose
    fps       : frames per second
    hold_last : extra copies of the final frame, so the loop pauses before restarting
    last_png  : also save the final frame as a PNG. True names it after the GIF
                ('<name>_last.png'); a string gives the filename; False/None skips it
    png_dpi   : resolution of that PNG
    Other arguments as in plot_advance(). Returns the GIF filename.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    results, mode = _select_mode_results(results, mode)
    fig, axs = _advance_figure()
    setup = _advance_setup(results, A, B, mode, on_tissue, show_A)
    frames = list(range(len(results))) + [len(results) - 1] * hold_last
    anim = FuncAnimation(fig, lambda k: _draw_advance_frame(axs, results, k, A, B, mode, setup),
                         frames=frames, interval=1000.0 / fps)
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
    if last_png:
        if last_png is True:
            last_png = os.path.splitext(filename)[0] + "_last.png"
        _draw_advance_frame(axs, results, len(results) - 1, A, B, mode, setup)
        fig.savefig(last_png, dpi=png_dpi)
        print("Saved " + last_png)
    plt.close(fig)
    return filename

#%% 
# ----------------------------------------------------------------------------
# Demo and validation
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_")
    
    EI = tube_EI(14.5, 10.2, 7.5)  # 7.5 mm ID tube, ~5454 N*mm^2
    span = 18.2                   # glottis -> arytenoid, mm

    # Your current depth table: a = distance beyond arytenoid, delta = offset.
    # Here they're placed in a frame with B at the origin and the unloaded
    # tube along +x, with the trachea deflecting the tip toward -y.
    # Replace with real CT coordinates when you have them.

    # From CT Scan
    A = (10.2, 15)
    B = (0.0, 0.0)
    C = [(-10.8, -18.39),(-11.8, -20.91),(-12.3, -30.25),(-8.7, -45.53),(-2, -54.58)]
    # Wall directions traced from CT, one per contact point (degrees,
    # pointing distally, same frame as the points).
    wall_angles = [-112, -103, -87.1, -69.5, -67]
    
    # From 2D Model
    # A = (40.8, 6.6)
    A = (40.8, -0.5)
    B = (0, 0)
    C = [(-33.6,-5.3),(-38.7,-8.0),(-49.3, -13.1),(-58.9,-17.8),(-68.92,-22.7)]
    wall_angles = [29.5, 26.3, 25.9, 26.2, 25.7]
    wall_angles = np.array(wall_angles)-180
    
    # Extra Parameters
    kws = dict(mu=0.7, phi_clamp=None, kappa0=-1/140, n_ramp=5, kB = None)
    
    
    # Compile list of end points
    Cs = [(a, d) for a, d in C]
    
    # Check Set-Up Figure
    figa, ax, prob = plot_setup(A, B, Cs, wall_angles, kappa0=-1/140)


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

    # # --- Example: effect of friction at the third contact point ---
    # # Same contact point, wall angle, pre-curve and clamp angle as the sweep
    # # below; only mu changes (0.1 to 0.7).
    # mus = [round(0.1 * k, 1) for k in range(1, 8)]
    # C3, w3 = Cs[2], wall_angles[2]
    # fric = {mode: [solve(A, B, C3, mode, EI, wall_angle=w3, mu=mu,
    #                      kappa0=-1/140, phi_clamp=-124) for mu in mus]
    #         for mode in MODES}
    # fdf = results_to_df(fric)
    # fdf["amplification"] = fdf["C_wall_normal"] / fdf["C_bending"]   # wall force / bending force
    # fdf.to_csv("ett_friction_sweep.csv", index=False)
    # print(f"\nFriction sweep at contact point 3, C = {C3}, wall angle = {w3} deg")
    # print(fdf[["mode", "mu", "impingement_deg", "P", "C_wall_normal", "C_wall_shear",
    #            "C_bending", "C_axial", "push", "amplification", "C_contact_ok"]]
    #       .round(2).to_string())
    # ffilename, ffig, fax = plot_friction_sweep(fric)
    # print("Saved " + ffilename)
    # afilename, afig, aax = plot_amplification_vs_mu(fric)
    # print("Saved " + afilename)

    # --- Example: spring support at B ---
    # B may move only perpendicular to the ORIGINAL A->B line (positive
    # displacement = along the left normal of A->B). kB is in N/mm; None = rigid.
    # kB_demo is a PLACEHOLDER: replace it with your fixture's stiffness.
    # (n_ramp=5 is enough for these cases and keeps the examples quick.)
    # kB_demo = 2.0
    # kws = dict(mu=0.4, phi_clamp=None, kappa0=-1/140, n_ramp=5)

    # # (a) effect of stiffness at the third contact point
    # kBs = [None, 20, 10, 5, 2, 1]
    # kfix = {mode: [solve(A, B, C3, mode, EI, wall_angle=w3, kB=k, **kws) for k in kBs]
    #         for mode in MODES}
    # kdf = results_to_df(kfix)
    # print(f"\nSpring at B, contact point 3 (kB = NaN means rigid), mu = 0.4")
    # print(kdf[["mode", "kB", "B_disp", "P", "C_wall_normal", "R_B", "B_normal",
    #            "push"]].round(2).to_string())

    # # (b) depth sweep with the spring; the shape plot shows initial and final B
    # spring = {m: sweep(A, B, Cs, m, EI, wall_angles=wall_angles, kB=kB_demo, **kws)
    #           for m in MODES}
    # sdf = results_to_df(spring)
    # sdf.to_csv("ett_sweep_results_springB.csv", index=False)
    # print(f"\nDepth sweep with kB = {kB_demo} N/mm")
    # print(sdf[["mode", "point", "s_BC", "B_disp", "P", "C_wall_normal", "R_B", "push"]]
    #       .round(2).to_string())
    # sfilename, sfig, sax = plot_shapes(spring, A, B, filename="ett_shapes_springB.png")
    # print("Saved " + sfilename)



    # --- Plots (optional) ---
    try:
        by_mode = {m: sweep(A, B, Cs, m, EI, wall_angles=wall_angles, mu=0.7,
                             phi_clamp = None, kappa0=-1/140) for m in MODES}
        by_mode = {m: sweep(A, B, Cs, m, EI, wall_angles=wall_angles, **kws) for m in MODES}
        df = results_to_df(by_mode)   # reuse the sweeps above; no second solve
        df.to_csv(dt_string+"ett_sweep_results.csv", index=False)
        print("\nSaved ett_sweep_results.csv")
        print(df[["mode", "point", "s_BC", "impingement_deg", "P", "C_wall_normal",
                  "C_wall_shear", "push", "A_normal", "B_normal"]].round(2).to_string())
        
        filename, fig, ax = plot_shapes(by_mode, A, B, filename = dt_string+"ett_shapes.png", orientation='vertical')
        filename2, fig2, ax2 = plot_force_vs_arc(by_mode,filename = dt_string+"ett_force_vs_arc.png")
        print("\nSaved" + filename)
        print("Saved" + filename2)
        
        
    except ImportError:
        pass
    
    # --- Example: tip advancing through the contact points, every increment kept ---
    # sweep_path interpolates the tip position and wall angle between the CT contact
    # points (n_steps increments per segment) and records every solve. The tip is
    # assumed to move in a straight line between the CT points.
    # path_kB reuses the PLACEHOLDER stiffness above; set it to None for a rigid B.
    path_kB = 0.2
    anim_mode = "pinned"        # model to animate: "cantilever", "pinned" or "fixed"
    # path = {m: sweep_path(A, B, Cs, m, EI, wall_angles=wall_angles, n_steps=6, mu=0.7,
    #                       phi_clamp = None, kappa0=-1/140, kB=None) for m in MODES}
    path = {m: sweep_path(A, B, Cs, m, EI, wall_angles=wall_angles, n_steps=6, **kws) for m in MODES}
    pdf = results_to_df(path)
    pdf.to_csv(dt_string+"ett_path_results.csv", index=False)
    print(f"\nTip advance, {len(path[anim_mode])} increments per model, kB = {path_kB}")
    print(pdf[pdf["is_key"]][["mode", "step", "tip_path", "s_BC", "B_disp", "P",
                              "C_wall_normal", "R_B"]].round(2).to_string())
    pfilename = animate_advance(path, A, B, mode=anim_mode,
                                filename=dt_string+f"ett_advance_{anim_mode}.gif")
    print("Saved " + pfilename)
    
    import matplotlib.pyplot as plt
    fig3, ax3 = plt.subplots()
    for line in ax[2].get_lines():   # ax[0]=cantilever, ax[1]=pinned, ax[2]=fixed
        ax3.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker = line.get_marker(), label = line.get_label())
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.4)
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title('Fixed')
    # ax3.legend(bbox_to_anchor=(0.5,0.7),loc='upper left')
    ax3.set_xmargin(0.5)
    
    