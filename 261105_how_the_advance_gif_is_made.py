# -*- coding: utf-8 -*-
"""
How the "tube advancing" GIF is made
====================================

This file explains how animate_advance() in
261028_ett_shooting_solver_v3_springb.py makes ett_advance_<mode>.gif, and
then rebuilds the same technique in a small toy example you can run and edit.
The toy needs no solver: it fakes the "results" with a bending arc, so it runs
in a couple of seconds.

Run it:   python 261105_how_the_advance_gif_is_made.py
Output:   toy_advance.gif and toy_advance_last.png in this folder.


THE BIG IDEA
------------
A GIF is a stack of still pictures. Matplotlib's FuncAnimation makes those
pictures by calling one function you write, update(k), once per frame. Each call
draws frame k onto the SAME figure; the writer (PillowWriter) grabs the figure
after each call and appends it to the GIF.

So the whole animation is just:

    results = [solve(...) for each tip position]      # 1. the data, one entry per frame
    setup   = fixed limits + arrow scale from ALL of results   # 2. so frames don't jump
    fig     = one figure with 3 panels                 # 3. the canvas
    def update(k): clear the panels, redraw frame k    # 4. one frame
    FuncAnimation(fig, update, frames=[0, 1, ..., last, last, last]).save(gif)  # 5.
    redraw the last frame, fig.savefig(png)            # 6. the still image


STEP BY STEP, WITH THE REAL FUNCTION NAMES
------------------------------------------
1. The frames come from sweep_path()
   sweep_path(A, B, C_list, mode, EI, wall_angles, n_steps=6, ...) walks the
   tube tip from one CT contact point to the next in n_steps straight-line
   increments (the wall angle is interpolated the same way). At every increment
   it calls solve(), starting from the previous solution so fsolve converges
   quickly, and keeps the whole result dictionary: the deformed shape
   r["shape"], the tip point r["C"], the force vectors r["fC"], r["fB"],
   r["fA"], their components (P, C_wall_normal, R_B, ...), and s_BC. With
   5 CT points and n_steps=6 that is 4*6 + 1 = 25 results = 25 frames.
   Increments that fail to converge are skipped, so a frame can be missing.

2. _select_mode_results() picks one model
   `path` in the demo is {"cantilever": [...], "pinned": [...], "fixed": [...]};
   mode="pinned" picks the pinned list.

3. _advance_setup() fixes everything that must NOT change between frames
   - Axis limits: it stacks every point that will ever be drawn (all shapes,
     all tip points, A, B, moved B, and every arrow tip) and takes their
     min/max plus 7 % padding. If each frame chose its own limits, the picture
     would jump around and the motion would be hard to read.
   - Arrow scale: one "mm of arrow per newton" for the whole run,
     scale = 0.2 * (size of the geometry) / (largest force in any frame).
     The largest force is drawn as an arrow 20 % as long as the geometry is
     wide, and arrows in different frames can be compared by eye.
   - Arrow direction: on_tissue=True flips the sign (sgn = -1), so the arrows
     show the force the tube puts on the tissue instead of the reverse.

4. _advance_figure() makes the canvas
   One 10.5 x 6.6 inch figure, a 2x2 grid: the tall left panel (both rows) is
   the beam shape; the right column holds the trachea force trace (top) and the
   arytenoid force trace (bottom).

5. _draw_advance_frame(axs, results, i, ...) draws frame i
   It starts with ax.clear() on all three panels and redraws from scratch.
   That is simpler than updating line objects and is fast enough for ~30
   frames.
   Left panel, back to front (zorder):
     - every shape in light grey: a faint "history" of the whole run
     - a dotted line through all tip positions, open circles at the CT points
     - the current shape in thick blue
     - a short black line for the tracheal wall at the tip, at its wall angle
     - B (red dot; with a spring, an open circle for the rest position, a
       diamond for where it moved), A (purple square)
     - force arrows from _arrow(): annotate() with an arrowstyle, plus a
       label "x.x N" just beyond the tip
     - a legend box with the arrow scale, and a title with s_BC
   Right panels, via the inner trace() helper, for each force component:
     - the whole curve, faint (alpha 0.3), so you can see where it is headed
     - the curve up to frame i, thick: it "grows" as the GIF plays
     - a dot at frame i, and a grey vertical line (axvline) as a cursor
     - fixed x/y limits computed from the whole run, for the same reason as
       in step 3

6. animate_advance() puts it together
       frames = list(range(len(results))) + [len(results) - 1] * hold_last
   The last frame is repeated hold_last (6) times, so the GIF pauses on the
   final state before it loops back to the start.
       anim = FuncAnimation(fig, lambda k: _draw_advance_frame(..., k, ...),
                            frames=frames, interval=1000 / fps)
       anim.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
   FuncAnimation is lazy: nothing is drawn until .save() runs; save() then
   calls the lambda once per entry of `frames`. PillowWriter (the Pillow
   imaging library) encodes the GIF, so no ffmpeg is needed. fps=6 means each
   frame shows for 1/6 s; dpi=90 keeps the file small.

7. The final-frame PNG (last_png=True)
   After saving, the last frame is drawn once more and written with
   fig.savefig(..., dpi=130) as <gif name>_last.png. Then plt.close(fig)
   frees the figure.
   (This option was added in PR #1. If your local copy still has the separate
   plot_advance(..., idx=-1) call at the end of the demo, run `git pull`.)


TOY EXAMPLE
-----------
Below, the same six steps are done with fake data: a 60 mm "tube" clamped at
the origin whose tip is pushed so its curvature grows frame by frame, with an
arrow at the tip and a force-vs-step trace. Change N_FRAMES, FPS or HOLD_LAST
and rerun to see the effect.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

N_FRAMES = 25        # like sweep_path: one result per tip position
FPS = 6              # frames per second in the GIF
HOLD_LAST = 6        # extra copies of the last frame (pause before looping)
GIF_NAME = "toy_advance.gif"
PNG_NAME = "toy_advance_last.png"


# --- Step 1: the data, one dictionary per frame ------------------------------
# In the real code these come from sweep_path(); here a circular arc with
# growing curvature stands in for the elastica solution.
def fake_results(n=N_FRAMES, length=60.0):
    results = []
    for k in range(n):
        kappa = 0.03 * k / (n - 1)                 # curvature, 1/mm
        s = np.linspace(0.0, length, 80)           # arc length along the tube
        if kappa == 0:
            x, y = s, np.zeros_like(s)
        else:                                      # arc that starts along +x, bends down
            x = np.sin(kappa * s) / kappa
            y = -(1 - np.cos(kappa * s)) / kappa
        shape = np.column_stack([x, y])
        tip = shape[-1]
        force = 5.0 * kappa / 0.03                 # pretend force grows with bending (N)
        phi = -kappa * length                      # tube angle at the tip
        normal = np.array([np.sin(phi), np.cos(phi)])   # perpendicular to the tube
        results.append(dict(shape=shape, C=tip, fC=force * normal, P=force, step=k))
    return results


# --- Step 3: fixed limits and arrow scale from ALL frames --------------------
def make_setup(results, arrow_frac=0.2):
    pts = np.vstack([r["shape"] for r in results])
    span = float((pts.max(0) - pts.min(0)).max())
    fmax = max(np.linalg.norm(r["fC"]) for r in results) or 1.0
    scale = arrow_frac * span / fmax               # mm of arrow per N
    tips = np.array([r["C"] + scale * r["fC"] for r in results])
    allp = np.vstack([pts, tips])
    pad = 0.07 * span
    lo, hi = allp.min(0) - pad, allp.max(0) + pad
    return dict(scale=scale, xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
                fmax=max(r["P"] for r in results))


# --- Step 4: the canvas -------------------------------------------------------
def make_figure():
    fig = plt.figure(figsize=(9, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    axs = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    # fixed margins (not tight_layout): every frame is cleared and redrawn, so the
    # layout has to leave room for labels that do not exist yet
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.25)
    return fig, axs


# --- Step 5: draw one frame ---------------------------------------------------
def draw_frame(axs, results, i, setup):
    ax_s, ax_t = axs
    ax_s.clear()
    ax_t.clear()                                   # start every frame from blank
    r = results[i]

    # left: faint history, current shape, force arrow
    for rr in results:
        ax_s.plot(rr["shape"][:, 0], rr["shape"][:, 1], color="0.85", lw=0.8)
    ax_s.plot(r["shape"][:, 0], r["shape"][:, 1], color="tab:blue", lw=2.6)
    ax_s.plot(0, 0, "o", color="tab:red", ms=7)    # the clamp
    vec = setup["scale"] * r["fC"]
    ax_s.annotate("", xy=r["C"] + vec, xytext=r["C"],
                  arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.2,
                                  shrinkA=0, shrinkB=0))
    # label just beyond the arrowhead, pushed 9 points further along the arrow
    # (same trick as _arrow() in the solver), so it never sits on the head
    L = float(np.linalg.norm(vec))
    u = vec / L if L > 0 else np.array([1.0, 0.0])
    ax_s.annotate(f"{r['P']:.1f} N", xy=r["C"] + vec, xytext=(9 * u[0], 9 * u[1]),
                  textcoords="offset points", ha="left" if u[0] >= 0 else "right",
                  va="bottom" if u[1] >= 0 else "top",
                  color="#c0392b", fontsize=8, fontweight="bold")
    ax_s.set_xlim(*setup["xlim"]); ax_s.set_ylim(*setup["ylim"])   # fixed limits
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.grid(alpha=0.3)
    ax_s.set_xlabel("x (mm)"); ax_s.set_ylabel("y (mm)")
    ax_s.set_title(f"frame {i + 1} of {len(results)}", fontsize=10)

    # right: whole curve faint, curve so far thick, dot + cursor at frame i
    x = np.array([rr["step"] for rr in results])
    y = np.array([rr["P"] for rr in results])
    ax_t.plot(x, y, color="k", lw=1, alpha=0.3)
    ax_t.plot(x[:i + 1], y[:i + 1], color="k", lw=2.2)
    ax_t.plot(x[i], y[i], "o", color="k", ms=6)
    ax_t.axvline(x[i], color="0.6", lw=0.8)
    ax_t.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax_t.set_ylim(0, 1.12 * setup["fmax"])
    ax_t.set_xlabel("step"); ax_t.set_ylabel("force (N)")
    ax_t.set_title("Force at tip", fontsize=10)
    ax_t.grid(alpha=0.3)


# --- Steps 6 and 7: animate, save the GIF, then the last frame as a PNG ------
def main():
    results = fake_results()
    setup = make_setup(results)
    fig, axs = make_figure()

    frames = list(range(len(results))) + [len(results) - 1] * HOLD_LAST
    anim = FuncAnimation(fig, lambda k: draw_frame(axs, results, k, setup),
                         frames=frames, interval=1000.0 / FPS)
    anim.save(GIF_NAME, writer=PillowWriter(fps=FPS), dpi=90)
    print(f"Saved {GIF_NAME}  ({len(frames)} frames at {FPS} fps "
          f"= {len(frames) / FPS:.1f} s per loop)")

    draw_frame(axs, results, len(results) - 1, setup)
    fig.savefig(PNG_NAME, dpi=130)
    print(f"Saved {PNG_NAME}")
    plt.close(fig)


if __name__ == "__main__":
    main()
