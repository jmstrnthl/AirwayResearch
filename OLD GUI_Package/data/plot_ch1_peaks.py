"""Plot seconds vs ch1-Newtons and mark the major peak(s) of each waveform.

A "waveform" is a contiguous stretch where the force rises above the resting
baseline (~0 N). Within each waveform, the highest point is always marked, plus
any other local peak whose prominence is at least MIN_REL_PROMINENCE of that
waveform's maximum.

Usage:
    python plot_ch1_peaks.py [path/to/file.csv]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- Tunable settings -------------------------------------------------------
ONSET_THRESHOLD_N = 0.05     # force above this (N) counts as "in a waveform"
MERGE_GAP_S = 1.0            # dips below threshold shorter than this are bridged
MIN_WAVEFORM_PEAK_N = 0.1    # ignore waveforms whose max is below this (noise)
MIN_REL_PROMINENCE = 0.5     # secondary peaks need prominence >= this * waveform max

# Test conditions: (label, first waveform, last waveform), numbered from 0.
GROUPS = [
    ("Rigid Stylet + Standard Tube",      0,  2),
    ("Rigid Stylet + Reinforced Tube",    3,  5),
    ("Dynamic Stylet + Standard Tube",    6, 15),
    ("Dynamic Stylet + Reinforced Tube", 16, 24),
]
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Peaks (numbered from 0) averaged in the bar chart.
BAR_CHART_PEAKS = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 16, 18, 20, 21, 25, 27]
# ---------------------------------------------------------------------------

DEFAULT_CSV = Path(__file__).with_name("LarynxData_10112026_2300.csv")


def load_data(csv_path):
    # Line 1 is a title ("LarynxData"); line 2 is the header. Rows end with
    # trailing ';', so index_col=False keeps columns aligned.
    df = pd.read_csv(csv_path, sep=";", skiprows=1, skipinitialspace=True,
                     index_col=False)
    df.columns = df.columns.str.strip()
    return df["seconds"].to_numpy(float), df["ch1-Newtons"].to_numpy(float)


def find_waveforms(t, y):
    """Return (start, end) index pairs (end exclusive) for each waveform."""
    active = y > ONSET_THRESHOLD_N
    edges = np.diff(active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    merged = []
    for s, e in zip(starts, ends):
        if merged and t[s] - t[merged[-1][1] - 1] < MERGE_GAP_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if y[s:e].max() >= MIN_WAVEFORM_PEAK_N]


def find_major_peaks(y, waveforms):
    peaks = []
    for s, e in waveforms:
        seg = y[s:e]
        # Pad with baseline so peaks at the segment edges can be detected.
        padded = np.concatenate(([0.0], seg, [0.0]))
        idx, _ = find_peaks(padded, prominence=MIN_REL_PROMINENCE * seg.max())
        idx = set((idx - 1).tolist())
        idx.add(int(np.argmax(seg)))  # always include the waveform maximum
        peaks.extend(s + i for i in sorted(idx))
    return np.array(peaks, dtype=int)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    t, y = load_data(csv_path)

    waveforms = find_waveforms(t, y)
    peaks = find_major_peaks(y, waveforms)

    # Normalize so the largest peak in the recording is 1.0.
    max_peak_n = y[peaks].max()
    y_norm = y / max_peak_n

    def group_of(w):
        return next((g for g, (_, lo, hi) in enumerate(GROUPS) if lo <= w <= hi),
                    None)

    print(f"{csv_path.name}: {len(waveforms)} waveforms, {len(peaks)} major peaks")
    print(f"Largest peak: {max_peak_n:.3f} N (normalized to 1.0)")
    print(f"{'peak':>4}  {'waveform':>8}  {'time (s)':>10}  {'force (N)':>10}"
          f"  {'normalized':>10}")
    peaks_list = []
    for n, i in enumerate(peaks, 0):
        w = next(k for k, (s, e) in enumerate(waveforms, 0) if s <= i < e)
        print(f"{n:>4}  {'W' + str(w):>8}  {t[i]:>10.3f}  {y[i]:>10.3f}"
              f"  {y_norm[i]:>10.3f}")
        peaks_list.append([n, w, t[i], y[i], y_norm[i]])

    peaks_list_df = pd.DataFrame(
        peaks_list,
        columns=['Peak', 'Waveform', 'Time (s)', 'Force (N)', 'Normalized'])
    peaks_list_df.to_csv(csv_path.stem + "_ch1_peaks.png")

    plot_force_trace(csv_path, t, y_norm, waveforms, peaks, group_of)
    plot_group_bars(csv_path, peaks_list_df, group_of, t, y_norm, waveforms,
                    peaks)
    plt.show()


def plot_force_trace(csv_path, t, y, waveforms, peaks, group_of):
    """Normalized force vs time, with waveforms, peaks and groups labeled."""
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.plot(t, y, color="#3a3a3a", linewidth=1, label="ch1 force")

    ax.plot(t[peaks], y[peaks], "o", markersize=8, markerfacecolor="none",
            markeredgecolor="#d62728", markeredgewidth=1.5,
            label=f"Major peaks ({len(peaks)})")
    ax.set_ylim(-0.11, 1.5)

    # Helper: data coordinates -> points, so labels can be spaced on screen.
    pt_per_px = 72.0 / fig.dpi

    def to_pts(x, yv):
        px, py = ax.transData.transform((x, yv))
        return px * pt_per_px, py * pt_per_px

    fig.tight_layout()  # fix the axes size before measuring label positions

    # Shade and number each waveform (W1, W2, ...) just below the baseline.
    # Each label goes on the first row where it clears the previous label.
    n_rows = 3
    last_x = [-1e9] * n_rows
    for w, (s, e) in enumerate(waveforms, 0):
        g = group_of(w)
        ax.axvspan(t[s], t[e - 1], linewidth=0, alpha=0.12,
                   color="#888888" if g is None else GROUP_COLORS[g])
        xc = (t[s] + t[e - 1]) / 2
        x_pt, _ = to_pts(xc, 0)
        row = next((r for r in range(n_rows) if x_pt - last_x[r] > 26),
                   int(np.argmin(last_x)))
        last_x[row] = x_pt
        ax.annotate(f"W{w}", (xc, 0), xytext=(0, -6 - 9 * row),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color="#555555")

    # Number each peak (1, 2, ...) above its circle. Labels and circles are
    # treated as boxes (in points); a label that would overlap any of them is
    # pushed higher until it is clear.
    circle_r = 5
    boxes = [(x - circle_r, x + circle_r, yv - circle_r, yv + circle_r)
             for x, yv in (to_pts(t[i], y[i]) for i in peaks)]

    def overlaps(b):
        return any(b[0] < o[1] and b[1] > o[0] and b[2] < o[3] and b[3] > o[2]
                   for o in boxes)

    for n, i in enumerate(peaks, 0):
        x_pt, y_pt = to_pts(t[i], y[i])
        half_w = 3 + 2.5 * len(str(n))
        label_y = y_pt + 7
        while overlaps((x_pt - half_w, x_pt + half_w, label_y, label_y + 9)):
            label_y += 3
        boxes.append((x_pt - half_w, x_pt + half_w, label_y, label_y + 9))
        ax.annotate(str(n), (t[i], y[i]), xytext=(0, label_y - y_pt),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#d62728")

    # Bracket and name each test condition across the top of the plot.
    for g, (label, lo, hi) in enumerate(GROUPS):
        if lo >= len(waveforms):
            continue
        hi = min(hi, len(waveforms) - 1)
        x0, x1 = t[waveforms[lo][0]], t[waveforms[hi][1] - 1]
        ax.plot([x0, x0, x1, x1], [1.24, 1.27, 1.27, 1.24],
                color=GROUP_COLORS[g], linewidth=2, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 1.29, label.replace(" + ", "\n+ "),
                ha="center", va="bottom", fontsize=9, color="#0b0b0b",
                linespacing=1.1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (normalized to largest peak)")
    ax.set_title(f"ch1 force vs time with major peaks — {csv_path.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False)
    fig.tight_layout()

    out_png = csv_path.with_name(csv_path.stem + "_ch1_peaks.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


def plot_group_bars(csv_path, peaks_df, group_of, t, y, waveforms, peaks):
    """Mean of the selected peaks per test condition, error bars = 1 SD.

    Under each bar is a sample waveform from that condition: the waveform
    holding the selected peak closest to the condition's mean.
    """
    sel = peaks_df[peaks_df["Peak"].isin(BAR_CHART_PEAKS)].copy()
    sel["Group"] = sel["Waveform"].map(group_of)

    labels, means, sds, values, samples = [], [], [], [], []
    print("\nSelected peaks by group (normalized force; SD is sample SD, n-1):")
    for g, (label, _, _) in enumerate(GROUPS):
        rows = sel[sel["Group"] == g]
        v = rows["Normalized"].to_numpy()
        labels.append(label)
        values.append(v)
        means.append(v.mean() if len(v) else np.nan)
        sds.append(v.std(ddof=1) if len(v) > 1 else np.nan)
        if len(v):
            closest = rows.iloc[int(np.argmin(np.abs(v - means[-1])))]
            samples.append((int(closest["Waveform"]), int(closest["Peak"])))
        else:
            samples.append(None)
        print(f"  {label:<34} n={len(v)}  mean={means[-1]:.3f}  SD={sds[-1]:.3f}"
              f"  peaks {rows['Peak'].tolist()}")

    n_groups = len(GROUPS)
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, n_groups, height_ratios=[2.2, 1], hspace=0.38,
                          wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(n_groups)
    ax.set_xlim(-0.5, n_groups - 0.5)  # one bar centered over each column
    ax.bar(x, means, width=0.6, color=GROUP_COLORS, edgecolor="white",
           linewidth=2, zorder=2)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="#0b0b0b",
                elinewidth=1.5, capsize=6, capthick=1.5, zorder=3)

    # Show the individual peaks behind each average.
    rng = np.random.default_rng(0)
    for xi, v in zip(x, values):
        jitter = rng.uniform(-0.12, 0.12, len(v))
        ax.plot(xi + jitter, v, "o", markersize=5, markerfacecolor="white",
                markeredgecolor="#3a3a3a", markeredgewidth=1, zorder=4)

    for xi, m, s, v in zip(x, means, sds, values):
        top = max(m + (0 if np.isnan(s) else s), v.max() if len(v) else m)
        ax.text(xi, top + 0.03, f"{m:.2f} ± {s:.2f}", ha="center",
                va="bottom", fontsize=9, color="#0b0b0b")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" + ", "\n+ ") for l in labels])
    ax.set_ylabel("Mean peak force (normalized to largest peak)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Average peak force by condition (mean ± SD) — {csv_path.name}")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Sample waveform under each bar, time measured from waveform start.
    pad_s = 1.0
    wave_axes = []
    for g, sample in enumerate(samples):
        wax = fig.add_subplot(gs[1, g], sharey=wave_axes[0] if wave_axes else None)
        wave_axes.append(wax)
        if sample is None:
            wax.set_axis_off()
            continue
        w, p = sample
        s, e = waveforms[w]
        t0 = t[s]
        win = (t >= t0 - pad_s) & (t <= t[e - 1] + pad_s)
        wax.plot(t[win] - t0, y[win], color=GROUP_COLORS[g], linewidth=1.5)
        pk = peaks[p]
        wax.plot(t[pk] - t0, y[pk], "o", markersize=7, markerfacecolor="none",
                 markeredgecolor="#d62728", markeredgewidth=1.5)
        wax.set_title(f"Sample: W{w} (peak {p})", fontsize=9, color="#52514e")
        wax.set_xlabel("Time from start (s)", fontsize=8)
        wax.tick_params(labelsize=8)
        wax.grid(True, alpha=0.3)
        wax.spines[["top", "right"]].set_visible(False)
        if g > 0:
            wax.tick_params(labelleft=False)
    wave_axes[0].set_ylabel("Force (normalized)", fontsize=8)
    wave_axes[0].set_ylim(-0.05, 1.05)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07)

    out_png = csv_path.with_name(csv_path.stem + "_ch1_peak_groups.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved bar chart to {out_png}")


if __name__ == "__main__":
    main()
