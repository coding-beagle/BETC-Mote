"""
Robot Arm Experiment Analyser
──────────────────────────────
Run this script and use the file picker to select one or more CSV files.

Figure 1 — Overview (all experiments on one figure):
  • Left:  Overlaid normal distribution curves of successful run durations
  • Right: Overlaid average-speed line charts per trial

Figure 2 — Comparison (shown when >1 file selected):
  • Overlaid duration KDE curves
  • Duration box-plot + Speed box-plot, each annotated with ANOVA results
  • Overlaid speed-per-trial line chart
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Colour palette (cycles across files) ─────────────────────────────────────
PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]
BG = "#F8F9FA"


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["result"] == "success"].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No successful runs found in {path.name}")
    return df


def compute_speeds(df: pd.DataFrame) -> np.ndarray:
    """Return within-run speeds (units/s): displacement / duration for each
    consecutive pair of targets. The first trial of each run is dropped since
    there is no prior position to diff against — cross-run boundaries are
    never included."""
    pos = df[["target_x", "target_y", "target_z"]].values
    displacements = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    durations = df["duration_s"].values[1:]
    return displacements / durations


def anova_summary(groups: list, metric: str) -> str:
    """Run one-way ANOVA and return a formatted string."""
    f, p = stats.f_oneway(*groups)
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    return f"One-way ANOVA — {metric}\n" f"F = {f:.3f},  p = {p:.4f}  ({sig} at α=0.05)"


# ── Overview figure with toggle: per-experiment ↔ combined ───────────────────


def plot_overview(datasets: list):
    """
    datasets: list of (name, df, color)
    Both panels are normal distributions:
      Left:  distribution of run durations
      Right: distribution of within-run speeds
                (cross-run boundaries are never included)
    Toggle button switches between per-experiment overlays and combined view.
    """
    from matplotlib.widgets import Button

    COMBINED_COLOR = "#4C72B0"

    # Pre-compute per-experiment data — skip any experiment with fewer than
    # 2 successful trials (can't compute even one speed value)
    exp_data = []
    for name, df, color in datasets:
        durations = df["duration_s"].values
        if len(df) < 2:
            print(
                f"  ⚠  {name}: only {len(df)} successful trial(s) — "
                f"skipping speed computation for this file"
            )
            speeds = np.array([])
        else:
            speeds = compute_speeds(df)  # within-run only; first trial dropped
        exp_data.append(
            dict(name=name, color=color, durations=durations, speeds=speeds)
        )

    # Pre-compute combined pools (safe to concatenate — within-run only)
    all_durations = np.concatenate([e["durations"] for e in exp_data])
    valid_speeds = [e["speeds"] for e in exp_data if len(e["speeds"]) > 0]
    all_speeds = np.concatenate(valid_speeds) if valid_speeds else np.array([])

    state = {"combined": False, "standardised": False, "skip_first_move": False}

    n_exp = len(exp_data)
    # Scale bottom margin so legends + buttons never overlap the plot area
    bottom_margin = min(0.08 + n_exp * 0.028, 0.45)

    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(
        left=0.07, right=0.97, top=0.91, bottom=bottom_margin, wspace=0.35
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    # ── Buttons — placed well below the plot area ──
    btn_y = bottom_margin * 0.18
    btn_height = min(0.06, bottom_margin * 0.35)
    ax_btn_mode = fig.add_axes([0.15, btn_y, 0.20, btn_height])
    ax_btn_std = fig.add_axes([0.40, btn_y, 0.20, btn_height])
    ax_btn_skip = fig.add_axes([0.65, btn_y, 0.20, btn_height])
    btn_mode = Button(
        ax_btn_mode, "Switch to: Combined", color="#E8EDF7", hovercolor="#C8D4F0"
    )
    btn_std = Button(
        ax_btn_std, "Standardise: Off", color="#F0EDE8", hovercolor="#DDD5C8"
    )
    btn_skip = Button(
        ax_btn_skip, "Skip 1st Move: Off", color="#F0EDE8", hovercolor="#DDD5C8"
    )
    btn_mode.label.set_fontsize(10)
    btn_std.label.set_fontsize(10)
    btn_skip.label.set_fontsize(10)
    # Keep strong references so Python doesn't garbage collect the Button
    # objects (which would silently disconnect click callbacks)
    fig._buttons = [btn_mode, btn_std, btn_skip]

    def _maybe_standardise(data):
        """Return (plot_data, mu, sigma).
        If standardised, returns z-scores; original mu/sigma always returned
        for the legend so the user still sees real units.
        If sigma == 0 (single point or all-identical values), standardisation
        is skipped and the raw data is returned."""
        mu, sigma = data.mean(), data.std()
        if state["standardised"] and sigma > 0:
            return (data - mu) / sigma, mu, sigma
        return data, mu, sigma

    def _draw_normal(ax, data, color, label, add_hist=False):
        """Plot a fitted normal curve (+ optional histogram) on ax,
        applying standardisation if active.
        Silently skips if data is empty or has zero variance."""
        if data is None or len(data) == 0:
            return
        plot_data, mu, sigma = _maybe_standardise(data)
        p_mu = plot_data.mean()
        p_sigma = plot_data.std()
        if p_sigma == 0 or not np.isfinite(p_sigma):
            # Can't fit a normal — just draw a vertical line at the single value
            ax.axvline(
                p_mu,
                color=color,
                linewidth=2.2,
                linestyle="--",
                label=f"{label}  μ={mu:.3f}  (n=1, no spread)",
            )
            return
        pad = max(p_sigma * 2, abs(p_mu) * 0.1, 1e-6)
        x = np.linspace(p_mu - pad * 3, p_mu + pad * 3, 300)
        if add_hist:
            bins = max(6, len(plot_data) // 3)
            ax.hist(
                plot_data,
                bins=bins,
                density=True,
                color=color,
                alpha=0.25,
                edgecolor="white",
                linewidth=0.8,
            )
        ax.plot(
            x,
            stats.norm.pdf(x, p_mu, p_sigma),
            color=color,
            linewidth=2.2,
            label=f"{label}  μ={mu:.3f}  σ={sigma:.3f}",
        )
        ax.axvline(p_mu, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    def _style(a, xlabel, title):
        if state["standardised"]:
            xlabel = f"{xlabel}  [z-score]"
        a.set_xlabel(xlabel, fontsize=11)
        a.set_ylabel("Probability Density", fontsize=11)
        a.set_title(title, fontsize=12, fontweight="bold")
        # Place legend below axes so it never covers buttons regardless of count
        a.legend(
            fontsize=9,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=max(1, min(3, n_exp // 2 + 1)),
            frameon=True,
            framealpha=0.9,
        )
        a.grid(axis="y", linestyle="--", alpha=0.4)
        a.spines[["top", "right"]].set_visible(False)

    def _active_speeds(speeds):
        """Return speeds with the first movement optionally removed."""
        if state["skip_first_move"] and len(speeds) > 1:
            return speeds[1:]
        return speeds

    def redraw():
        ax1.cla()
        ax2.cla()
        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)
        if state["combined"]:
            # Recompute combined pool respecting skip_first_move
            spd_pool = [
                _active_speeds(e["speeds"]) for e in exp_data if len(e["speeds"]) > 0
            ]
            combined_spd = np.concatenate(spd_pool) if spd_pool else np.array([])
            _draw_normal(
                ax1, all_durations, COMBINED_COLOR, "All trials", add_hist=True
            )
            _draw_normal(ax2, combined_spd, COMBINED_COLOR, "All trials", add_hist=True)
            subtitle = "Combined"
        else:
            for e in exp_data:
                _draw_normal(ax1, e["durations"], e["color"], e["name"])
                _draw_normal(ax2, _active_speeds(e["speeds"]), e["color"], e["name"])
            subtitle = "Per Experiment"
        _style(ax1, "Duration (s)", "Distribution of Run Durations")
        spd_xlabel = "Speed (units / s)"
        if state["skip_first_move"]:
            spd_xlabel += "  [1st move excluded]"
        _style(ax2, spd_xlabel, "Distribution of Within-Run Speeds")
        tags = []
        if state["standardised"]:
            tags.append("standardised")
        if state["skip_first_move"]:
            tags.append("1st move excluded")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        fig.suptitle(
            f"Robot Arm Experiments — {subtitle}{tag_str}",
            fontsize=14,
            fontweight="bold",
        )
        fig.canvas.draw_idle()

    def on_toggle_mode(event):
        state["combined"] = not state["combined"]
        btn_mode.label.set_text(
            "Switch to: Per Experiment" if state["combined"] else "Switch to: Combined"
        )
        redraw()

    def on_toggle_std(event):
        state["standardised"] = not state["standardised"]
        btn_std.label.set_text(
            "Standardise: On" if state["standardised"] else "Standardise: Off"
        )
        btn_std.color = "#D8F0D8" if state["standardised"] else "#F0EDE8"
        btn_std.hovercolor = "#B8E0B8" if state["standardised"] else "#DDD5C8"
        redraw()

    def on_toggle_skip(event):
        state["skip_first_move"] = not state["skip_first_move"]
        btn_skip.label.set_text(
            "Skip 1st Move: On" if state["skip_first_move"] else "Skip 1st Move: Off"
        )
        btn_skip.color = "#D8F0D8" if state["skip_first_move"] else "#F0EDE8"
        btn_skip.hovercolor = "#B8E0B8" if state["skip_first_move"] else "#DDD5C8"
        redraw()

    btn_mode.on_clicked(on_toggle_mode)
    btn_std.on_clicked(on_toggle_std)
    btn_skip.on_clicked(on_toggle_skip)
    redraw()


# ── Multi-file comparison figure ──────────────────────────────────────────────


def plot_comparison(datasets: list):
    """
    datasets: list of (name, df, color)
    Produces:
      Row 1: overlaid duration KDE curves (full width)
      Row 2: duration box-plot + speed box-plot (with ANOVA annotations)
      Row 3: overlaid speed-per-trial line chart (full width)
    """
    names = [d[0] for d in datasets]
    dfs = [d[1] for d in datasets]
    colors = [d[2] for d in datasets]
    dur_groups = [df["duration_s"].values for df in dfs]
    speed_groups = [compute_speeds(df) for df in dfs]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Cross-Experiment Comparison", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.65, wspace=0.35)

    # ── Row 0: Overlaid duration KDE curves ──
    ax_kde = fig.add_subplot(gs[0, :])
    ax_kde.set_facecolor(BG)
    for name, durs, col in zip(names, dur_groups, colors):
        mu, sigma = durs.mean(), durs.std()
        x = np.linspace(max(0, durs.min() - 2), durs.max() + 2, 300)
        ax_kde.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color=col,
            linewidth=2.2,
            label=f"{name}  μ={mu:.2f}s  σ={sigma:.2f}s",
        )
        ax_kde.axvline(mu, color=col, linestyle=":", linewidth=1.2, alpha=0.6)
    ax_kde.set_xlabel("Duration (s)", fontsize=11)
    ax_kde.set_ylabel("Probability Density", fontsize=11)
    ax_kde.set_title("Duration Distributions — Overlay", fontsize=12, fontweight="bold")
    ax_kde.legend(fontsize=9)
    ax_kde.grid(axis="y", linestyle="--", alpha=0.4)
    ax_kde.spines[["top", "right"]].set_visible(False)

    # ── Row 1a: Duration box-plot ──
    ax_db = fig.add_subplot(gs[1, 0])
    ax_db.set_facecolor(BG)
    bp1 = ax_db.boxplot(
        dur_groups, patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )
    for patch, col in zip(bp1["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax_db.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax_db.set_ylabel("Duration (s)", fontsize=11)
    ax_db.set_title("Duration Box-Plot", fontsize=12, fontweight="bold")
    ax_db.grid(axis="y", linestyle="--", alpha=0.4)
    ax_db.spines[["top", "right"]].set_visible(False)
    anova_dur = anova_summary(dur_groups, "Duration")
    ax_db.text(
        0.5,
        -0.32,
        anova_dur,
        transform=ax_db.transAxes,
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#EEF2FF", ec="#4C72B0", alpha=0.9),
    )

    # ── Row 1b: Speed box-plot ──
    ax_sb = fig.add_subplot(gs[1, 1])
    ax_sb.set_facecolor(BG)
    bp2 = ax_sb.boxplot(
        speed_groups, patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )
    for patch, col in zip(bp2["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax_sb.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax_sb.set_ylabel("Within-Run Speed (units / s)", fontsize=11)
    ax_sb.set_title("Within-Run Speed Box-Plot", fontsize=12, fontweight="bold")
    ax_sb.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sb.spines[["top", "right"]].set_visible(False)
    anova_spd = anova_summary(speed_groups, "Speed")
    ax_sb.text(
        0.5,
        -0.32,
        anova_spd,
        transform=ax_sb.transAxes,
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF5EE", ec="#DD8452", alpha=0.9),
    )

    # ── Row 2: Overlaid speed-per-trial line chart ──
    ax_sl = fig.add_subplot(gs[2, :])
    ax_sl.set_facecolor(BG)
    for name, df, col in zip(names, dfs, colors):
        speeds = compute_speeds(df)
        trial_nums = df["trial"].values[1:]
        ax_sl.plot(
            trial_nums,
            speeds,
            color=col,
            linewidth=2,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=col,
            markeredgewidth=1.8,
            label=name,
            zorder=3,
        )
    ax_sl.set_xlabel("Trial Number", fontsize=11)
    ax_sl.set_ylabel("Average Speed (units / s)", fontsize=11)
    ax_sl.set_title("Speed per Trial — All Experiments", fontsize=12, fontweight="bold")
    ax_sl.legend(fontsize=9)
    ax_sl.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sl.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    paths = filedialog.askopenfilenames(
        title="Select experiment CSV file(s)  [hold Ctrl/Cmd for multiple]",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if not paths:
        messagebox.showinfo("No files selected", "No files were selected. Exiting.")
        sys.exit(0)

    paths = [Path(p) for p in paths]
    print(f"\nLoaded {len(paths)} file(s):")
    for p in paths:
        print(f"  • {p.name}")

    datasets = []
    for i, path in enumerate(paths):
        try:
            df = load_csv(path)
            color = PALETTE[i % len(PALETTE)]
            datasets.append((path.stem, df, color))
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    if not datasets:
        sys.exit(1)

    # Console summary per file
    for name, df, color in datasets:
        durs = df["duration_s"].values
        speeds = compute_speeds(df)
        print(f"\n── {name} ──")
        print(
            f"  Durations  n={len(durs)}  μ={durs.mean():.3f}s  "
            f"σ={durs.std():.3f}s  "
            f"min={durs.min():.3f}s  max={durs.max():.3f}s"
        )
        print(
            f"  Speeds     n={len(speeds)}  μ={speeds.mean():.4f}  "
            f"min={speeds.min():.4f}  max={speeds.max():.4f}  (units/s)"
        )

    # Single combined overview figure
    plot_overview(datasets)

    # Comparison + ANOVA (only when multiple files loaded)
    if len(datasets) > 1:
        plot_comparison(datasets)
        dur_groups = [d[1]["duration_s"].values for d in datasets]
        speed_groups = [compute_speeds(d[1]) for d in datasets]
        print("\n" + anova_summary(dur_groups, "Duration"))
        print(anova_summary(speed_groups, "Speed"))

    plt.show()


if __name__ == "__main__":
    main()
