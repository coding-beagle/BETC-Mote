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
    """Return per-trial average speed (units/s) starting from trial 2."""
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
    Shows a two-panel figure with a Toggle button to switch between:
      • Split  — each experiment plotted separately, colour-coded
      • Combined — all data merged into one distribution / one speed line
    """
    from matplotlib.widgets import Button

    COMBINED_COLOR = "#4C72B0"

    # Pre-compute per-experiment data
    exp_data = []
    for name, df, color in datasets:
        durations = df["duration_s"].values
        speeds = compute_speeds(df)
        trials = df["trial"].values[1:]
        exp_data.append(
            dict(
                name=name,
                color=color,
                durations=durations,
                speeds=speeds,
                trials=trials,
            )
        )

    # Pre-compute combined data
    all_durations = np.concatenate([e["durations"] for e in exp_data])
    all_speeds = np.concatenate([e["speeds"] for e in exp_data])
    # Re-index trials sequentially across all experiments
    all_trials = np.arange(1, len(all_speeds) + 1)

    state = {"combined": False}

    fig = plt.figure(figsize=(14, 7))
    # Leave bottom margin for the button
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.18, wspace=0.35)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    # ── Button ──
    ax_btn = fig.add_axes([0.42, 0.04, 0.16, 0.07])
    btn = Button(ax_btn, "Switch to: Combined", color="#E8EDF7", hovercolor="#C8D4F0")
    btn.label.set_fontsize(10)

    def draw_split():
        ax1.cla()
        ax2.cla()
        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)

        all_trial_nums = []
        for e in exp_data:
            mu, sigma = e["durations"].mean(), e["durations"].std()
            x = np.linspace(
                max(0, e["durations"].min() - 1), e["durations"].max() + 1, 300
            )
            ax1.plot(
                x,
                stats.norm.pdf(x, mu, sigma),
                color=e["color"],
                linewidth=2.2,
                label=f"{e['name']}  μ={mu:.2f}s  σ={sigma:.2f}s",
            )
            ax1.axvline(mu, color=e["color"], linestyle=":", linewidth=1.2, alpha=0.6)

            ax2.plot(
                e["trials"],
                e["speeds"],
                color=e["color"],
                linewidth=2.2,
                marker="o",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=e["color"],
                markeredgewidth=2,
                zorder=3,
                label=f"{e['name']}  μ={e['speeds'].mean():.3f} u/s",
            )
            ax2.axhline(
                e["speeds"].mean(),
                color=e["color"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
            )
            all_trial_nums.extend(e["trials"].tolist())

        _style_axes(ax1, ax2, sorted(set(all_trial_nums)))
        fig.suptitle(
            "Robot Arm Experiments — Per Experiment", fontsize=14, fontweight="bold"
        )

    def draw_combined():
        ax1.cla()
        ax2.cla()
        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)

        mu, sigma = all_durations.mean(), all_durations.std()
        x = np.linspace(max(0, all_durations.min() - 1), all_durations.max() + 1, 300)
        ax1.hist(
            all_durations,
            bins=max(6, len(all_durations) // 3),
            density=True,
            color=COMBINED_COLOR,
            alpha=0.3,
            edgecolor="white",
            linewidth=0.8,
        )
        ax1.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color=COMBINED_COLOR,
            linewidth=2.5,
            label=f"All trials  μ={mu:.2f}s  σ={sigma:.2f}s",
        )
        ax1.axvline(mu, color=COMBINED_COLOR, linestyle="--", linewidth=1.4, alpha=0.8)

        ax2.plot(
            all_trials,
            all_speeds,
            color=COMBINED_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=COMBINED_COLOR,
            markeredgewidth=2,
            zorder=3,
            label=f"All trials  μ={all_speeds.mean():.3f} u/s",
        )
        ax2.axhline(
            all_speeds.mean(),
            color=COMBINED_COLOR,
            linestyle="--",
            linewidth=1.4,
            alpha=0.6,
        )

        _style_axes(ax1, ax2, all_trials.tolist())
        fig.suptitle("Robot Arm Experiments — Combined", fontsize=14, fontweight="bold")

    def _style_axes(a1, a2, trial_list):
        a1.set_xlabel("Duration (s)", fontsize=11)
        a1.set_ylabel("Probability Density", fontsize=11)
        a1.set_title("Distribution of Run Durations", fontsize=12, fontweight="bold")
        a1.legend(fontsize=9)
        a1.grid(axis="y", linestyle="--", alpha=0.4)
        a1.spines[["top", "right"]].set_visible(False)

        # Only tick every other trial if there are many
        ticks = trial_list if len(trial_list) <= 15 else trial_list[::2]
        a2.set_xticks(ticks)
        a2.set_xlabel("Trial", fontsize=11)
        a2.set_ylabel("Average Speed (units / s)", fontsize=11)
        a2.set_title(
            "Average Speed Between Consecutive Targets", fontsize=12, fontweight="bold"
        )
        a2.legend(fontsize=9)
        a2.grid(axis="y", linestyle="--", alpha=0.4)
        a2.spines[["top", "right"]].set_visible(False)

    def on_toggle(event):
        state["combined"] = not state["combined"]
        if state["combined"]:
            draw_combined()
            btn.label.set_text("Switch to: Per Experiment")
        else:
            draw_split()
            btn.label.set_text("Switch to: Combined")
        fig.canvas.draw_idle()

    btn.on_clicked(on_toggle)

    # Initial draw
    draw_split()
    fig.canvas.draw_idle()


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
    ax_sb.set_ylabel("Average Speed (units / s)", fontsize=11)
    ax_sb.set_title("Speed Box-Plot", fontsize=12, fontweight="bold")
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
