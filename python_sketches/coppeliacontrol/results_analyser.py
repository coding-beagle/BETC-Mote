"""
Robot Arm Experiment Analyser
──────────────────────────────
Run this script and use the file picker to select one or more CSV files (Group A).

Figure 1 — Overview:
  • Left:  Overlaid normal distribution curves of successful run durations
  • Right: Distribution of within-run speeds
  Group A = cool colours (blues/greens), Group B = warm colours (oranges/reds).
  Toggle button switches between per-experiment overlays and combined view.
  "Add Group B…" button loads a second set of CSVs for comparison.

Figure 2 — Comparison (shown when Group B is loaded):
  • Group A pooled vs Group B pooled for durations and speeds
  • Duration KDE overlay, box-plots, speed box-plot — all annotated with ANOVA
  • Group mean speed per move with ±1 SD band
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

# ── Colour palettes ───────────────────────────────────────────────────────────
PALETTE_A = ["#4C72B0", "#55A868", "#64B5CD", "#3A9E7A", "#2E6FA3", "#76B7B2"]  # cool
PALETTE_B = ["#DD8452", "#C44E52", "#E6A817", "#D45F86", "#C7622E", "#E8734A"]  # warm

BG = "#F8F9FA"

# ── Shared state ──────────────────────────────────────────────────────────────
groups: dict = {"A": [], "B": []}  # each entry: name/df/color/durations/speeds
group_titles: dict = {"A": "Group A", "B": "Group B"}  # user-editable display names
_comparison_fig = None


def _color_for(group: str, index: int) -> str:
    palette = PALETTE_A if group == "A" else PALETTE_B
    return palette[index % len(palette)]


def _tk_root() -> tk.Tk:
    """Return the single persistent hidden Tk root, creating it if needed."""
    if not hasattr(_tk_root, "_inst") or not _tk_root._inst.winfo_exists():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        _tk_root._inst = root
    return _tk_root._inst


def _ask_title(group_key: str, default: str) -> str:
    """Show a modal Toplevel dialog asking the user to name a group."""
    result = [default]
    root = _tk_root()

    win = tk.Toplevel(root)
    win.title("Name this group")
    win.resizable(False, False)
    win.attributes("-topmost", True)
    win.grab_set()

    tk.Label(
        win, text="Enter a display name for this group:", font=("Helvetica", 11), pady=8
    ).pack(padx=16)

    var = tk.StringVar(value=default)
    entry = tk.Entry(win, textvariable=var, font=("Helvetica", 11), width=28)
    entry.pack(padx=16, pady=4)
    entry.select_range(0, tk.END)
    entry.focus_set()

    def _ok(event=None):
        val = var.get().strip()
        result[0] = val if val else default
        win.grab_release()
        win.destroy()

    def _cancel(event=None):
        win.grab_release()
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="OK", width=10, command=_ok).pack(side=tk.LEFT, padx=6)
    tk.Button(btn_frame, text="Cancel", width=10, command=_cancel).pack(
        side=tk.LEFT, padx=6
    )
    win.bind("<Return>", _ok)
    win.bind("<Escape>", _cancel)

    root.wait_window(win)
    return result[0]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["result"] == "success"].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No successful runs found in {path.name}")
    return df


def compute_speeds(df: pd.DataFrame) -> np.ndarray:
    pos = df[["target_x", "target_y", "target_z"]].values
    displacements = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    durations = df["duration_s"].values[1:]
    return displacements / durations


def anova_summary(groups_data: list, metric: str) -> str:
    f, p = stats.f_oneway(*groups_data)
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    return f"One-way ANOVA — {metric}\nF = {f:.3f},  p = {p:.4f}  ({sig} at α=0.05)"


def _build_entry(name: str, df: pd.DataFrame, color: str) -> dict:
    if len(df) < 2:
        print(
            f"  ⚠  {name}: only {len(df)} successful trial(s) — skipping speed computation"
        )
        speeds = np.array([])
    else:
        speeds = compute_speeds(df)
    return dict(
        name=name, df=df, color=color, durations=df["duration_s"].values, speeds=speeds
    )


def _pool_group(group_key: str):
    """Return (pooled_durations, pooled_speeds) for all experiments in a group."""
    entries = groups[group_key]
    durs = (
        np.concatenate([e["durations"] for e in entries]) if entries else np.array([])
    )
    spds_list = [e["speeds"] for e in entries if len(e["speeds"]) > 0]
    spds = np.concatenate(spds_list) if spds_list else np.array([])
    return durs, spds


def _load_files_dialog(title: str) -> list:
    root = _tk_root()
    paths = filedialog.askopenfilenames(
        parent=root,
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    return [Path(p) for p in paths]


def _ingest_paths(paths: list, group_key: str):
    """Load CSVs into the given group, avoiding name collisions across all groups."""
    existing = {e["name"] for g in groups.values() for e in g}
    for path in paths:
        name = path.stem
        unique = name
        n = 2
        while unique in existing:
            unique = f"{name}_{n}"
            n += 1
        try:
            df = load_csv(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            continue
        color = _color_for(group_key, len(groups[group_key]))
        entry = _build_entry(unique, df, color)
        groups[group_key].append(entry)
        existing.add(unique)
        print(
            f"  ✓ Group {group_key}: {unique}  "
            f"(n={len(entry['durations'])}  μ={entry['durations'].mean():.3f}s)"
        )


# ── Overview figure ───────────────────────────────────────────────────────────


def build_overview_figure():
    from matplotlib.widgets import Button

    state = {"combined": False, "standardised": False, "skip_first_move": False}

    fig = plt.figure(figsize=(14, 7), num="Overview")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.22, wspace=0.35)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Button axes — stored so we can show/hide them
    ax_btn_mode = fig.add_axes([0.05, 0.05, 0.14, 0.07])
    ax_btn_std = fig.add_axes([0.21, 0.05, 0.13, 0.07])
    ax_btn_skip = fig.add_axes([0.36, 0.05, 0.14, 0.07])
    ax_btn_add_b = fig.add_axes([0.52, 0.05, 0.13, 0.07])
    ax_btn_add_a = fig.add_axes([0.67, 0.05, 0.10, 0.07])
    ax_btn_ren_a = fig.add_axes([0.79, 0.05, 0.09, 0.07])
    ax_btn_ren_b = fig.add_axes([0.90, 0.05, 0.09, 0.07])

    btn_mode = Button(
        ax_btn_mode, "Switch to: Combined", color="#E8EDF7", hovercolor="#C8D4F0"
    )
    btn_std = Button(
        ax_btn_std, "Standardise: Off", color="#F0EDE8", hovercolor="#DDD5C8"
    )
    btn_skip = Button(
        ax_btn_skip, "Skip 1st Move: Off", color="#F0EDE8", hovercolor="#DDD5C8"
    )
    btn_add_b = Button(
        ax_btn_add_b, "Add Group B…", color="#FFF3E0", hovercolor="#FFE0B2"
    )
    btn_add_a = Button(ax_btn_add_a, "Add to A…", color="#E3F2FD", hovercolor="#BBDEFB")
    btn_ren_a = Button(ax_btn_ren_a, "Rename A", color="#F3E5F5", hovercolor="#E1BEE7")
    btn_ren_b = Button(ax_btn_ren_b, "Rename B", color="#FCE4EC", hovercolor="#F8BBD0")

    _btn_axes = [
        ax_btn_mode,
        ax_btn_std,
        ax_btn_skip,
        ax_btn_add_b,
        ax_btn_add_a,
        ax_btn_ren_a,
        ax_btn_ren_b,
    ]

    for b in (btn_mode, btn_std, btn_skip, btn_add_b, btn_add_a, btn_ren_a, btn_ren_b):
        b.label.set_fontsize(9)
    fig._buttons = [
        btn_mode,
        btn_std,
        btn_skip,
        btn_add_b,
        btn_add_a,
        btn_ren_a,
        btn_ren_b,
    ]

    def _maybe_std(data):
        mu, sigma = data.mean(), data.std()
        if state["standardised"] and sigma > 0:
            return (data - mu) / sigma, mu, sigma
        return data, mu, sigma

    def _draw_normal(ax, data, color, label, linestyle="-", add_hist=False):
        if data is None or len(data) == 0:
            return
        plot_data, mu, sigma = _maybe_std(data)
        p_mu, p_sigma = plot_data.mean(), plot_data.std()
        if p_sigma == 0 or not np.isfinite(p_sigma):
            ax.axvline(
                p_mu,
                color=color,
                linewidth=2.2,
                linestyle="--",
                label=f"{label}  μ={mu:.3f}  (no spread)",
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
                alpha=0.18,
                edgecolor="white",
                linewidth=0.8,
            )
        ax.plot(
            x,
            stats.norm.pdf(x, p_mu, p_sigma),
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=f"{label}  μ={mu:.3f}  σ={sigma:.3f}",
        )
        ax.axvline(p_mu, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    def _style(a, xlabel, title):
        if state["standardised"]:
            xlabel = f"{xlabel}  [z-score]"
        a.set_xlabel(xlabel, fontsize=11)
        a.set_ylabel("Probability Density", fontsize=11)
        a.set_title(title, fontsize=12, fontweight="bold")
        n_total = sum(len(groups[k]) for k in ("A", "B"))
        if state["combined"]:
            a.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.9)
        else:
            a.legend(
                fontsize=9,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=max(1, min(3, n_total // 2 + 1)),
                frameon=True,
                framealpha=0.9,
            )
        a.grid(axis="y", linestyle="--", alpha=0.4)
        a.spines[["top", "right"]].set_visible(False)
        a.set_facecolor(BG)

    def _active(arr):
        if state["skip_first_move"] and len(arr) > 1:
            return arr[1:]
        return arr

    def redraw():
        ax1.cla()
        ax2.cla()

        if state["combined"]:
            # One pooled curve per group
            for gk in ("A", "B"):
                if not groups[gk]:
                    continue
                col = groups[gk][0]["color"]
                lbl = group_titles[gk]
                durs, spds = _pool_group(gk)
                _draw_normal(ax1, _active(durs), col, lbl, add_hist=True)
                _draw_normal(ax2, _active(spds), col, lbl, add_hist=True)
            subtitle = "Combined (per group)"
        else:
            # One curve per experiment; Group B dashed to distinguish at a glance
            for gk in ("A", "B"):
                ls = "-" if gk == "A" else "--"
                for e in groups[gk]:
                    _draw_normal(
                        ax1,
                        _active(e["durations"]),
                        e["color"],
                        e["name"],
                        linestyle=ls,
                    )
                    _draw_normal(
                        ax2, _active(e["speeds"]), e["color"], e["name"], linestyle=ls
                    )
            subtitle = "Per Experiment"

        dur_xl = "Duration (s)" + (
            "  [1st excluded]" if state["skip_first_move"] else ""
        )
        spd_xl = "Speed (m / s)" + (
            "  [1st move excluded]" if state["skip_first_move"] else ""
        )
        _style(ax1, dur_xl, "Distribution of Run Durations")
        _style(ax2, spd_xl, "Distribution of Within-Run Speeds")

        tags = (["standardised"] if state["standardised"] else []) + (
            ["1st excluded"] if state["skip_first_move"] else []
        )
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        n_a, n_b = len(groups["A"]), len(groups["B"])
        ta, tb = group_titles["A"], group_titles["B"]
        group_str = f"{ta}: {n_a} exp." + (
            f"  |  {tb}: {n_b} exp." if n_b else "  (add Group B to compare)"
        )
        fig.suptitle(
            f"Robot Arm Experiments — {subtitle}{tag_str}\n{group_str}",
            fontsize=13,
            fontweight="bold",
        )
        fig.canvas.draw_idle()

    fig._redraw = redraw

    def on_toggle_mode(event):
        state["combined"] = not state["combined"]
        btn_mode.label.set_text(
            "Switch to: Per Exp." if state["combined"] else "Switch to: Combined"
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

    def _add_and_refresh(group_key: str, title: str):
        global _comparison_fig
        paths = _load_files_dialog(title)
        if not paths:
            return
        # Ask for a group title the first time files are added to this group
        if not groups[group_key]:
            default = "Group A" if group_key == "A" else "Group B"
            group_titles[group_key] = _ask_title(group_key, default)
        _ingest_paths(paths, group_key)
        redraw()
        # Rebuild comparison figure whenever both groups have data
        if groups["A"] and groups["B"]:
            if _comparison_fig is not None:
                try:
                    plt.close(_comparison_fig)
                except Exception:
                    pass
            _comparison_fig = build_comparison_figure()
            _comparison_fig.canvas.draw_idle()
            _print_anova()

    def _rename_group(group_key: str):
        global _comparison_fig
        current = group_titles[group_key]
        new_title = _ask_title(group_key, current)
        if new_title == current:
            return
        group_titles[group_key] = new_title
        redraw()
        if groups["A"] and groups["B"] and _comparison_fig is not None:
            try:
                plt.close(_comparison_fig)
            except Exception:
                pass
            _comparison_fig = build_comparison_figure()
            _comparison_fig.canvas.draw_idle()

    btn_mode.on_clicked(on_toggle_mode)
    btn_std.on_clicked(on_toggle_std)
    btn_skip.on_clicked(on_toggle_skip)
    btn_add_b.on_clicked(lambda e: _add_and_refresh("B", "Select Group B CSV file(s)"))
    btn_add_a.on_clicked(
        lambda e: _add_and_refresh("A", "Add more Group A CSV file(s)")
    )
    btn_ren_a.on_clicked(lambda e: _rename_group("A"))
    btn_ren_b.on_clicked(lambda e: _rename_group("B"))

    # ── Hide/show buttons (press H) ──────────────────────────────────────────
    state["btns_visible"] = True

    def _toggle_buttons(event=None):
        visible = not state["btns_visible"]
        state["btns_visible"] = visible
        for ax in _btn_axes:
            ax.set_visible(visible)
        # Shrink bottom margin when hidden so plots use the full height,
        # restore it when shown so buttons don't overlap the axes.
        fig.subplots_adjust(bottom=0.22 if visible else 0.05)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "key_press_event", lambda e: _toggle_buttons() if e.key == "h" else None
    )

    redraw()
    return fig


# ── Comparison figure (Group A pooled vs Group B pooled) ──────────────────────


def build_comparison_figure():
    dur_a, spd_a = _pool_group("A")
    dur_b, spd_b = _pool_group("B")

    n_a_exp, n_b_exp = len(groups["A"]), len(groups["B"])
    ta, tb = group_titles["A"], group_titles["B"]
    label_a = f"{ta}  ({n_a_exp} exp, n={len(dur_a)})"
    label_b = f"{tb}  ({n_b_exp} exp, n={len(dur_b)})"
    col_a, col_b = PALETTE_A[0], PALETTE_B[0]

    fig = plt.figure(figsize=(16, 14), num="Comparison")
    fig.suptitle(f"{ta} vs {tb} — Pooled Comparison", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.65, wspace=0.35)

    # ── Row 0: Duration KDE overlay ──
    ax_kde = fig.add_subplot(gs[0, :])
    ax_kde.set_facecolor(BG)
    for durs, col, lbl in [(dur_a, col_a, label_a), (dur_b, col_b, label_b)]:
        mu, sigma = durs.mean(), durs.std()
        x = np.linspace(max(0, durs.min() - 2), durs.max() + 2, 300)
        ax_kde.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color=col,
            linewidth=2.5,
            label=f"{lbl}  μ={mu:.2f}s  σ={sigma:.2f}s",
        )
        ax_kde.fill_between(x, stats.norm.pdf(x, mu, sigma), alpha=0.12, color=col)
        ax_kde.axvline(mu, color=col, linestyle=":", linewidth=1.4, alpha=0.7)
    ax_kde.set_xlabel("Duration (s)", fontsize=11)
    ax_kde.set_ylabel("Probability Density", fontsize=11)
    ax_kde.set_title(
        f"Duration Distributions — {ta} vs {tb} (pooled)",
        fontsize=12,
        fontweight="bold",
    )
    ax_kde.legend(fontsize=10)
    ax_kde.grid(axis="y", linestyle="--", alpha=0.4)
    ax_kde.spines[["top", "right"]].set_visible(False)

    # ── Row 1a: Duration box-plot ──
    ax_db = fig.add_subplot(gs[1, 0])
    ax_db.set_facecolor(BG)
    bp1 = ax_db.boxplot(
        [dur_a, dur_b], patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )
    for patch, col in zip(bp1["boxes"], [col_a, col_b]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax_db.set_xticks([1, 2])
    ax_db.set_xticklabels([label_a, label_b], fontsize=9)
    ax_db.set_ylabel("Duration (s)", fontsize=11)
    ax_db.set_title("Duration Box-Plot", fontsize=12, fontweight="bold")
    ax_db.grid(axis="y", linestyle="--", alpha=0.4)
    ax_db.spines[["top", "right"]].set_visible(False)
    if len(dur_a) > 0 and len(dur_b) > 0:
        ax_db.text(
            0.5,
            -0.28,
            anova_summary([dur_a, dur_b], "Duration"),
            transform=ax_db.transAxes,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEF2FF", ec=col_a, alpha=0.9),
        )

    # ── Row 1b: Speed box-plot ──
    ax_sb = fig.add_subplot(gs[1, 1])
    ax_sb.set_facecolor(BG)
    if len(spd_a) > 0 and len(spd_b) > 0:
        bp2 = ax_sb.boxplot(
            [spd_a, spd_b],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, col in zip(bp2["boxes"], [col_a, col_b]):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax_sb.set_xticks([1, 2])
        ax_sb.set_xticklabels([label_a, label_b], fontsize=9)
        ax_sb.text(
            0.5,
            -0.28,
            anova_summary([spd_a, spd_b], "Speed"),
            transform=ax_sb.transAxes,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF5EE", ec=col_b, alpha=0.9),
        )
    ax_sb.set_ylabel("Within-Run Speed (units / s)", fontsize=11)
    ax_sb.set_title("Speed Box-Plot", fontsize=12, fontweight="bold")
    ax_sb.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sb.spines[["top", "right"]].set_visible(False)

    # ── Row 2: Group mean speed per move with ±1 SD band ──
    ax_sl = fig.add_subplot(gs[2, :])
    ax_sl.set_facecolor(BG)
    for gk, col, lbl in [("A", col_a, ta), ("B", col_b, tb)]:
        entries = [e for e in groups[gk] if len(e["speeds"]) > 0]
        if not entries:
            continue
        max_len = max(len(e["speeds"]) for e in entries)
        matrix = np.full((len(entries), max_len), np.nan)
        for i, e in enumerate(entries):
            matrix[i, : len(e["speeds"])] = e["speeds"]
        mean_spd = np.nanmean(matrix, axis=0)
        std_spd = np.nanstd(matrix, axis=0)
        xs = np.arange(1, max_len + 1)
        ax_sl.plot(
            xs,
            mean_spd,
            color=col,
            linewidth=2.5,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=col,
            markeredgewidth=2,
            label=lbl,
            zorder=3,
        )
        ax_sl.fill_between(
            xs,
            mean_spd - std_spd,
            mean_spd + std_spd,
            color=col,
            alpha=0.15,
            label=f"{lbl} ±1 SD",
        )
    ax_sl.set_xlabel("Move Index (within run)", fontsize=11)
    ax_sl.set_ylabel("Mean Speed (units / s)", fontsize=11)
    ax_sl.set_title(
        "Group Mean Speed per Move  (±1 SD band)", fontsize=12, fontweight="bold"
    )
    ax_sl.legend(fontsize=9)
    ax_sl.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sl.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def _print_anova():
    dur_a, spd_a = _pool_group("A")
    dur_b, spd_b = _pool_group("B")
    ta, tb = group_titles["A"], group_titles["B"]
    print("\n" + anova_summary([dur_a, dur_b], f"Duration ({ta} vs {tb})"))
    if len(spd_a) and len(spd_b):
        print(anova_summary([spd_a, spd_b], f"Speed ({ta} vs {tb})"))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    root = _tk_root()

    paths = filedialog.askopenfilenames(
        parent=root,
        title="Select Group A CSV file(s)  [hold Ctrl/Cmd for multiple]",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if not paths:
        messagebox.showinfo("No files selected", "No files were selected. Exiting.")
        sys.exit(0)

    # Ask for Group A's title before loading
    group_titles["A"] = _ask_title("A", "Group A")

    print(f"\n{group_titles['A']} — loading {len(paths)} file(s):")
    _ingest_paths([Path(p) for p in paths], "A")

    if not groups["A"]:
        sys.exit(1)

    build_overview_figure()
    plt.show()


if __name__ == "__main__":
    main()
