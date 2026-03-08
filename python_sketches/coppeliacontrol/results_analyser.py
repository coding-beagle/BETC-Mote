"""
Robot Arm Experiment Analyser
──────────────────────────────
Run this script and use the file picker to select one or more CSV files (Group A).
Handles both Reach experiment CSVs and Transport experiment CSVs automatically.

Reach CSV columns (required):  result, duration_s, target_x, target_y, target_z
Transport CSV columns:          result, duration_s,
                                cube_x/y/z, drop_x/y/z,
                                start_x/y/z (optional),
                                dist_start_to_cube, dist_start_to_drop (optional),
                                phase_approach_s, phase_grip_s,
                                phase_carry_s, phase_place_s (optional)

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

Figure 3 — Transport Phase Analysis (press P, transport CSVs only):
  • Stacked bar charts of mean time per phase per experiment
  • Box-plots comparing phase time distributions across groups
  • Scatter: dist_start_to_cube vs duration (task difficulty proxy)
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

# ── Disable conflicting matplotlib default keybindings ───────────────────────
import matplotlib

matplotlib.rcParams["keymap.home"] = []
matplotlib.rcParams["keymap.back"] = []
matplotlib.rcParams["keymap.forward"] = []

# ── Colour palettes ───────────────────────────────────────────────────────────
PALETTE_A = ["#4C72B0", "#55A868", "#64B5CD", "#3A9E7A", "#2E6FA3", "#76B7B2"]
PALETTE_B = ["#DD8452", "#C44E52", "#E6A817", "#D45F86", "#C7622E", "#E8734A"]

# Phase colours (consistent across all transport plots)
PHASE_COLOURS = {
    "approach": "#4C72B0",
    "grip": "#55A868",
    "carry": "#E6A817",
    "place": "#C44E52",
}
PHASE_ORDER = ["approach", "grip", "carry", "place"]

BG = "#F8F9FA"

# ── Shared state ──────────────────────────────────────────────────────────────
groups: dict = {"A": [], "B": []}
group_titles: dict = {"A": "Group A", "B": "Group B"}
_comparison_fig = None


# ── CSV type detection ────────────────────────────────────────────────────────


def detect_csv_type(df: pd.DataFrame) -> str:
    """Return 'transport' or 'reach' based on column presence."""
    if "cube_x" in df.columns:
        return "transport"
    return "reach"


def _has_phase_cols(df: pd.DataFrame) -> bool:
    return "phase_approach_s" in df.columns


def _has_start_cols(df: pd.DataFrame) -> bool:
    return "dist_start_to_cube" in df.columns


# ── Colour / title helpers ────────────────────────────────────────────────────


def _color_for(group: str, index: int) -> str:
    palette = PALETTE_A if group == "A" else PALETTE_B
    return palette[index % len(palette)]


def _tk_root() -> tk.Tk:
    if not hasattr(_tk_root, "_inst") or not _tk_root._inst.winfo_exists():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        _tk_root._inst = root
    return _tk_root._inst


def _ask_title(group_key: str, default: str) -> str:
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


# ── CSV loading ───────────────────────────────────────────────────────────────


def load_csv(path: Path):
    """Return (success_df, total_row_count, csv_type). Raises if no successful rows."""
    raw = pd.read_csv(path)
    total = len(raw)
    csv_type = detect_csv_type(raw)
    df = raw[raw["result"] == "success"].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No successful runs found in {path.name}")
    return df, total, csv_type


# ── Speed computation ─────────────────────────────────────────────────────────


def compute_speeds(df: pd.DataFrame, csv_type: str) -> np.ndarray:
    """
    Reach:     speed = displacement between consecutive target positions / duration
    Transport: speed = straight-line cube→drop distance / trial duration
               (each trial is one move, so this gives m/s for the task)
    """
    if csv_type == "reach":
        pos = df[["target_x", "target_y", "target_z"]].values
        displacements = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        durations = df["duration_s"].values[1:]
        return displacements / durations
    else:
        # cube→drop straight-line distance per trial
        cube = df[["cube_x", "cube_y", "cube_z"]].values
        drop = df[["drop_x", "drop_y", "drop_z"]].values
        distances = np.linalg.norm(drop - cube, axis=1)
        durations = df["duration_s"].values
        return distances / np.where(durations > 0, durations, np.nan)


def anova_summary(groups_data: list, metric: str) -> str:
    f, p = stats.f_oneway(*groups_data)
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    return f"One-way ANOVA — {metric}\nF = {f:.3f},  p = {p:.4f}  ({sig} at α=0.05)"


def _build_entry(
    name: str, df: pd.DataFrame, color: str, total_rows: int, csv_type: str
) -> dict:
    if len(df) < 2 and csv_type == "reach":
        print(
            f"  ⚠  {name}: only {len(df)} successful trial(s) — skipping speed computation"
        )
        speeds = np.array([])
    else:
        speeds = compute_speeds(df, csv_type)

    # Phase splits — only for transport with phase columns
    phase_splits = {}
    if csv_type == "transport" and _has_phase_cols(df):
        for ph in PHASE_ORDER:
            col = f"phase_{ph}_s"
            if col in df.columns:
                phase_splits[ph] = df[col].values

    # Difficulty metrics
    dist_start_to_cube = (
        df["dist_start_to_cube"].values if _has_start_cols(df) else np.array([])
    )
    dist_start_to_drop = (
        df["dist_start_to_drop"].values
        if "dist_start_to_drop" in df.columns
        else np.array([])
    )

    return dict(
        name=name,
        df=df,
        color=color,
        durations=df["duration_s"].values,
        speeds=speeds,
        n_success=len(df),
        n_total=total_rows,
        csv_type=csv_type,
        phase_splits=phase_splits,  # dict: phase -> np.ndarray of per-trial times
        dist_start_to_cube=dist_start_to_cube,
        dist_start_to_drop=dist_start_to_drop,
    )


def _pool_group(group_key: str):
    entries = groups[group_key]
    durs = (
        np.concatenate([e["durations"] for e in entries]) if entries else np.array([])
    )
    spds_list = [e["speeds"] for e in entries if len(e["speeds"]) > 0]
    spds = np.concatenate(spds_list) if spds_list else np.array([])
    return durs, spds


def _pool_phase_splits(group_key: str) -> dict:
    """Pool per-trial phase times across all transport entries in a group."""
    result = {ph: [] for ph in PHASE_ORDER}
    for e in groups[group_key]:
        if e["csv_type"] != "transport":
            continue
        for ph in PHASE_ORDER:
            arr = e["phase_splits"].get(ph, np.array([]))
            if len(arr):
                result[ph].extend(arr.tolist())
    return {ph: np.array(v) for ph, v in result.items()}


def _load_files_dialog(title: str) -> list:
    root = _tk_root()
    paths = filedialog.askopenfilenames(
        parent=root,
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    return [Path(p) for p in paths]


def _ingest_paths(paths: list, group_key: str):
    existing = {e["name"] for g in groups.values() for e in g}
    for path in paths:
        name = path.stem
        unique = name
        n = 2
        while unique in existing:
            unique = f"{name}_{n}"
            n += 1
        try:
            df, total_rows, csv_type = load_csv(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            continue
        color = _color_for(group_key, len(groups[group_key]))
        entry = _build_entry(unique, df, color, total_rows, csv_type)
        groups[group_key].append(entry)
        existing.add(unique)
        type_tag = f"[{csv_type}]"
        print(
            f"  ✓ Group {group_key}: {unique}  {type_tag}  "
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
    fig._buttons = [
        btn_mode,
        btn_std,
        btn_skip,
        btn_add_b,
        btn_add_a,
        btn_ren_a,
        btn_ren_b,
    ]
    for b in fig._buttons:
        b.label.set_fontsize(9)

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

    def _active(arr, csv_type="reach"):
        # skip_first_move only meaningful for reach (sequential targets)
        if state["skip_first_move"] and csv_type == "reach" and len(arr) > 1:
            return arr[1:]
        return arr

    def redraw():
        ax1.cla()
        ax2.cla()

        if state["combined"]:
            for gk in ("A", "B"):
                if not groups[gk]:
                    continue
                col = groups[gk][0]["color"]
                lbl = group_titles[gk]
                durs, spds = _pool_group(gk)
                _draw_normal(ax1, durs, col, lbl, add_hist=True)
                _draw_normal(ax2, spds, col, lbl, add_hist=True)
            subtitle = "Combined (per group)"
        else:
            for gk in ("A", "B"):
                ls = "-" if gk == "A" else "--"
                for e in groups[gk]:
                    _draw_normal(
                        ax1,
                        _active(e["durations"], e["csv_type"]),
                        e["color"],
                        e["name"],
                        linestyle=ls,
                    )
                    _draw_normal(
                        ax2,
                        _active(e["speeds"], e["csv_type"]),
                        e["color"],
                        e["name"],
                        linestyle=ls,
                    )
            subtitle = "Per Experiment"

        # Right-axis label differs by dataset type
        all_types = {e["csv_type"] for g in groups.values() for e in g}
        if "transport" in all_types and "reach" not in all_types:
            spd_label = "Task Speed (cube→drop dist / duration, m/s)"
        elif "reach" in all_types and "transport" not in all_types:
            spd_label = "Speed (units / s)"
        else:
            spd_label = "Speed (units / s  or  m/s)"

        dur_xl = "Duration (s)" + (
            "  [1st excluded]" if state["skip_first_move"] else ""
        )
        _style(ax1, dur_xl, "Distribution of Run Durations")
        _style(ax2, spd_label, "Distribution of Within-Run Speeds")

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
        if not groups[group_key]:
            default = "Group A" if group_key == "A" else "Group B"
            group_titles[group_key] = _ask_title(group_key, default)
        _ingest_paths(paths, group_key)
        redraw()
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

    def _refocus(event=None):
        fig.canvas.get_tk_widget().focus_set()

    for _b in fig._buttons:
        _b.on_clicked(_refocus)

    state["btns_visible"] = True

    def _toggle_buttons(event=None):
        visible = not state["btns_visible"]
        state["btns_visible"] = visible
        for ax in _btn_axes:
            ax.set_visible(visible)
        fig.subplots_adjust(bottom=0.22 if visible else 0.05)
        fig.canvas.draw_idle()

    _metrics_state = {"fig": None}
    _phase_state = {"fig": None}

    def _toggle_metrics(event=None):
        existing = _metrics_state["fig"]
        if existing is not None and plt.fignum_exists(existing.number):
            plt.close(existing)
            _metrics_state["fig"] = None
        else:
            mfig = build_metrics_figure()
            _metrics_state["fig"] = mfig
            mfig.canvas.manager.show()
            mfig.canvas.draw_idle()

    def _toggle_phase(event=None):
        existing = _phase_state["fig"]
        if existing is not None and plt.fignum_exists(existing.number):
            plt.close(existing)
            _phase_state["fig"] = None
        else:
            transport_entries = [
                e for g in groups.values() for e in g if e["csv_type"] == "transport"
            ]
            if not transport_entries:
                messagebox.showinfo(
                    "No transport data", "Load at least one transport CSV first."
                )
                return
            pfig = build_phase_figure()
            _phase_state["fig"] = pfig
            pfig.canvas.manager.show()
            pfig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "h":
            _toggle_buttons()
        elif event.key == "m":
            _toggle_metrics()
        elif event.key == "p":
            _toggle_phase()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    redraw()
    return fig


# ── Comparison figure ─────────────────────────────────────────────────────────


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

    # Row 0: Duration KDE overlay
    ax_kde = fig.add_subplot(gs[0, :])
    ax_kde.set_facecolor(BG)
    for durs, col, lbl in [(dur_a, col_a, label_a), (dur_b, col_b, label_b)]:
        if len(durs) == 0:
            continue
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

    # Row 1a: Duration box-plot
    ax_db = fig.add_subplot(gs[1, 0])
    ax_db.set_facecolor(BG)
    if len(dur_a) and len(dur_b):
        bp1 = ax_db.boxplot(
            [dur_a, dur_b],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, col in zip(bp1["boxes"], [col_a, col_b]):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax_db.set_xticks([1, 2])
        ax_db.set_xticklabels([label_a, label_b], fontsize=9)
        ax_db.text(
            0.5,
            -0.28,
            anova_summary([dur_a, dur_b], "Duration"),
            transform=ax_db.transAxes,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEF2FF", ec=col_a, alpha=0.9),
        )
    ax_db.set_ylabel("Duration (s)", fontsize=11)
    ax_db.set_title("Duration Box-Plot", fontsize=12, fontweight="bold")
    ax_db.grid(axis="y", linestyle="--", alpha=0.4)
    ax_db.spines[["top", "right"]].set_visible(False)

    # Row 1b: Speed box-plot
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
    ax_sb.set_ylabel("Speed (m/s or units/s)", fontsize=11)
    ax_sb.set_title("Speed Box-Plot", fontsize=12, fontweight="bold")
    ax_sb.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sb.spines[["top", "right"]].set_visible(False)

    # Row 2: Group mean speed per move / trial with ±1 SD band
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
    ax_sl.set_xlabel("Move / Trial Index", fontsize=11)
    ax_sl.set_ylabel("Mean Speed (m/s or units/s)", fontsize=11)
    ax_sl.set_title(
        "Group Mean Speed per Move  (±1 SD band)", fontsize=12, fontweight="bold"
    )
    ax_sl.legend(fontsize=9)
    ax_sl.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sl.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


# ── Metrics figure ────────────────────────────────────────────────────────────


def build_metrics_figure():
    has_b = bool(groups["B"])
    n_cols = 2 if has_b else 1
    ta, tb = group_titles["A"], group_titles["B"]

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(7 * n_cols, 9),
        num="Success Rate",
        gridspec_kw={"height_ratios": [1, 1.6]},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Success Rate Analysis", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor(BG)

    col_info = [("A", ta, PALETTE_A[0])]
    if has_b:
        col_info.append(("B", tb, PALETTE_B[0]))

    for col_idx, (gk, title, col) in enumerate(col_info):
        ax = axes[0, col_idx]
        ax.set_facecolor(BG)
        entries = groups[gk]
        total = sum(e["n_total"] for e in entries)
        success = sum(e["n_success"] for e in entries)
        failure = total - success
        rate = success / total * 100 if total else 0

        wedge_colors = [col, "#D9D9D9"]
        _, _, autotexts = ax.pie(
            [success, failure],
            labels=["Success", "Failure"],
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10),
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_title(
            f"{title}\n{success}/{total} successful  ({rate:.1f}%)",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )

    return fig


# ── Phase analysis figure (transport only, press P) ───────────────────────────


def build_phase_figure():
    """
    Four-row transport phase analysis:

    Row 0 (full width): Stacked bar — mean time per phase, one bar per experiment.
                        Bars are grouped by group (A left, B right).
    Row 1:              Pie charts — average proportion of trial time spent in each
                        phase.  One pie per loaded group, plus a combined pie when
                        both groups are present.
    Row 2 left:         Box-plots of per-trial phase times pooled across all
                        transport entries, one box per phase, both groups overlaid.
    Row 2 right:        Scatter — dist_start_to_cube vs trial duration,
                        coloured by group, with linear regression lines.
                        (Only shown if dist columns are present.)
    """
    ta, tb = group_titles["A"], group_titles["B"]
    has_b_group = bool(groups["B"]) and any(
        e["csv_type"] == "transport" for e in groups["B"]
    )

    # Pie row: 2 pies if only A, 3 if both A and B
    n_pies = 3 if has_b_group else 2  # per-group + combined  (or just A + combined)
    pie_cols = n_pies

    fig = plt.figure(figsize=(16, 15), num="Phase Analysis")
    fig.suptitle("Transport Phase Analysis", fontsize=14, fontweight="bold")

    # GridSpec: row 0 = stacked bar (full width), row 1 = pies, row 2 = box + scatter
    gs = gridspec.GridSpec(
        3,
        pie_cols,
        figure=fig,
        hspace=0.65,
        wspace=0.38,
        height_ratios=[1.1, 1.2, 1.1],
    )

    ax_stack = fig.add_subplot(gs[0, :])
    ax_box = fig.add_subplot(gs[2, : pie_cols // 2])
    ax_scatter = fig.add_subplot(gs[2, pie_cols // 2 :])

    for ax in (ax_stack, ax_box, ax_scatter):
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)

    # Pie axes are created later, after we know how many groups have data

    # Collect all transport entries across both groups, tagged with group
    all_transport = []
    for gk in ("A", "B"):
        for e in groups[gk]:
            if e["csv_type"] == "transport":
                all_transport.append((gk, e))

    if not all_transport:
        ax_stack.text(
            0.5,
            0.5,
            "No transport data loaded.",
            ha="center",
            va="center",
            transform=ax_stack.transAxes,
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    # ── Row 0: Stacked bar — mean phase time per experiment ──────────────────
    bar_labels = []
    bar_bottoms = np.zeros(len(all_transport))
    phase_handles = {}

    for ph_idx, ph in enumerate(PHASE_ORDER):
        ph_means = []
        for gk, e in all_transport:
            arr = e["phase_splits"].get(ph, np.array([]))
            ph_means.append(arr.mean() if len(arr) > 0 else 0.0)
        ph_col = PHASE_COLOURS[ph]
        bars = ax_stack.bar(
            range(len(all_transport)),
            ph_means,
            bottom=bar_bottoms,
            color=ph_col,
            alpha=0.82,
            label=ph.capitalize(),
            edgecolor="white",
            linewidth=0.6,
        )
        phase_handles[ph] = bars[0]
        bar_bottoms += np.array(ph_means)

    # Build x-axis labels, mark group boundary
    x_labels = []
    for gk, e in all_transport:
        prefix = "A·" if gk == "A" else "B·"
        x_labels.append(prefix + e["name"])

    ax_stack.set_xticks(range(len(all_transport)))
    ax_stack.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax_stack.set_ylabel("Mean Time per Phase (s)", fontsize=11)
    ax_stack.set_title(
        "Mean Phase Time per Experiment (stacked)", fontsize=12, fontweight="bold"
    )
    ax_stack.legend(
        handles=list(phase_handles.values()),
        labels=[ph.capitalize() for ph in PHASE_ORDER],
        fontsize=9,
        loc="upper right",
    )
    ax_stack.grid(axis="y", linestyle="--", alpha=0.4)

    # Draw group-boundary separator line
    n_a = sum(1 for gk, _ in all_transport if gk == "A")
    if 0 < n_a < len(all_transport):
        ax_stack.axvline(
            n_a - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.5
        )
        ax_stack.text(
            n_a - 0.5,
            ax_stack.get_ylim()[1] * 0.98,
            f"← {ta}   {tb} →",
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    # ── Row 1: Pie charts — average phase-time proportions ───────────────────
    # Build one pie per group (only transport entries), plus a combined pie.
    pie_specs = []  # list of (title, {ph: mean_seconds})
    for gk, label in [("A", ta), ("B", tb)]:
        t_entries = [e for e in groups[gk] if e["csv_type"] == "transport"]
        if not t_entries:
            continue
        splits = _pool_phase_splits(gk)
        means = {
            ph: splits[ph].mean() if len(splits[ph]) > 0 else 0.0 for ph in PHASE_ORDER
        }
        if sum(means.values()) > 0:
            pie_specs.append((label, means))

    # Combined pie (all transport entries regardless of group)
    all_splits = {ph: [] for ph in PHASE_ORDER}
    for gk in ("A", "B"):
        s = _pool_phase_splits(gk)
        for ph in PHASE_ORDER:
            if len(s[ph]):
                all_splits[ph].extend(s[ph].tolist())
    combined_means = {ph: np.mean(v) if v else 0.0 for ph, v in all_splits.items()}
    if sum(combined_means.values()) > 0:
        pie_specs.append(("Combined", combined_means))

    pie_colours = [PHASE_COLOURS[ph] for ph in PHASE_ORDER]
    pie_explode = [0.03] * len(PHASE_ORDER)  # slight separation on every slice

    for pie_idx, (pie_title, means) in enumerate(pie_specs):
        col_span = pie_cols // len(pie_specs) if len(pie_specs) <= pie_cols else 1
        # Spread pies evenly; if only 1 spec, centre it
        if len(pie_specs) == 1:
            ax_pie = fig.add_subplot(gs[1, 1])
        else:
            ax_pie = fig.add_subplot(gs[1, pie_idx])

        values = [means[ph] for ph in PHASE_ORDER]
        total = sum(values)
        labels = [
            (
                f"{ph.capitalize()}\n{means[ph]:.2f}s\n({means[ph]/total*100:.1f}%)"
                if total > 0
                else ph.capitalize()
            )
            for ph in PHASE_ORDER
        ]

        wedges, texts = ax_pie.pie(
            values,
            labels=labels,
            colors=pie_colours,
            explode=pie_explode,
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.6),
            textprops=dict(fontsize=8.5),
            labeldistance=1.18,
        )
        for w in wedges:
            w.set_alpha(0.88)

        ax_pie.set_title(
            f"{pie_title}\nAvg phase breakdown  (total {total:.2f}s)",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

    # ── Row 2 left: Box-plots of per-trial phase times by group ──────────────
    has_b_transport = any(gk == "B" for gk, _ in all_transport)
    n_phase_groups = 2 if has_b_transport else 1
    positions = []
    box_data = []
    tick_pos = []
    tick_lbl = []

    group_keys_present = ["A"] + (["B"] if has_b_transport else [])
    group_cols = {"A": PALETTE_A[0], "B": PALETTE_B[0]}

    spacing = n_phase_groups + 1
    for ph_idx, ph in enumerate(PHASE_ORDER):
        base = ph_idx * spacing
        tick_pos.append(base + (n_phase_groups - 1) / 2)
        tick_lbl.append(ph.capitalize())
        for gi, gk in enumerate(group_keys_present):
            splits = _pool_phase_splits(gk)
            arr = splits.get(ph, np.array([]))
            if len(arr) == 0:
                arr = np.array([0.0])
            pos = base + gi
            positions.append(pos)
            box_data.append(arr)

    bp = ax_box.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for i, (patch, pos) in enumerate(zip(bp["boxes"], positions)):
        gk = group_keys_present[i % n_phase_groups]
        patch.set_facecolor(group_cols[gk])
        patch.set_alpha(0.65)

    ax_box.set_xticks(tick_pos)
    ax_box.set_xticklabels(tick_lbl, fontsize=10)
    ax_box.set_ylabel("Phase Duration (s)", fontsize=11)
    ax_box.set_title(
        "Phase Time Distributions by Group", fontsize=12, fontweight="bold"
    )
    ax_box.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend for group colours
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=group_cols[gk], alpha=0.65, label=group_titles[gk])
        for gk in group_keys_present
    ]
    ax_box.legend(handles=legend_patches, fontsize=9)

    # ── Row 2 right: Scatter — dist_start_to_cube vs duration ───────────────
    has_dist_data = any(len(e["dist_start_to_cube"]) > 0 for _, e in all_transport)

    if has_dist_data:
        for gk, col, lbl in [("A", PALETTE_A[0], ta), ("B", PALETTE_B[0], tb)]:
            entries_t = [e for g, e in all_transport if g == gk]
            if not entries_t:
                continue
            d_arr = np.concatenate(
                [
                    e["dist_start_to_cube"]
                    for e in entries_t
                    if len(e["dist_start_to_cube"]) > 0
                ]
            )
            dur_arr = np.concatenate(
                [e["durations"] for e in entries_t if len(e["dist_start_to_cube"]) > 0]
            )
            if len(d_arr) < 2:
                continue
            ax_scatter.scatter(
                d_arr * 100,
                dur_arr,
                color=col,
                alpha=0.55,
                s=35,
                label=lbl,
                edgecolors="white",
                linewidth=0.4,
            )
            # Linear regression
            slope, intercept, r, p_val, _ = stats.linregress(d_arr, dur_arr)
            x_fit = np.linspace(d_arr.min(), d_arr.max(), 100)
            sig_txt = "✓" if p_val < 0.05 else "✗"
            ax_scatter.plot(
                x_fit * 100,
                slope * x_fit + intercept,
                color=col,
                linewidth=1.8,
                linestyle="--",
                label=f"{lbl} fit  r={r:.2f} {sig_txt}",
            )

        ax_scatter.set_xlabel("Distance: Start → Cube (cm)", fontsize=11)
        ax_scatter.set_ylabel("Trial Duration (s)", fontsize=11)
        ax_scatter.set_title(
            "Task Difficulty: Start Distance vs Duration",
            fontsize=12,
            fontweight="bold",
        )
        ax_scatter.legend(fontsize=8)
        ax_scatter.grid(linestyle="--", alpha=0.4)
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "No start-position data available.\n"
            "(requires start_x/y/z columns in CSV)",
            ha="center",
            va="center",
            transform=ax_scatter.transAxes,
            fontsize=10,
            color="#888888",
        )
        ax_scatter.set_title(
            "Task Difficulty (no start-pos data)", fontsize=12, fontweight="bold"
        )

    fig.tight_layout()
    return fig


def _print_anova():
    dur_a, spd_a = _pool_group("A")
    dur_b, spd_b = _pool_group("B")
    ta, tb = group_titles["A"], group_titles["B"]
    if len(dur_a) and len(dur_b):
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

    group_titles["A"] = _ask_title("A", "Group A")

    print(f"\n{group_titles['A']} — loading {len(paths)} file(s):")
    _ingest_paths([Path(p) for p in paths], "A")

    if not groups["A"]:
        sys.exit(1)

    # Print keybinds reminder
    print("\nKeybinds (click plot first):")
    print("  H — toggle button bar")
    print("  M — toggle success-rate figure")
    print("  P — toggle phase analysis figure  (transport CSVs only)")

    build_overview_figure()
    plt.show()


if __name__ == "__main__":
    main()
