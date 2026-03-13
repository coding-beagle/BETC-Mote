"""
plot_joint_angles.py
────────────────────
Visualises joint angles from a kinematics CSV file produced by the YuMi
joystick controller.

Usage
-----
    python plot_joint_angles.py            # opens file dialog immediately
    python plot_joint_angles.py data.csv   # loads file directly

Controls
--------
  File button  – open a new CSV via dialog
  Sliders      – trim the visible frame window (start / end)
  Checkboxes   – toggle individual joint traces
  H key        – hide / show all UI elements (leaves only the plot)
  R key        – reset sliders to full range
  Scroll wheel – zoom x-axis around the cursor

CSV expectations
----------------
The file must contain a column ``sim_time`` and at least one column named
``j1_deg`` … ``j7_deg``.  All other columns are ignored.
Optional columns used for background shading: ``phase``, ``ik_status``.
"""

from __future__ import annotations
import sys
import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import AutoMinorLocator

# ── palette ───────────────────────────────────────────────────────────────────
JOINT_COLOURS = [
    "#4e9af1",  # j1 – blue
    "#f1c94e",  # j2 – amber
    "#6dde8a",  # j3 – green
    "#f16e4e",  # j4 – orange-red
    "#bf7fff",  # j5 – purple
    "#4ef1e8",  # j6 – cyan
    "#f14e9a",  # j7 – pink
]


IK_FAIL_COLOUR = "#ff2222"
IK_DAMPED_COLOUR = "#ffaa22"
BG_COLOUR = "#ffffff"
PANEL_COLOUR = "#cacaca"
TEXT_COLOUR = "#000000"
ACCENT_COLOUR = "#4e9af1"
GRID_COLOUR = "#ffffff"
FONT_FAMILY = "monospace"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_kinematics(path: str) -> dict:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("CSV file is empty.")
    if "sim_time" not in fieldnames:
        raise ValueError("CSV must contain a 'sim_time' column.")

    deg_cols = sorted(
        [c for c in fieldnames if c.endswith("_deg") and c[0] == "j"],
        key=lambda c: int(c[1 : c.index("_")]),
    )
    if not deg_cols:
        raise ValueError("No joint angle columns (j1_deg … j7_deg) found.")

    sim_time = np.array([float(r["sim_time"]) for r in rows])
    joints = {
        col.replace("_deg", ""): np.array([float(r[col]) for r in rows])
        for col in deg_cols
    }
    phase = [r.get("phase", "") for r in rows] if "phase" in fieldnames else None
    ik_status = (
        [r.get("ik_status", "") for r in rows] if "ik_status" in fieldnames else None
    )

    return {
        "sim_time": sim_time,
        "joints": joints,
        "phase": phase,
        "ik_status": ik_status,
        "n_frames": len(rows),
        "path": path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────


class JointAngleViewer:
    PANEL_WIDTH = 220

    def __init__(self, root: tk.Tk, initial_file: Optional[str] = None):
        self.root = root
        self._data = None
        self._ui_visible = True

        self.root.title("Joint Angle Viewer")
        # self.root.configure(bg=BG_COLOUR)
        self.root.minsize(900, 520)

        self._build_layout()
        self._bind_keys()

        if initial_file and os.path.isfile(initial_file):
            self._load_file(initial_file)
        else:
            self._prompt_open_file()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self._pane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=BG_COLOUR,
            sashwidth=4,
            sashrelief=tk.FLAT,
        )
        self._pane.pack(fill=tk.BOTH, expand=True)

        self._panel = tk.Frame(
            self._pane, bg=PANEL_COLOUR, width=self.PANEL_WIDTH, padx=8, pady=8
        )
        self._panel.pack_propagate(False)
        self._pane.add(self._panel, minsize=self.PANEL_WIDTH)
        self._build_panel()

        self._plot_frame = tk.Frame(self._pane, bg=BG_COLOUR)
        self._pane.add(self._plot_frame, minsize=400)

        self._fig, self._ax = plt.subplots(figsize=(10, 5))
        self._fig.patch.set_facecolor(BG_COLOUR)
        self._ax.set_facecolor(BG_COLOUR)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._toolbar = NavigationToolbar2Tk(self._canvas, self._plot_frame)
        self._toolbar.configure(bg=PANEL_COLOUR)
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._canvas.mpl_connect("scroll_event", self._on_scroll)

    def _build_panel(self):
        p = self._panel

        tk.Button(
            p,
            text="📂  Open CSV",
            command=self._prompt_open_file,
            bg="#2a3a4a",
            fg=TEXT_COLOUR,
            activebackground="#3a5a7a",
            activeforeground="white",
            relief=tk.FLAT,
            font=(FONT_FAMILY, 10),
            padx=6,
            pady=4,
        ).pack(fill=tk.X, pady=(0, 4))

        self._file_label = tk.Label(
            p,
            text="No file loaded",
            bg=PANEL_COLOUR,
            fg="#666677",
            font=(FONT_FAMILY, 9),
            wraplength=self.PANEL_WIDTH - 20,
            justify=tk.LEFT,
        )
        self._file_label.pack(fill=tk.X, pady=(0, 8))

        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ── frame range ───────────────────────────────────────────────────────
        tk.Label(
            p,
            text="FRAME RANGE",
            bg=PANEL_COLOUR,
            fg=ACCENT_COLOUR,
            font=(FONT_FAMILY, 9, "bold"),
        ).pack(anchor=tk.W, pady=(6, 2))

        self._start_var = tk.IntVar(value=0)
        self._end_var = tk.IntVar(value=100)

        for label_text, var, slider_attr, label_attr, cb in [
            (
                "Start",
                self._start_var,
                "_start_slider",
                "_start_label",
                self._on_start_slider,
            ),
            ("End", self._end_var, "_end_slider", "_end_label", self._on_end_slider),
        ]:
            row = tk.Frame(p, bg=PANEL_COLOUR)
            row.pack(fill=tk.X)
            tk.Label(
                row,
                text=label_text,
                bg=PANEL_COLOUR,
                fg=TEXT_COLOUR,
                font=(FONT_FAMILY, 9),
                width=5,
                anchor=tk.W,
            ).pack(side=tk.LEFT)
            lbl = tk.Label(
                row,
                text=str(var.get()),
                bg=PANEL_COLOUR,
                fg=TEXT_COLOUR,
                font=(FONT_FAMILY, 9),
                width=6,
                anchor=tk.E,
            )
            lbl.pack(side=tk.RIGHT)
            setattr(self, label_attr, lbl)

            slider = tk.Scale(
                p,
                from_=0,
                to=100,
                orient=tk.HORIZONTAL,
                variable=var,
                showvalue=False,
                bg=PANEL_COLOUR,
                fg=TEXT_COLOUR,
                troughcolor="#333344",
                highlightthickness=0,
                command=cb,
            )
            slider.pack(fill=tk.X)
            setattr(self, slider_attr, slider)

        tk.Button(
            p,
            text="Reset range",
            command=self._reset_range,
            bg="#2a2a3a",
            fg="#888899",
            activebackground="#3a3a4a",
            activeforeground=TEXT_COLOUR,
            relief=tk.FLAT,
            font=(FONT_FAMILY, 9),
            pady=2,
        ).pack(fill=tk.X, pady=(2, 8))

        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ── joint checkboxes ──────────────────────────────────────────────────
        tk.Label(
            p,
            text="JOINTS",
            bg=PANEL_COLOUR,
            fg=ACCENT_COLOUR,
            font=(FONT_FAMILY, 9, "bold"),
        ).pack(anchor=tk.W, pady=(6, 2))

        self._joint_vars = {}
        self._checkbox_frame = tk.Frame(p, bg=PANEL_COLOUR)
        self._checkbox_frame.pack(fill=tk.X)

        btn_row = tk.Frame(p, bg=PANEL_COLOUR)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        for txt, cmd in [
            ("All", self._select_all_joints),
            ("None", self._deselect_all_joints),
        ]:
            tk.Button(
                btn_row,
                text=txt,
                command=cmd,
                bg="#2a2a3a",
                fg="#888899",
                activebackground="#3a3a4a",
                activeforeground=TEXT_COLOUR,
                relief=tk.FLAT,
                font=(FONT_FAMILY, 9),
                padx=4,
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        tk.Label(
            p,
            text="H  →  hide/show UI\nR  →  reset frame range",
            bg=PANEL_COLOUR,
            fg="#555566",
            font=(FONT_FAMILY, 8),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _populate_checkboxes(self):
        for w in self._checkbox_frame.winfo_children():
            w.destroy()
        self._joint_vars.clear()

        if self._data is None:
            return

        for idx, joint_name in enumerate(self._data["joints"]):
            var = tk.BooleanVar(value=True)
            var.trace_add("write", lambda *_, jn=joint_name: self._redraw())
            self._joint_vars[joint_name] = var

            col = JOINT_COLOURS[idx % len(JOINT_COLOURS)]
            row = tk.Frame(self._checkbox_frame, bg=PANEL_COLOUR)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, bg=col, width=2, relief=tk.FLAT).pack(
                side=tk.LEFT, padx=(0, 4)
            )
            tk.Checkbutton(
                row,
                text=joint_name.upper(),
                variable=var,
                bg=PANEL_COLOUR,
                fg=TEXT_COLOUR,
                selectcolor="#333344",
                activebackground=PANEL_COLOUR,
                activeforeground="white",
                font=(FONT_FAMILY, 10),
                anchor=tk.W,
            ).pack(side=tk.LEFT, fill=tk.X)

    # ── file loading ──────────────────────────────────────────────────────────

    def _prompt_open_file(self):
        path = filedialog.askopenfilename(
            title="Select kinematics CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            data = load_kinematics(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self._data = data
        n = data["n_frames"]

        for slider, lbl, val in [
            (self._start_slider, self._start_label, 0),
            (self._end_slider, self._end_label, n - 1),
        ]:
            slider.config(to=n - 1)
            lbl.config(text=str(val))

        self._start_var.set(0)
        self._end_var.set(n - 1)
        self._file_label.config(text=os.path.basename(path), fg=TEXT_COLOUR)

        self._populate_checkboxes()
        self._redraw()

    # ── slider callbacks ──────────────────────────────────────────────────────

    def _on_start_slider(self, val):
        v = int(val)
        if self._data and v >= self._end_var.get():
            v = max(0, self._end_var.get() - 1)
            self._start_var.set(v)
        self._start_label.config(text=str(v))
        self._redraw()

    def _on_end_slider(self, val):
        v = int(val)
        if self._data and v <= self._start_var.get():
            v = min(self._data["n_frames"] - 1, self._start_var.get() + 1)
            self._end_var.set(v)
        self._end_label.config(text=str(v))
        self._redraw()

    def _reset_range(self):
        if self._data is None:
            return
        n = self._data["n_frames"]
        self._start_var.set(0)
        self._end_var.set(n - 1)
        self._start_label.config(text="0")
        self._end_label.config(text=str(n - 1))
        self._redraw()

    def _select_all_joints(self):
        for var in self._joint_vars.values():
            var.set(True)

    def _deselect_all_joints(self):
        for var in self._joint_vars.values():
            var.set(False)

    # ── hotkeys ───────────────────────────────────────────────────────────────

    def _bind_keys(self):
        for key in ("<KeyPress-h>", "<KeyPress-H>"):
            self.root.bind(key, self._toggle_ui)
        for key in ("<KeyPress-r>", "<KeyPress-R>"):
            self.root.bind(key, lambda e: self._reset_range())

    def _toggle_ui(self, event=None):
        self._ui_visible = not self._ui_visible
        if self._ui_visible:
            self._pane.add(
                self._panel, before=self._plot_frame, minsize=self.PANEL_WIDTH
            )
            self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self._pane.forget(self._panel)
            self._toolbar.pack_forget()
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ── scroll-to-zoom ────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.xdata is None:
            return
        ax = self._ax
        xlim = ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        factor = 0.85 if event.button == "up" else 1.15
        ratio = (event.xdata - xlim[0]) / xrange
        nr = xrange * factor
        ax.set_xlim(event.xdata - ratio * nr, event.xdata + (1 - ratio) * nr)
        self._canvas.draw_idle()

    # ── main draw ─────────────────────────────────────────────────────────────

    def _redraw(self):
        if self._data is None:
            return

        data = self._data
        start = self._start_var.get()
        end = self._end_var.get()
        if start >= end:
            return

        t = data["sim_time"][start : end + 1]
        joints = data["joints"]
        ik = data["ik_status"]

        ax = self._ax
        ax.cla()
        ax.set_facecolor(BG_COLOUR)

        # ── STEP 1: draw traces first so y-limits come from the actual data ───
        legend_handles = []
        any_plotted = False

        for idx, (joint_name, angles) in enumerate(joints.items()):
            if not self._joint_vars.get(joint_name, tk.BooleanVar(value=False)).get():
                continue
            col = JOINT_COLOURS[idx % len(JOINT_COLOURS)]
            chunk = angles[start : end + 1]
            ax.plot(t, chunk, color=col, linewidth=1.4, alpha=0.92, zorder=3)
            legend_handles.append(mpatches.Patch(color=col, label=joint_name.upper()))
            any_plotted = True

        # ── STEP 2: lock y-limits BEFORE drawing any background fills ─────────
        # This prevents fills from distorting the axis range.
        if any_plotted:
            ax.autoscale(enable=True, axis="y")
            ylo, yhi = ax.get_ylim()
            pad = max((yhi - ylo) * 0.05, 1.0)
            ylo -= pad
            yhi += pad
        else:
            ylo, yhi = -180.0, 180.0

        ax.set_ylim(ylo, yhi)

        # ── STEP 3: IK status bands (zorder=2) ────────────────────────────────
        if ik is not None:
            ik_slice = ik[start : end + 1]
            fail_mask = np.array([s == "failed" for s in ik_slice])
            damped_mask = np.array([s == "damped" for s in ik_slice])

            if fail_mask.any():
                ax.fill_between(
                    t,
                    ylo,
                    yhi,
                    where=fail_mask,
                    color=IK_FAIL_COLOUR,
                    alpha=0.30,
                    linewidth=0,
                    zorder=2,
                )
                legend_handles.append(
                    mpatches.Patch(color=IK_FAIL_COLOUR, alpha=0.6, label="IK failed")
                )

            if damped_mask.any():
                ax.fill_between(
                    t,
                    ylo,
                    yhi,
                    where=damped_mask,
                    color=IK_DAMPED_COLOUR,
                    alpha=0.15,
                    linewidth=0,
                    zorder=2,
                )
                legend_handles.append(
                    mpatches.Patch(color=IK_DAMPED_COLOUR, alpha=0.5, label="IK damped")
                )

        # Re-assert limits; fills can nudge them slightly
        ax.set_ylim(ylo, yhi)

        # ── axes styling ──────────────────────────────────────────────────────
        ax.set_xlabel(
            "Simulation time  (s)",
            color=TEXT_COLOUR,
            fontfamily=FONT_FAMILY,
            fontsize=10,
        )
        ax.set_ylabel(
            "Joint angle  (°)", color=TEXT_COLOUR, fontfamily=FONT_FAMILY, fontsize=10
        )
        ax.tick_params(colors=TEXT_COLOUR, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOUR)
        ax.grid(
            True, color=GRID_COLOUR, linewidth=0.6, linestyle="--", alpha=0.7, zorder=0
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(
            True,
            which="minor",
            color=GRID_COLOUR,
            linewidth=0.3,
            linestyle=":",
            alpha=0.4,
            zorder=0,
        )

        ax.axhline(0, color=GRID_COLOUR, linewidth=0.8, linestyle="-", zorder=1)

        n_shown = end - start + 1
        ax.set_title(
            f"Joint Angles vs Time",
            color=TEXT_COLOUR,
            fontfamily=FONT_FAMILY,
            fontsize=10,
            pad=8,
        )

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                framealpha=0.3,
                facecolor=PANEL_COLOUR,
                edgecolor=GRID_COLOUR,
                labelcolor=TEXT_COLOUR,
                prop={"family": FONT_FAMILY, "size": 9},
            )

        if not any_plotted:
            ax.text(
                0.5,
                0.5,
                "No joints selected",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#555566",
                fontsize=14,
                fontfamily=FONT_FAMILY,
            )

        self._fig.tight_layout()
        self._canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    initial_file = sys.argv[1] if len(sys.argv) > 1 else None

    root = tk.Tk()
    root.geometry("1200x640")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TSeparator", background=GRID_COLOUR)

    JointAngleViewer(root, initial_file=initial_file)
    root.mainloop()


if __name__ == "__main__":
    main()
