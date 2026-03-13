import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os

BG = "SystemButtonFace"
BG2 = "SystemButtonFace"
ACCENT = "#0078d4"
TEXT = "black"
SUBTLE = "#555555"


class PlotterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Plotter")
        self.geometry("1200x760")
        self.configure(bg=BG)
        self.df = None
        self.canvas = None
        self.toolbar = None
        self.y_checks: dict[str, tk.BooleanVar] = {}
        # draw-line state
        self._draw_mode = False
        self._draw_start = None
        self._preview_line = None
        self._drawn_lines = []
        self._ax = None
        self._fig = None
        self._annotation_color = "darkorange"
        self._build_ui()

    def _build_ui(self):
        # ── Scrollable sidebar ────────────────────────────────────────────────
        sb_outer = tk.Frame(self, width=285)
        sb_outer.pack(side="left", fill="y")
        sb_outer.pack_propagate(False)

        sb_canvas = tk.Canvas(sb_outer, width=270, highlightthickness=0)
        sb_scroll = ttk.Scrollbar(sb_outer, orient="vertical", command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=sb_scroll.set)
        sb_scroll.pack(side="right", fill="y")
        sb_canvas.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(sb_canvas, width=270)
        sb_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind(
            "<Configure>",
            lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")),
        )

        # mousewheel scrolling
        def _on_mw(event):
            sb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        sb_canvas.bind_all("<MouseWheel>", _on_mw)

        tk.Label(sidebar, text="📊 CSV Plotter", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=16, pady=(20, 2)
        )
        tk.Label(sidebar, text="Load, select, visualise", font=("Segoe UI", 9)).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        tk.Button(
            sidebar,
            text="Open File",
            font=("Segoe UI", 11, "bold"),
            padx=8,
            pady=6,
            cursor="hand2",
            command=self._open_file,
        ).pack(fill="x", padx=16, pady=(8, 4))

        self.file_label = tk.Label(
            sidebar, text="No file loaded", font=("Segoe UI", 9), wraplength=230
        )
        self.file_label.pack(padx=16, pady=(0, 10))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # X axis
        tk.Label(sidebar, text="X Axis", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(10, 2)
        )
        self.x_var = tk.StringVar()
        self.x_menu = ttk.Combobox(
            sidebar,
            textvariable=self.x_var,
            state="readonly",
            font=("Segoe UI", 10),
            width=26,
        )
        self.x_menu["values"] = ["— load a file —"]
        self.x_menu.pack(padx=16, pady=(0, 10))

        # Y axis checkboxes
        tk.Label(
            sidebar, text="Y Axis (one or more)", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=16, pady=(0, 4))

        y_outer = tk.Frame(sidebar)
        y_outer.pack(fill="x", padx=16, pady=(0, 10))

        self.y_canvas = tk.Canvas(y_outer, highlightthickness=0, height=200)
        y_scroll = ttk.Scrollbar(
            y_outer, orient="vertical", command=self.y_canvas.yview
        )
        self.y_canvas.configure(yscrollcommand=y_scroll.set)
        self.y_inner = tk.Frame(self.y_canvas)
        self.y_canvas.create_window((0, 0), window=self.y_inner, anchor="nw")
        self.y_inner.bind(
            "<Configure>",
            lambda e: self.y_canvas.configure(scrollregion=self.y_canvas.bbox("all")),
        )
        self.y_canvas.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Chart type
        tk.Label(sidebar, text="Chart Type", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(8, 2)
        )
        self.chart_var = tk.StringVar(value="Line")
        chart_menu = ttk.Combobox(
            sidebar,
            textvariable=self.chart_var,
            state="readonly",
            values=["Line", "Scatter", "Bar", "Area", "Step"],
            font=("Segoe UI", 10),
            width=26,
        )
        chart_menu.pack(padx=16, pady=(0, 12))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Labels / titles
        tk.Label(sidebar, text="Labels & Title", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(8, 2)
        )

        tk.Label(sidebar, text="Plot title", font=("Segoe UI", 9)).pack(
            anchor="w", padx=16
        )
        self.title_var = tk.StringVar()
        tk.Entry(
            sidebar, textvariable=self.title_var, font=("Segoe UI", 10), width=28
        ).pack(padx=16, pady=(2, 6))

        tk.Label(sidebar, text="X axis label", font=("Segoe UI", 9)).pack(
            anchor="w", padx=16
        )
        self.xlabel_var = tk.StringVar()
        tk.Entry(
            sidebar, textvariable=self.xlabel_var, font=("Segoe UI", 10), width=28
        ).pack(padx=16, pady=(2, 6))

        tk.Label(sidebar, text="Y axis label", font=("Segoe UI", 9)).pack(
            anchor="w", padx=16
        )
        self.ylabel_var = tk.StringVar()
        tk.Entry(
            sidebar, textvariable=self.ylabel_var, font=("Segoe UI", 10), width=28
        ).pack(padx=16, pady=(2, 10))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # X axis range
        tk.Label(sidebar, text="X Axis Range", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(8, 2)
        )
        range_row = tk.Frame(sidebar)
        range_row.pack(fill="x", padx=16, pady=(2, 12))
        tk.Label(range_row, text="From", font=("Segoe UI", 9)).pack(side="left")
        self.xmin_var = tk.StringVar()
        tk.Entry(
            range_row, textvariable=self.xmin_var, font=("Segoe UI", 10), width=8
        ).pack(side="left", padx=(4, 10))
        tk.Label(range_row, text="To", font=("Segoe UI", 9)).pack(side="left")
        self.xmax_var = tk.StringVar()
        tk.Entry(
            range_row, textvariable=self.xmax_var, font=("Segoe UI", 10), width=8
        ).pack(side="left", padx=(4, 0))

        # Buttons
        self.plot_btn = tk.Button(
            sidebar,
            text="Plot",
            font=("Segoe UI", 11, "bold"),
            padx=8,
            pady=8,
            cursor="hand2",
            state="disabled",
            command=self._plot,
        )
        self.plot_btn.pack(fill="x", padx=16, pady=(0, 4))

        self.clear_btn = tk.Button(
            sidebar,
            text="Clear",
            font=("Segoe UI", 10),
            padx=8,
            pady=6,
            cursor="hand2",
            state="disabled",
            command=self._clear_plot,
        )
        self.clear_btn.pack(fill="x", padx=16, pady=(0, 8))

        self.clear_lines_btn = tk.Button(
            sidebar,
            text="Clear Drawn Lines",
            font=("Segoe UI", 10),
            padx=8,
            pady=6,
            cursor="hand2",
            state="disabled",
            command=self._clear_drawn_lines,
        )
        self.clear_lines_btn.pack(fill="x", padx=16, pady=(0, 8))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        tk.Label(sidebar, text="Draw Lines", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(8, 2)
        )
        tk.Label(
            sidebar,
            text="Click & drag on the plot to\ndraw a dashed annotation line.",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 6))
        self.draw_btn = tk.Button(
            sidebar,
            text="✏  Draw Mode: OFF",
            font=("Segoe UI", 10),
            padx=8,
            pady=6,
            cursor="hand2",
            state="disabled",
            command=self._toggle_draw_mode,
        )
        self.draw_btn.pack(fill="x", padx=16, pady=(0, 16))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Legend labels
        tk.Label(sidebar, text="Legend Labels", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(8, 2)
        )
        tk.Label(sidebar, text="Data series label", font=("Segoe UI", 9)).pack(
            anchor="w", padx=16
        )
        self.legend_data_var = tk.StringVar(value="")
        tk.Entry(
            sidebar, textvariable=self.legend_data_var, font=("Segoe UI", 10), width=28
        ).pack(padx=16, pady=(2, 6))
        tk.Label(sidebar, text="Drawn lines label", font=("Segoe UI", 9)).pack(
            anchor="w", padx=16
        )
        self.legend_line_var = tk.StringVar(value="")
        tk.Entry(
            sidebar, textvariable=self.legend_line_var, font=("Segoe UI", 10), width=28
        ).pack(padx=16, pady=(2, 10))
        self.legend_apply_btn = tk.Button(
            sidebar,
            text="Apply Legend",
            font=("Segoe UI", 10),
            padx=8,
            pady=5,
            cursor="hand2",
            state="disabled",
            command=self._apply_legend,
        )
        self.legend_apply_btn.pack(fill="x", padx=16, pady=(0, 16))

        # ── Plot area ─────────────────────────────────────────────────────────
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side="right", fill="both", expand=True)

        self.placeholder = tk.Label(
            self.plot_frame,
            text="Open a file and select columns to plot",
            font=("Segoe UI", 13),
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

    # ── File handling ─────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("TSV files", "*.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls"):
                self.df = pd.read_excel(path)
            elif ext in (".tsv", ".txt"):
                self.df = pd.read_csv(path, sep="\t")
            else:
                self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.file_label.configure(text=os.path.basename(path))
        self._populate_columns()
        self.plot_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")

    def _populate_columns(self):
        cols = list(self.df.columns)
        self.x_menu["values"] = cols
        self.x_var.set(cols[0])

        for w in self.y_inner.winfo_children():
            w.destroy()
        self.y_checks.clear()

        for col in cols:
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(
                self.y_inner, text=col, variable=var, font=("Segoe UI", 10), anchor="w"
            )
            cb.pack(fill="x", padx=4, pady=1)
            self.y_checks[col] = var

        if len(cols) > 1:
            self.y_checks[cols[1]].set(True)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot(self):
        if self.df is None:
            return

        x_col = self.x_var.get()
        y_cols = [col for col, var in self.y_checks.items() if var.get()]

        if not y_cols:
            messagebox.showwarning("No Y column", "Select at least one Y-axis column.")
            return

        self._clear_plot(keep_data=True)
        self.placeholder.place_forget()

        self._drawn_lines = []  # reset drawn lines on new plot
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        self._fig = fig
        self._ax = ax

        chart = self.chart_var.get()

        # If x axis is 'frame' and numeric, reindex to fill every integer frame gap
        is_frame = x_col.strip().lower() == "frame" and pd.api.types.is_numeric_dtype(
            self.df[x_col]
        )
        if is_frame:
            frame_min = int(self.df[x_col].min())
            frame_max = int(self.df[x_col].max())
            full_index = pd.RangeIndex(frame_min, frame_max + 1)
            plot_df = self.df.groupby(x_col).mean().reindex(full_index)
        else:
            plot_df = self.df.set_index(x_col)

        for y_col in y_cols:
            x_data = plot_df.index
            y_data = plot_df[y_col]
            if chart == "Line":
                ax.plot(x_data, y_data, label=y_col)
            elif chart == "Scatter":
                ax.scatter(x_data, y_data, label=y_col, s=20, alpha=0.8)
            elif chart == "Bar":
                ax.bar(x_data, y_data, label=y_col, alpha=0.8)
            elif chart == "Area":
                ax.fill_between(x_data, y_data, label=y_col, alpha=0.5)
                ax.plot(x_data, y_data, linewidth=1.5)
            elif chart == "Step":
                ax.step(x_data, y_data, label=y_col, linewidth=2, where="mid")

        # Titles / labels (fall back to auto if left blank)
        xlabel = self.xlabel_var.get().strip() or x_col
        ylabel = self.ylabel_var.get().strip() or (
            ", ".join(y_cols) if len(y_cols) <= 3 else "Values"
        )
        title = (
            self.title_var.get().strip() or f"{chart} — {x_col} vs {', '.join(y_cols)}"
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=14)
        if len(y_cols) > 1:
            ax.legend(fontsize=10)

        # X axis range
        try:
            xmin = float(self.xmin_var.get().strip())
            xmax = float(self.xmax_var.get().strip())
            ax.set_xlim(xmin, xmax)
        except ValueError:
            pass  # leave matplotlib to auto-range if fields are blank/invalid

        fig.tight_layout(pad=2)

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))

        toolbar_frame = tk.Frame(self.plot_frame)
        toolbar_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # wire up draw-line mouse events
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.draw_btn.configure(state="normal")
        self.clear_lines_btn.configure(state="normal")
        self.legend_apply_btn.configure(state="normal")

    # -- Draw-line mode -------------------------------------------------------

    def _toggle_draw_mode(self):
        self._draw_mode = not self._draw_mode
        if self._draw_mode:
            self.draw_btn.configure(text="Draw Mode: ON")
        else:
            self.draw_btn.configure(text="Draw Mode: OFF")
            self._draw_start = None
            if self._preview_line:
                try:
                    self._preview_line.remove()
                except Exception:
                    pass
                self._preview_line = None
                self.canvas.draw_idle()

    def _on_press(self, event):
        if not self._draw_mode or event.inaxes is None:
            return
        self._draw_start = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self._draw_mode or self._draw_start is None or event.inaxes is None:
            return
        x0, y0 = self._draw_start
        x1, y1 = event.xdata, event.ydata
        if self._preview_line:
            try:
                self._preview_line.remove()
            except Exception:
                pass
        (self._preview_line,) = self._ax.plot(
            [x0, x1],
            [y0, y1],
            linestyle=(0, (6, 4)),
            color=self._annotation_color,
            linewidth=1.5,
            alpha=0.6,
        )
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._draw_mode or self._draw_start is None:
            return
        if event.inaxes is None or event.xdata is None:
            self._draw_start = None
            if self._preview_line:
                try:
                    self._preview_line.remove()
                except Exception:
                    pass
                self._preview_line = None
                self.canvas.draw_idle()
            return
        x0, y0 = self._draw_start
        x1, y1 = event.xdata, event.ydata
        self._draw_start = None
        self._preview_line = None
        self._drawn_lines.append((x0, y0, x1, y1))
        self._redraw_annotation_lines()

    def _redraw_annotation_lines(self):
        for artist in list(self._ax.lines):
            if getattr(artist, "_is_annotation", False):
                artist.remove()
        for x0, y0, x1, y1 in self._drawn_lines:
            (line,) = self._ax.plot(
                [x0, x1],
                [y0, y1],
                linestyle=(0, (8, 4)),
                color=self._annotation_color,
                linewidth=1.8,
                alpha=0.9,
            )
            line._is_annotation = True
        self.canvas.draw_idle()

    def _clear_drawn_lines(self):
        self._drawn_lines.clear()
        if self._ax:
            for artist in list(self._ax.lines):
                if getattr(artist, "_is_annotation", False):
                    artist.remove()
            self.canvas.draw_idle()

    def _apply_legend(self):
        self._update_legend()
        if self.canvas:
            self.canvas.draw_idle()

    def _update_legend(self):
        if self._ax is None:
            return
        import matplotlib.lines as mlines

        handles, labels = [], []
        # data series handles (real artists)
        for artist in self._ax.get_lines():
            if (
                not getattr(artist, "_is_annotation", False)
                and artist.get_label()
                and not artist.get_label().startswith("_")
            ):
                handles.append(artist)
                labels.append(artist.get_label())
        for coll in self._ax.collections:
            if coll.get_label() and not coll.get_label().startswith("_"):
                handles.append(coll)
                labels.append(coll.get_label())
        # override data label if user typed one (only when a single series)
        custom_data_label = self.legend_data_var.get().strip()
        if custom_data_label and len(labels) == 1:
            labels[0] = custom_data_label
        elif custom_data_label and len(labels) > 1:
            # prefix all data labels with the custom text
            labels = [f"{custom_data_label} ({l})" for l in labels]
        # drawn lines proxy
        custom_line_label = self.legend_line_var.get().strip() or "Drawn lines"
        proxy = mlines.Line2D(
            [],
            [],
            color=self._annotation_color,
            linestyle=(0, (8, 4)),
            linewidth=1.8,
            label=custom_line_label,
        )
        handles.append(proxy)
        labels.append(custom_line_label)
        if handles:
            self._ax.legend(handles, labels, fontsize=10)

    def _clear_plot(self, keep_data=False):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        plt.close("all")
        if not keep_data:
            self.placeholder.place(relx=0.5, rely=0.5, anchor="center")


if __name__ == "__main__":
    app = PlotterApp()
    app.mainloop()
