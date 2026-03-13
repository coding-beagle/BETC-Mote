import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os

BG = "#1e1e2e"
BG2 = "#2a2a3e"
ACCENT = "#4A9EFF"
TEXT = "#e0e0e0"
SUBTLE = "#888888"


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
        self._build_ui()

    def _build_ui(self):
        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = tk.Frame(self, bg=BG2, width=270)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="📊 CSV Plotter",
            bg=BG2,
            fg=TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=16, pady=(20, 2))
        tk.Label(
            sidebar,
            text="Load, select, visualise",
            bg=BG2,
            fg=SUBTLE,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(0, 12))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        tk.Button(
            sidebar,
            text="Open File",
            bg=ACCENT,
            fg="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            padx=8,
            pady=6,
            cursor="hand2",
            command=self._open_file,
        ).pack(fill="x", padx=16, pady=(8, 4))

        self.file_label = tk.Label(
            sidebar,
            text="No file loaded",
            bg=BG2,
            fg=SUBTLE,
            font=("Segoe UI", 9),
            wraplength=230,
        )
        self.file_label.pack(padx=16, pady=(0, 10))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # X axis
        tk.Label(
            sidebar, text="X Axis", bg=BG2, fg=TEXT, font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=16, pady=(10, 2))
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
            sidebar,
            text="Y Axis (one or more)",
            bg=BG2,
            fg=TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=16, pady=(0, 4))

        y_outer = tk.Frame(sidebar, bg=BG2)
        y_outer.pack(fill="x", padx=16, pady=(0, 10))

        self.y_canvas = tk.Canvas(y_outer, bg=BG, highlightthickness=0, height=200)
        y_scroll = ttk.Scrollbar(
            y_outer, orient="vertical", command=self.y_canvas.yview
        )
        self.y_canvas.configure(yscrollcommand=y_scroll.set)
        self.y_inner = tk.Frame(self.y_canvas, bg=BG)
        self.y_canvas.create_window((0, 0), window=self.y_inner, anchor="nw")
        self.y_inner.bind(
            "<Configure>",
            lambda e: self.y_canvas.configure(scrollregion=self.y_canvas.bbox("all")),
        )
        self.y_canvas.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Chart type
        tk.Label(
            sidebar, text="Chart Type", bg=BG2, fg=TEXT, font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=16, pady=(8, 2))
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

        # Buttons
        self.plot_btn = tk.Button(
            sidebar,
            text="Plot",
            bg="#22c55e",
            fg="white",
            relief="flat",
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
            bg="#3a3a5a",
            fg=TEXT,
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=6,
            cursor="hand2",
            state="disabled",
            command=self._clear_plot,
        )
        self.clear_btn.pack(fill="x", padx=16, pady=(0, 16))

        # ── Plot area ─────────────────────────────────────────────────────────
        self.plot_frame = tk.Frame(self, bg=BG)
        self.plot_frame.pack(side="right", fill="both", expand=True)

        self.placeholder = tk.Label(
            self.plot_frame,
            text="Open a file and select columns to plot",
            bg=BG,
            fg=SUBTLE,
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
                self.y_inner,
                text=col,
                variable=var,
                bg=BG,
                fg=TEXT,
                selectcolor=BG2,
                activebackground=BG,
                activeforeground=ACCENT,
                font=("Segoe UI", 10),
                anchor="w",
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

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor("#0d0d1a")

        chart = self.chart_var.get()
        colors = plt.cm.tab10.colors

        for i, y_col in enumerate(y_cols):
            color = colors[i % len(colors)]
            x_data, y_data = self.df[x_col], self.df[y_col]
            if chart == "Line":
                ax.plot(x_data, y_data, label=y_col, color=color, linewidth=2)
            elif chart == "Scatter":
                ax.scatter(x_data, y_data, label=y_col, color=color, s=20, alpha=0.8)
            elif chart == "Bar":
                ax.bar(x_data, y_data, label=y_col, color=color, alpha=0.8)
            elif chart == "Area":
                ax.fill_between(x_data, y_data, label=y_col, color=color, alpha=0.5)
                ax.plot(x_data, y_data, color=color, linewidth=1.5)
            elif chart == "Step":
                ax.step(
                    x_data, y_data, label=y_col, color=color, linewidth=2, where="mid"
                )

        ax.set_xlabel(x_col, color=TEXT, fontsize=11)
        ax.set_ylabel(
            ", ".join(y_cols) if len(y_cols) <= 3 else "Values", color=TEXT, fontsize=11
        )
        ax.set_title(
            f"{chart} — {x_col} vs {', '.join(y_cols)}", color=TEXT, fontsize=13, pad=14
        )
        ax.tick_params(colors=TEXT)
        ax.spines[:].set_color("#333355")
        ax.grid(color="#222244", linestyle="--", linewidth=0.6, alpha=0.6)
        if len(y_cols) > 1:
            ax.legend(
                facecolor="#0f0f2a", edgecolor="#333355", labelcolor=TEXT, fontsize=10
            )

        fig.tight_layout(pad=2)

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))

        toolbar_frame = tk.Frame(self.plot_frame, bg=BG)
        toolbar_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

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
