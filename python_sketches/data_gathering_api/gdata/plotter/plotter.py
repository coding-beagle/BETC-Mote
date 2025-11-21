import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from pathlib import Path


class CSVPlotter:
    def __init__(self):
        self.datasets = {}
        self.current_dataset = None
        plt.ion()  # Enable interactive mode

    def search_csvs(self, pattern="*.csv", directory="."):
        """Search for CSV files matching pattern in directory"""
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path, recursive=True)
        return files

    def load_csv(self, filepath, name=None):
        """Load a CSV file into the dataset"""
        try:
            df = pd.read_csv(filepath)
            if name is None:
                name = Path(filepath).stem
            self.datasets[name] = {"data": df, "path": filepath}
            print(
                f"✓ Loaded '{name}' with {len(df)} rows and columns: {list(df.columns)}"
            )
            return True
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return False

    def list_datasets(self):
        """List all loaded datasets"""
        if not self.datasets:
            print("No datasets loaded.")
            return
        print("\nLoaded datasets:")
        for i, (name, info) in enumerate(self.datasets.items(), 1):
            df = info["data"]
            print(f"  {i}. {name} ({len(df)} rows, {len(df.columns)} columns)")

    def select_dataset(self, name):
        """Select a dataset to work with"""
        if name in self.datasets:
            self.current_dataset = name
            print(f"Selected dataset: {name}")
            return True
        print(f"Dataset '{name}' not found.")
        return False

    def plot_columns(self, x_col, y_cols, dataset_name=None):
        """Plot selected columns"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return

        df = self.datasets[dataset_name]["data"]

        # Validate columns
        missing = [col for col in [x_col] + y_cols if col not in df.columns]
        if missing:
            print(f"Columns not found: {missing}")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for y_col in y_cols:
            ax.plot(df[x_col], df[y_col], label=y_col, marker="o", markersize=3)

        ax.set_xlabel(x_col)
        ax.set_ylabel("Value")
        ax.set_title(f'{dataset_name}: {", ".join(y_cols)} vs {x_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        print(f"✓ Plot created with {len(y_cols)} series")

    def show_columns(self, dataset_name=None):
        """Show available columns in dataset"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return

        df = self.datasets[dataset_name]["data"]
        print(f"\nColumns in '{dataset_name}':")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

    def run(self):
        """Main interactive loop"""
        print("=" * 60)
        print("Interactive CSV Plotter")
        print("=" * 60)
        print("Commands:")
        print("  search [pattern] [dir] - Search for CSV files")
        print("  load <file> [name]     - Load a CSV file")
        print("  list                   - List loaded datasets")
        print("  select <name>          - Select a dataset")
        print("  columns [name]         - Show columns in dataset")
        print("  plot <x> <y1> [y2...]  - Plot columns (x-axis, y-axes)")
        print("  clear                  - Close all plots")
        print("  quit                   - Exit")
        print("=" * 60)

        while True:
            try:
                cmd = input("\n> ").strip()
                if not cmd:
                    continue

                parts = cmd.split()
                action = parts[0].lower()

                if action == "quit" or action == "exit":
                    plt.close("all")
                    print("Goodbye!")
                    break

                elif action == "search":
                    pattern = parts[1] if len(parts) > 1 else "*.csv"
                    directory = parts[2] if len(parts) > 2 else "."
                    files = self.search_csvs(pattern, directory)
                    if files:
                        print(f"\nFound {len(files)} CSV file(s):")
                        for i, f in enumerate(files, 1):
                            print(f"  {i}. {f}")
                    else:
                        print("No CSV files found.")

                elif action == "load":
                    if len(parts) < 2:
                        print("Usage: load <file> [name]")
                        continue
                    filepath = parts[1]
                    name = parts[2] if len(parts) > 2 else None
                    self.load_csv(filepath, name)

                elif action == "list":
                    self.list_datasets()

                elif action == "select":
                    if len(parts) < 2:
                        print("Usage: select <name>")
                        continue
                    self.select_dataset(parts[1])

                elif action == "columns":
                    name = parts[1] if len(parts) > 1 else None
                    self.show_columns(name)

                elif action == "plot":
                    if len(parts) < 3:
                        print("Usage: plot <x_column> <y_column1> [y_column2 ...]")
                        continue
                    if not self.current_dataset:
                        print("No dataset selected. Use 'select <name>' first.")
                        continue
                    x_col = parts[1]
                    y_cols = parts[2:]
                    self.plot_columns(x_col, y_cols)

                elif action == "clear":
                    plt.close("all")
                    print("All plots closed.")

                else:
                    print(f"Unknown command: {action}")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")
