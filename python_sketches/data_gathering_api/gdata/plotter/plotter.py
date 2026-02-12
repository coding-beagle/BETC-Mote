import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings


class CSVCompleter(Completer):
    def __init__(self, plotter):
        self.plotter = plotter
        self.commands = {
            "search": "Search for CSV files",
            "load": "Load a CSV file",
            "list": "List loaded datasets",
            "select": "Select a dataset",
            "columns": "Show columns in dataset",
            "plot": "Plot columns",
            "filter": "Filter current dataset",
            "reset": "Reset filters on dataset",
            "info": "Show dataset info and filters",
            "clear": "Close all plots",
            "quit": "Exit",
            "exit": "Exit",
            "help": "Show help",
        }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split()

        # If we're at the start or just typed a space, suggest commands
        if not parts or (len(parts) == 1 and not text.endswith(" ")):
            word = parts[0] if parts else ""
            for cmd, desc in self.commands.items():
                if cmd.startswith(word.lower()):
                    yield Completion(cmd, start_position=-len(word), display_meta=desc)

        # Context-aware completions
        elif len(parts) >= 1:
            cmd = parts[0].lower()

            # Autocomplete dataset names for select, columns commands
            if (
                cmd in ["select", "columns", "filter", "reset", "info"]
                and len(parts) <= 2
            ):
                word = parts[1] if len(parts) == 2 else ""
                for name in self.plotter.datasets.keys():
                    if name.startswith(word):
                        yield Completion(
                            name, start_position=-len(word), display_meta="dataset"
                        )

            # Autocomplete column names for plot command
            elif cmd == "plot" and self.plotter.current_dataset:
                df = self.plotter.get_filtered_data()
                word = parts[-1] if not text.endswith(" ") else ""

                for col in df.columns:
                    col_str = str(col)
                    # Handle columns with spaces by quoting them
                    if " " in col_str:
                        col_str = f'"{col_str}"'

                    if col_str.startswith(word) or col_str.strip('"').startswith(word):
                        yield Completion(
                            col_str,
                            start_position=-len(word),
                            display_meta=f"column ({len(parts)-1})",
                        )

            # Autocomplete column names for filter command
            elif cmd == "filter" and self.plotter.current_dataset and len(parts) >= 2:
                df = self.plotter.datasets[self.plotter.current_dataset]["data"]
                word = parts[-1] if not text.endswith(" ") else ""

                # After dataset name, suggest column names
                if len(parts) >= 2:
                    for col in df.columns:
                        col_str = str(col)
                        if " " in col_str:
                            col_str = f'"{col_str}"'

                        if col_str.startswith(word) or col_str.strip('"').startswith(
                            word
                        ):
                            yield Completion(
                                col_str,
                                start_position=-len(word),
                                display_meta="column to filter",
                            )

            # Autocomplete file paths for load command
            elif cmd == "load" and len(parts) <= 2:
                word = parts[1] if len(parts) == 2 else ""
                # Get CSV files in current directory
                try:
                    csv_files = glob.glob("*.csv")
                    for f in csv_files:
                        if f.startswith(word):
                            yield Completion(
                                f, start_position=-len(word), display_meta="CSV file"
                            )
                except:
                    pass


class CSVPlotter:
    def __init__(self):
        self.datasets = {}
        self.current_dataset = None
        self.filters = {}  # Store filters for each dataset
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
            self.filters[name] = None  # Initialize no filters
            print(
                f"✓ Loaded '{name}' with {len(df)} rows and columns: {list(df.columns)}"
            )
            return True
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return False

    def get_filtered_data(self, dataset_name=None):
        """Get filtered data for a dataset"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            return None

        df = self.datasets[dataset_name]["data"]
        filter_info = self.filters.get(dataset_name)

        if filter_info is None:
            return df

        # Apply filters
        filtered_df = df.copy()

        if "row_range" in filter_info:
            start, end = filter_info["row_range"]
            filtered_df = filtered_df.iloc[start:end]

        if "column_filters" in filter_info:
            for col, condition in filter_info["column_filters"].items():
                op, value = condition
                if op == "==":
                    filtered_df = filtered_df[filtered_df[col] == value]
                elif op == "!=":
                    filtered_df = filtered_df[filtered_df[col] != value]
                elif op == ">":
                    filtered_df = filtered_df[filtered_df[col] > value]
                elif op == "<":
                    filtered_df = filtered_df[filtered_df[col] < value]
                elif op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= value]
                elif op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= value]

        return filtered_df

    def set_filter(
        self,
        dataset_name=None,
        row_start=None,
        row_end=None,
        column=None,
        operator=None,
        value=None,
    ):
        """Set filters for a dataset"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return False

        if self.filters[dataset_name] is None:
            self.filters[dataset_name] = {}

        # Set row range filter
        if row_start is not None or row_end is not None:
            df = self.datasets[dataset_name]["data"]
            start = row_start if row_start is not None else 0
            end = row_end if row_end is not None else len(df)
            self.filters[dataset_name]["row_range"] = (start, end)
            print(f"✓ Set row filter: rows {start} to {end}")

        # Set column filter
        if column is not None and operator is not None and value is not None:
            if "column_filters" not in self.filters[dataset_name]:
                self.filters[dataset_name]["column_filters"] = {}

            # Try to convert value to appropriate type
            df = self.datasets[dataset_name]["data"]
            if column in df.columns:
                try:
                    dtype = df[column].dtype
                    if dtype in ["int64", "float64"]:
                        value = float(value)
                except:
                    pass

            self.filters[dataset_name]["column_filters"][column] = (operator, value)
            print(f"✓ Set column filter: {column} {operator} {value}")

        # Show filtered dataset info
        filtered_df = self.get_filtered_data(dataset_name)
        original_df = self.datasets[dataset_name]["data"]
        print(f"   Filtered: {len(filtered_df)}/{len(original_df)} rows")
        return True

    def reset_filter(self, dataset_name=None):
        """Reset filters for a dataset"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return False

        self.filters[dataset_name] = None
        print(f"✓ Reset filters for '{dataset_name}'")
        return True

    def show_info(self, dataset_name=None):
        """Show detailed info about a dataset including filters"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return

        df = self.datasets[dataset_name]["data"]
        filtered_df = self.get_filtered_data(dataset_name)
        filter_info = self.filters.get(dataset_name)

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Path: {self.datasets[dataset_name]['path']}")
        print(f"Original rows: {len(df)}")
        print(f"Filtered rows: {len(filtered_df)}")
        print(f"Columns: {len(df.columns)}")

        if filter_info:
            print(f"\nActive Filters:")
            if "row_range" in filter_info:
                start, end = filter_info["row_range"]
                print(f"  - Row range: {start} to {end}")
            if "column_filters" in filter_info:
                for col, (op, val) in filter_info["column_filters"].items():
                    print(f"  - {col} {op} {val}")
        else:
            print(f"\nNo active filters")

        print(f"{'='*60}\n")

        """List all loaded datasets"""
        if not self.datasets:
            print("No datasets loaded.")
            return
        print("\nLoaded datasets:")
        for i, (name, info) in enumerate(self.datasets.items(), 1):
            df = info["data"]
            current = " [SELECTED]" if name == self.current_dataset else ""
            print(f"  {i}. {name} ({len(df)} rows, {len(df.columns)} columns){current}")

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

        df = self.get_filtered_data(dataset_name)

        # Handle quoted column names
        x_col = x_col.strip('"')
        y_cols = [col.strip('"') for col in y_cols]

        # Validate columns
        missing = [col for col in [x_col] + y_cols if col not in df.columns]
        if missing:
            print(f"Columns not found: {missing}")
            print(f"Available columns: {list(df.columns)}")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for y_col in y_cols:
            ax.plot(df[x_col], df[y_col], label=y_col, marker="o", markersize=3)

        ax.set_xlabel(x_col)
        ax.set_ylabel("Value")

        # Add filter info to title if filters are active
        title = f'{dataset_name}: {", ".join(y_cols)} vs {x_col}'
        if self.filters.get(dataset_name):
            title += " [FILTERED]"
        ax.set_title(title)

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        original_len = len(self.datasets[dataset_name]["data"])
        print(
            f"✓ Plot created with {len(y_cols)} series ({len(df)}/{original_len} rows)"
        )

    def show_columns(self, dataset_name=None):
        """Show available columns in dataset"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return

        df = self.get_filtered_data(dataset_name)
        print(f"\nColumns in '{dataset_name}' ({len(df)} rows):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

    def show_help(self):
        """Show help message"""
        print("\n" + "=" * 60)
        print("Commands:")
        print("  search [pattern] [dir] - Search for CSV files")
        print("  load <file> [name]     - Load a CSV file")
        print("  list                   - List loaded datasets")
        print("  select <name>          - Select a dataset")
        print("  columns [name]         - Show columns in dataset")
        print("  plot <x> <y1> [y2...]  - Plot columns (x-axis, y-axes)")
        print("  clear                  - Close all plots")
        print("  help                   - Show this help")
        print("  quit                   - Exit")
        print("=" * 60)
        print("\nTips:")
        print("  - Use TAB for autocomplete")
        print("  - Press F2 to show/hide completion menu")
        print("  - Column names with spaces should be quoted")
        print("  - Use up/down arrows for command history")
        print()

    def run(self):
        """Main interactive loop with autocomplete"""
        print("=" * 60)
        print("Interactive CSV Plotter (with autocomplete!)")
        print("=" * 60)
        print("Type 'help' for commands, TAB for autocomplete, F2 to toggle menu")
        print("=" * 60)

        # Create custom style
        style = Style.from_dict(
            {
                "prompt": "#00aa00 bold",
                "completion-menu.completion": "bg:#008888 #ffffff",
                "completion-menu.completion.current": "bg:#00aaaa #000000",
                "completion-menu.meta.completion": "bg:#006666 #ffffff",
                "completion-menu.meta.completion.current": "bg:#008888 #ffffff",
            }
        )

        # Create key bindings for toggling completion menu
        kb = KeyBindings()

        @kb.add("f2")
        def _(event):
            """Toggle completion menu visibility"""
            buff = event.app.current_buffer
            if buff.complete_state:
                buff.complete_state = None
            else:
                buff.start_completion()

        # Create session with completer
        completer = CSVCompleter(self)
        session = PromptSession(
            completer=completer,
            style=style,
            complete_while_typing=True,
            enable_history_search=True,
            key_bindings=kb,
            complete_in_thread=True,
            # Show completion menu with suggestions
            mouse_support=True,
        )

        while True:
            try:
                # Show current dataset in prompt
                prompt_text = (
                    f"[{self.current_dataset}] > " if self.current_dataset else "> "
                )
                cmd = session.prompt(HTML(f"<prompt>{prompt_text}</prompt>")).strip()

                if not cmd:
                    continue

                parts = cmd.split()
                action = parts[0].lower()

                if action in ["quit", "exit"]:
                    plt.close("all")
                    print("Goodbye!")
                    break

                elif action == "help":
                    self.show_help()

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

                elif action == "info":
                    name = parts[1] if len(parts) > 1 else None
                    self.show_info(name)

                elif action == "filter":
                    if len(parts) < 3:
                        print(
                            "Usage: filter <start> <end> OR filter <column> <operator> <value>"
                        )
                        print("Examples: filter 10 100  OR  filter age > 25")
                        continue
                    if not self.current_dataset:
                        print("No dataset selected. Use 'select <n>' first.")
                        continue

                    # Check if it's a row range filter (two numbers)
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                        self.set_filter(row_start=start, row_end=end)
                    except ValueError:
                        # It's a column filter
                        if len(parts) < 4:
                            print("Usage: filter <column> <operator> <value>")
                            print("Operators: ==, !=, >, <, >=, <=")
                            continue
                        col = parts[1].strip('"')
                        op = parts[2]
                        val = " ".join(parts[3:])  # Handle values with spaces
                        if op not in ["==", "!=", ">", "<", ">=", "<="]:
                            print(f"Invalid operator: {op}")
                            print("Valid operators: ==, !=, >, <, >=, <=")
                            continue
                        self.set_filter(column=col, operator=op, value=val)

                elif action == "reset":
                    name = parts[1] if len(parts) > 1 else None
                    self.reset_filter(name)

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
                    print(
                        f"Unknown command: {action}. Type 'help' for available commands."
                    )

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
                continue
            except EOFError:
                plt.close("all")
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    plotter = CSVPlotter()
    plotter.run()
