"""
Interactive Lambda Vertex-Coloring Algorithm with NetworkX, Tkinter, and Matplotlib.

Features:
  - Step-by-step UNDO engine (Ctrl+Z or button)
  - Game completion summary dialog & file export option
  - Custom adjacency list input & vertex initialization
  - Drag-and-drop canvas node positioning
  - Direct click-to-select and right-click context menu coloring
  - Dynamic layout switching (Spring, Circular, Kamada-Kawai, Shell, Spectral, Planar)
  - Full implementation of Lambda-1, Lambda-2, Majority Coloring, and Neighborhood-Share Filters

Dependencies:
    pip install networkx matplotlib

Run with:
    python lambda_coloring_app.py
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from itertools import combinations

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

COLOR_MAP = {'r': '#E63946', 'g': '#2A9D8F', 'b': '#457B9D'}
COLOR_NAMES = {'r': 'Red (r)', 'g': 'Green (g)', 'b': 'Blue (b)'}
COLOR_LETTERS = list(COLOR_MAP.keys())
UNCOLORED_FILL = '#D3D3D3'
INF = float('inf')


class LambdaColoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lambda Vertex-Coloring - Interactive Canvas")
        self.root.geometry("1380x870")

        # Game & Graph State
        self.G = None
        self.POS = {}
        self.coloring = {}
        self.round_no = 0
        self.lambdas = {}
        self.FULL_DEG = {}
        self.FULL_NEIGHBORS = {}
        self.current_menu = {}
        self.move_history = []  # Stack of dicts for Undo functionality

        # Drag & Click State
        self.selected_node = None
        self.dragging_node = None

        self.setup_ui()
        self.build_default_graph()

        # Keyboard shortcuts
        self.root.bind("<Control-z>", lambda event: self.undo_move())

    # -----------------------------------------------------------------
    # UI Setup
    # -----------------------------------------------------------------
    def setup_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left Sidebar Panel
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew")

        # 1. Graph Controls Box
        graph_box = ttk.LabelFrame(left_panel, text="1. Graph & Layout Controls", padding="10")
        graph_box.pack(fill=tk.X, pady=5)

        ttk.Label(graph_box, text="Vertices (n):").grid(row=0, column=0, sticky=tk.W)
        self.entry_n = ttk.Entry(graph_box, width=6)
        self.entry_n.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.entry_n.insert(0, "5")

        ttk.Label(graph_box, text="Layout:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.cbo_layout = ttk.Combobox(
            graph_box, values=["Spring", "Circular", "Kamada-Kawai", "Shell", "Spectral", "Planar"],
            width=12, state="readonly"
        )
        self.cbo_layout.grid(row=0, column=3, padx=5, sticky=tk.W)
        self.cbo_layout.set("Spring")
        self.cbo_layout.bind("<<ComboboxSelected>>", self.recalculate_layout)

        ttk.Label(
            graph_box,
            text="Adjacency List (one line per vertex 0..n-1):"
        ).grid(row=1, column=0, columnspan=4, pady=(5, 2), sticky=tk.W)

        self.txt_adj = tk.Text(graph_box, width=34, height=4)
        self.txt_adj.grid(row=2, column=0, columnspan=4, pady=5)
        self.txt_adj.insert(tk.END, "1 2\n0 2 3\n0 1 4\n1\n2")

        btn_build = ttk.Button(graph_box, text="Build & Initialize Graph", command=self.build_graph)
        btn_build.grid(row=3, column=0, columnspan=4, pady=5)

        # 2. Game Action Control Box
        self.game_box = ttk.LabelFrame(left_panel, text="2. Move & Color Control", padding="10")
        self.game_box.pack(fill=tk.X, pady=5)

        self.lbl_round_info = ttk.Label(self.game_box, text="Round: -", font=('Helvetica', 10, 'bold'))
        self.lbl_round_info.pack(anchor=tk.W, pady=2)

        self.lbl_instruction = ttk.Label(self.game_box, text="Click a node on canvas or select from menu.", wraplength=280)
        self.lbl_instruction.pack(anchor=tk.W, pady=5)

        move_frame = ttk.Frame(self.game_box)
        move_frame.pack(fill=tk.X, pady=5)

        ttk.Label(move_frame, text="Selected Node:").grid(row=0, column=0, sticky=tk.W)
        self.lbl_selected = ttk.Label(move_frame, text="None", font=('Helvetica', 10, 'bold'), foreground="#E63946")
        self.lbl_selected.grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(move_frame, text="Color:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.cbo_color = ttk.Combobox(move_frame, width=12, state="readonly")
        self.cbo_color.grid(row=1, column=1, padx=5, sticky=tk.W)

        btn_frame = ttk.Frame(self.game_box)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_submit = ttk.Button(btn_frame, text="Apply Color Move", command=self.apply_move, state=tk.DISABLED)
        self.btn_submit.grid(row=0, column=0, padx=2)

        self.btn_undo = ttk.Button(btn_frame, text="Undo Last Move (Ctrl+Z)", command=self.undo_move, state=tk.DISABLED)
        self.btn_undo.grid(row=0, column=1, padx=2)

        # 3. Live Parameter Table
        table_box = ttk.LabelFrame(left_panel, text="3. Live Lambda Parameters", padding="10")
        table_box.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("v", "l1", "l2", "lam", "c", "E_v")
        self.tree = ttk.Treeview(table_box, columns=columns, show="headings", height=8)
        self.tree.heading("v", text="v")
        self.tree.heading("l1", text="λ1")
        self.tree.heading("l2", text="λ2")
        self.tree.heading("lam", text="λ")
        self.tree.heading("c", text="c")
        self.tree.heading("E_v", text="E(v)")

        for col, w in zip(columns, [30, 45, 45, 45, 30, 70]):
            self.tree.column(col, width=w, anchor=tk.CENTER)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Right Display Panel (Interactive Canvas)
        right_panel = ttk.Frame(self.root, padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind Mouse Interactions
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_drag)

    # -----------------------------------------------------------------
    # Interactive Canvas Handlers
    # -----------------------------------------------------------------
    def on_canvas_click(self, event):
        if event.inaxes != self.ax or self.G is None:
            return

        click_x, click_y = event.xdata, event.ydata
        closest_node = None
        min_dist = float("inf")

        for node, (x, y) in self.POS.items():
            dist = math.hypot(click_x - x, click_y - y)
            if dist < min_dist and dist < 0.15:  # Hit threshold
                min_dist = dist
                closest_node = node

        if closest_node is not None:
            self.dragging_node = closest_node
            self.select_node(closest_node)

            if event.button == 3:  # Right click popup menu
                self.show_context_menu(closest_node)

    def on_canvas_drag(self, event):
        if self.dragging_node is None or event.inaxes != self.ax:
            return
        self.POS[self.dragging_node] = (event.xdata, event.ydata)
        self.draw_graph()

    def on_canvas_release(self, event):
        self.dragging_node = None

    def show_context_menu(self, node):
        allowed = self.current_menu.get(node, [])
        if not allowed:
            return

        menu = tk.Menu(self.root, tearoff=0)
        for c in allowed:
            menu.add_command(
                label=f"Color {COLOR_NAMES[c]}",
                command=lambda color=c: self.apply_direct_color(node, color)
            )
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def select_node(self, node):
        self.selected_node = node
        self.lbl_selected.config(text=str(node))

        allowed_colors = self.current_menu.get(node, [])
        self.cbo_color['values'] = [COLOR_NAMES[c] for c in allowed_colors]
        if allowed_colors:
            self.cbo_color.set(COLOR_NAMES[allowed_colors[0]])
        else:
            self.cbo_color.set('')

        self.draw_graph()

    # -----------------------------------------------------------------
    # Graph Construction & Layout Systems
    # -----------------------------------------------------------------
    def build_default_graph(self):
        self.build_graph()

    def build_graph(self):
        try:
            n = int(self.entry_n.get().strip())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive integer for n.")
            return

        lines = self.txt_adj.get("1.0", tk.END).strip().splitlines()

        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        for i in range(min(n, len(lines))):
            raw = lines[i].strip()
            if raw:
                try:
                    nbrs = [int(x) for x in raw.split()]
                    for j in nbrs:
                        if 0 <= j < n and j != i:
                            self.G.add_edge(i, j)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid input in adjacency line {i+1}.")
                    return

        self.FULL_DEG = {v: self.G.degree(v) for v in self.G.nodes()}
        self.FULL_NEIGHBORS = {v: list(self.G.neighbors(v)) for v in self.G.nodes()}

        self.recalculate_layout()
        self.coloring = {}
        self.move_history = []
        self.round_no = 0
        self.btn_undo.config(state=tk.DISABLED)
        self.start_next_round()

    def recalculate_layout(self, event=None):
        if self.G is None or self.G.number_of_nodes() == 0:
            return

        layout_type = self.cbo_layout.get()
        if layout_type == "Spring":
            self.POS = nx.spring_layout(self.G, seed=42)
        elif layout_type == "Circular":
            self.POS = nx.circular_layout(self.G)
        elif layout_type == "Kamada-Kawai":
            self.POS = nx.kamada_kawai_layout(self.G)
        elif layout_type == "Shell":
            self.POS = nx.shell_layout(self.G)
        elif layout_type == "Spectral":
            self.POS = nx.spectral_layout(self.G)
        elif layout_type == "Planar":
            try:
                self.POS = nx.planar_layout(self.G)
            except nx.NetworkXException:
                self.POS = nx.spring_layout(self.G, seed=42)

        self.draw_graph()

    # -----------------------------------------------------------------
    # Algorithmic Lambda Logic
    # -----------------------------------------------------------------
    def uncolored_degree(self, v):
        return sum(1 for u in self.FULL_NEIGHBORS[v] if u not in self.coloring)

    def uncolored_neighbors(self, v):
        return [u for u in self.FULL_NEIGHBORS[v] if u not in self.coloring]

    def is_majority_colored(self, u, coloring_state=None):
        col_map = self.coloring if coloring_state is None else coloring_state
        if u not in col_map:
            return False
        col = col_map[u]
        deg = self.FULL_DEG[u]
        same_count = sum(1 for w in self.FULL_NEIGHBORS[u] if col_map.get(w) == col)
        return same_count == (deg // 2)

    def vertex_eligible_colors(self, v):
        excluded = set()
        for u in self.FULL_NEIGHBORS[v]:
            if self.is_majority_colored(u):
                excluded.add(self.coloring[u])
        return [c for c in COLOR_LETTERS if c not in excluded]

    def compute_lambdas(self):
        result = {}
        for v in self.G.nodes():
            if v in self.coloring:
                continue

            Nc = self.uncolored_neighbors(v)
            E_v = self.vertex_eligible_colors(v)
            c = len(E_v)

            lambda_1 = INF
            if c == 0:
                lambda_1 = 0
            elif c <= len(Nc):
                for combo in combinations(Nc, c):
                    s = sum(self.uncolored_degree(u) // 2 + 1 for u in combo)
                    if s < lambda_1:
                        lambda_1 = s

            lambda_2 = INF
            k = c - 1
            if k >= 0:
                dv = self.uncolored_degree(v)
                base = dv // 2 + 1
                if k == 0:
                    lambda_2 = base
                elif k <= len(Nc):
                    for combo in combinations(Nc, k):
                        s = sum(self.uncolored_degree(u) // 2 + 1 for u in combo) + base
                        if s < lambda_2:
                            lambda_2 = s

            lam = min(lambda_1, lambda_2)
            result[v] = (lambda_1, lambda_2, lam, c, E_v)

        return result

    def candidate_colors_for_vertex(self, v):
        nearby = [
            u for u in self.coloring
            if nx.has_path(self.G, v, u) and nx.shortest_path_length(self.G, v, u) <= 2
        ]
        counts = {c: 0 for c in COLOR_LETTERS}
        for u in nearby:
            counts[self.coloring[u]] += 1

        total_neighbors = len(self.FULL_NEIGHBORS[v])
        nbr_counts = {c: 0 for c in COLOR_LETTERS}
        for u in self.FULL_NEIGHBORS[v]:
            if u in self.coloring:
                nbr_counts[self.coloring[u]] += 1

        share_eligible = [c for c in COLOR_LETTERS if nbr_counts[c] <= (total_neighbors // 2)]
        v_eligible = set(self.vertex_eligible_colors(v))
        eligible = [c for c in COLOR_LETTERS if c in v_eligible and c in share_eligible]

        if not eligible:
            return []

        ranked = sorted(eligible, key=lambda c: counts[c])

        survivors = []
        for c in ranked:
            trial = dict(self.coloring)
            trial[v] = c
            if not self.is_majority_colored(v, trial):
                survivors.append(c)

        return survivors if survivors else eligible

    # -----------------------------------------------------------------
    # Game Round & Undo Management
    # -----------------------------------------------------------------
    def start_next_round(self):
        if len(self.coloring) == self.G.number_of_nodes():
            self.lbl_instruction.config(text="Game Complete! All vertices colored.")
            self.btn_submit.config(state=tk.DISABLED)
            self.draw_graph()
            self.show_game_over_dialog("COMPLETE - All vertices colored!")
            return

        self.round_no += 1
        self.lambdas = self.compute_lambdas()
        self.update_table()

        uncolored = [v for v in self.G.nodes() if v not in self.coloring]
        is_restricted_round = (self.round_no % 2 == 1)

        self.lbl_round_info.config(
            text=f"Round {self.round_no} ({'ODD - Restricted' if is_restricted_round else 'EVEN - Free Choice'})"
        )

        if is_restricted_round:
            min_val = min(self.lambdas[v][2] for v in uncolored)
            v_options = [v for v in uncolored if self.lambdas[v][2] == min_val]

            self.current_menu = {}
            for v in v_options:
                cand = self.candidate_colors_for_vertex(v)
                if cand:
                    self.current_menu[v] = cand

            if not self.current_menu:
                self.lbl_instruction.config(text="Game Over: No valid moves remain.")
                self.btn_submit.config(state=tk.DISABLED)
                self.show_game_over_dialog("WINNER - No valid moves left for Restricted Round!")
                return

            self.lbl_instruction.config(
                text=f"Min λ = {min_val}. Select node in {list(self.current_menu.keys())} to color."
            )

        else:
            self.current_menu = {}
            for v in uncolored:
                offered = self.vertex_eligible_colors(v)
                if offered:
                    self.current_menu[v] = offered

            if not self.current_menu:
                self.lbl_instruction.config(text="Game Over: No valid moves remain.")
                self.btn_submit.config(state=tk.DISABLED)
                self.show_game_over_dialog("WINNER - No eligible colors left!")
                return

            self.lbl_instruction.config(text="Even Round: Click any uncolored node to color it.")

        self.selected_node = None
        self.lbl_selected.config(text="None")
        self.cbo_color.set('')
        self.cbo_color['values'] = []
        self.btn_submit.config(state=tk.NORMAL)
        self.btn_undo.config(state=tk.NORMAL if self.move_history else tk.DISABLED)

        self.draw_graph()

    def apply_direct_color(self, node, color_code):
        # Save state to undo history
        round_type = "Restricted" if self.round_no % 2 == 1 else "Free Choice"
        self.move_history.append({
            'round': self.round_no,
            'round_type': round_type,
            'node': node,
            'color': color_code,
            'color_name': COLOR_NAMES[color_code]
        })

        self.coloring[node] = color_code
        self.start_next_round()

    def apply_move(self):
        if self.selected_node is None:
            messagebox.showwarning("Warning", "Please click a node on the graph canvas to select it.")
            return

        selected_str = self.cbo_color.get()
        color_code = next((k for k, v in COLOR_NAMES.items() if v == selected_str), None)

        if not color_code or color_code not in self.current_menu.get(self.selected_node, []):
            messagebox.showwarning("Warning", "Invalid color option for selected vertex.")
            return

        self.apply_direct_color(self.selected_node, color_code)

    def undo_move(self):
        if not self.move_history:
            return

        last_move = self.move_history.pop()
        node = last_move['node']

        if node in self.coloring:
            del self.coloring[node]

        # Step back round calculation
        self.round_no = max(0, self.round_no - 2)
        self.start_next_round()

    # -----------------------------------------------------------------
    # Move Export & Completion Dialog
    # -----------------------------------------------------------------
    def show_game_over_dialog(self, reason_str):
        dialog = tk.Toplevel(self.root)
        dialog.title("Game Over - Summary & Move Sequence")
        dialog.geometry("550x450")

        ttk.Label(dialog, text="Game Over!", font=('Helvetica', 14, 'bold')).pack(pady=5)
        ttk.Label(dialog, text=reason_str, font=('Helvetica', 10, 'italic')).pack(pady=2)

        ttk.Label(dialog, text="Sequence of Played Moves:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, padx=15, pady=(10, 2))

        txt_frame = ttk.Frame(dialog)
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        txt_summary = tk.Text(txt_frame, height=12, width=60)
        scroll = ttk.Scrollbar(txt_frame, command=txt_summary.yview)
        txt_summary.configure(yscrollcommand=scroll.set)

        txt_summary.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        move_text = self.format_move_sequence()
        txt_summary.insert(tk.END, move_text)
        txt_summary.config(state=tk.DISABLED)

        btn_box = ttk.Frame(dialog)
        btn_box.pack(fill=tk.X, pady=10)

        btn_save = ttk.Button(btn_box, text="Save Move History to File...", command=lambda: self.save_moves_to_file(move_text))
        btn_save.pack(side=tk.LEFT, padx=15)

        btn_close = ttk.Button(btn_box, text="Close Window", command=dialog.destroy)
        btn_close.pack(side=tk.RIGHT, padx=15)

    def format_move_sequence(self):
        if not self.move_history:
            return "No moves played."

        lines = ["=== LAMBDA VERTEX COLORING MOVE LOG ===", ""]
        for idx, m in enumerate(self.move_history, 1):
            lines.append(
                f"Move {idx:02d} | Round {m['round']} ({m['round_type']}) -> "
                f"Colored Vertex [{m['node']}] with {m['color_name']}"
            )
        lines.append("\n=== END OF RECORD ===")
        return "\n".join(lines)

    def save_moves_to_file(self, content):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Game Move Sequence"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Move sequence saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file: {str(e)}")

    # -----------------------------------------------------------------
    # Rendering & Updates
    # -----------------------------------------------------------------
    def update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for v in sorted(self.lambdas):
            l1, l2, lam, c, E_v = self.lambdas[v]
            fmt = lambda x: "∞" if x == INF else str(x)
            self.tree.insert("", tk.END, values=(v, fmt(l1), fmt(l2), fmt(lam), c, ",".join(E_v)))

    def draw_graph(self):
        self.ax.clear()

        node_colors = []
        labels = {}

        for v in self.G.nodes():
            if v in self.coloring:
                node_colors.append(COLOR_MAP[self.coloring[v]])
                labels[v] = f"{v}:{self.coloring[v]}"
            else:
                node_colors.append(UNCOLORED_FILL)
                lam = self.lambdas.get(v, (None, None, None, None, None))[2]
                lam_str = "∞" if lam == INF else str(lam)
                labels[v] = f"{v}:{lam_str}"

        node_border_colors = []
        linewidths = []
        for v in self.G.nodes():
            if v == self.selected_node:
                node_border_colors.append('#D90429')  # Selection Highlight Red
                linewidths.append(3.5)
            elif v in self.current_menu:
                node_border_colors.append('#2B2D42')  # Active Move Candidate
                linewidths.append(2.0)
            else:
                node_border_colors.append('#8D99AE')  # Subdued
                linewidths.append(1.0)

        nx.draw_networkx_nodes(
            self.G, self.POS, ax=self.ax, node_color=node_colors, node_size=900,
            edgecolors=node_border_colors, linewidths=linewidths
        )
        nx.draw_networkx_edges(self.G, self.POS, ax=self.ax, width=1.5, edge_color='#8D99AE')
        nx.draw_networkx_labels(self.G, self.POS, labels=labels, ax=self.ax, font_size=10, font_weight='bold')

        self.ax.set_title(f"Interactive Graph Canvas (Round {self.round_no}) - Drag nodes to reposition", fontsize=11)
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = LambdaColoringGUI(root)
    root.mainloop()