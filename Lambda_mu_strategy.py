import sys
import os
import math
import copy
import re
from itertools import combinations
import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog, QGroupBox,
    QSplitter, QShortcut, QMenu, QAction, QSpinBox, QTabWidget
)
from PyQt5.QtGui import QKeySequence, QCursor
from PyQt5.QtCore import Qt

INF = float('inf')
COLOR_MAP = {'r': '#e74c3c', 'g': '#2ecc71', 'b': '#3498db'} # Red, Green, Blue hex
COLOR_LETTERS = ['r', 'g', 'b']
UNCOLORED_FILL = '#dcdde1'


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Pure Python hierarchical layout for trees."""
    if not nx.is_tree(G):
        return nx.spring_layout(G)

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[0]

    def _hierarchy_pos(G, root, left, right, vert_loc, pos, parent=None):
        pos[root] = ((left + right) / 2, vert_loc)
        neighbors = [n for n in G.neighbors(root) if n != parent]
        if len(neighbors) != 0:
            dx = (right - left) / len(neighbors)
            nextx = left
            for neighbor in neighbors:
                _hierarchy_pos(G, neighbor, nextx, nextx + dx, vert_loc - vert_gap, pos, parent=root)
                nextx += dx
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_loc, {})


class LambdaColoringEngine:
    """Core game engine managing structural mu-vectors, live lambda values, and v9 rules."""
    def __init__(self, adj_text, num_vertices=None):
        self.G = nx.Graph()
        self.parse_adjacency_list(adj_text, num_vertices)
        
        self.coloring = {} # vertex -> 'r'|'g'|'b'
        self.move_history = [] # list of (round_no, vertex, color, is_restricted)
        self.state_stack = []  # Stack of (coloring_snapshot, mu_snapshot) for UNDO
        
        # Static structural properties
        self.full_deg = {v: self.G.degree(v) for v in self.G.nodes()}
        self.full_neighbors = {v: list(self.G.neighbors(v)) for v in self.G.nodes()}
        self.is_pendant = {v: (self.full_deg[v] == 1) for v in self.G.nodes()}
        
        self.mu = self.init_mu_vectors()

    def parse_adjacency_list(self, text, num_vertices=None):
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        edges = []
        max_parsed_v = -1
        declared_nodes = set()
        
        for line in lines:
            match = re.match(r'^\s*v?(\d+)\s*[:\-\=](.*)$', line)
            if match:
                u = int(match.group(1))
                declared_nodes.add(u)
                max_parsed_v = max(max_parsed_v, u)
                
                rest = match.group(2)
                neighbors = re.findall(r'\d+', rest)
                for tok in neighbors:
                    v = int(tok)
                    if v != u:
                        edges.append((u, v))
                        declared_nodes.add(v)
                        max_parsed_v = max(max_parsed_v, v)

        if num_vertices is not None and num_vertices > 0:
            n = max(num_vertices, max_parsed_v + 1)
        else:
            n = max(len(declared_nodes), max_parsed_v + 1)

        self.G.add_nodes_from(range(n))
        for u, v in edges:
            if 0 <= u < n and 0 <= v < n:
                self.G.add_edge(u, v)

    def pendant_neighbor_count(self, v):
        return sum(1 for u in self.full_neighbors[v] if self.is_pendant[u])

    def non_pendant_neighbor_count(self, v):
        return self.full_deg[v] - self.pendant_neighbor_count(v)

    def init_mu_vectors(self):
        """Structural initialisation of mu vectors."""
        mu = {}
        for v in self.G.nodes():
            if self.is_pendant[v]:
                val = 1
            elif self.pendant_neighbor_count(v) > self.full_deg[v] // 2:
                val = INF
            else:
                val = self.full_deg[v] // 2 + 1
            mu[v] = {c: val for c in COLOR_LETTERS}
        return mu

    def min_mu(self, v):
        return min(self.mu[v][c] for c in COLOR_LETTERS)

    def apply_mu_update(self, colored_vertex, color):
        """Updated v9 rules for mu propagation."""
        if self.is_pendant[colored_vertex]:
            for v in self.full_neighbors[colored_vertex]:
                self.mu[v][color] = 0
        else:
            for v in self.full_neighbors[colored_vertex]:
                if self.mu[v][color] != INF:
                    self.mu[v][color] -= 1

            # Own mu update
            u = colored_vertex
            if self.mu[u][color] != INF:
                self.mu[u][color] -= 1
            for j in COLOR_LETTERS:
                if j != color:
                    self.mu[u][j] = INF

    def is_majority_colored(self, u, trial_coloring=None):
        col_map = trial_coloring if trial_coloring is not None else self.coloring
        if u not in col_map:
            return False
        col = col_map[u]
        deg = self.full_deg[u]
        same_count = sum(1 for w in self.full_neighbors[u] if col_map.get(w) == col)
        return same_count == deg // 2

    def vertex_eligible_colors(self, v):
        excluded = set()
        for u in self.full_neighbors[v]:
            if self.is_majority_colored(u):
                excluded.add(self.coloring[u])
        return [c for c in COLOR_LETTERS if c not in excluded and self.mu[v][c] != 0]

    def min_mu_over_eligible(self, v):
        E_v = self.vertex_eligible_colors(v)
        if not E_v:
            return INF
        return min(self.mu[v][c] for c in E_v)

    def compute_lambdas(self):
        """Updated lambda calculator based on v9 specs."""
        result = {}
        for v in self.G.nodes():
            if v in self.coloring:
                continue

            E_v = self.vertex_eligible_colors(v)
            c = len(E_v)

            if self.is_pendant[v]:
                result[v] = (INF, INF, INF, c, E_v)
                continue

            N_v = self.full_neighbors[v]

            lambda_1 = INF
            if c == 0:
                lambda_1 = 0
            elif c <= len(N_v):
                for combo in combinations(N_v, c):
                    s = sum(self.min_mu_over_eligible(u) for u in combo)
                    if s < lambda_1:
                        lambda_1 = s

            lambda_2 = INF
            k = c - 1
            if k >= 0:
                base = self.min_mu_over_eligible(v)
                if k == 0:
                    lambda_2 = base
                elif k <= len(N_v):
                    for combo in combinations(N_v, k):
                        s = sum(self.min_mu_over_eligible(u) for u in combo) + base
                        if s < lambda_2:
                            lambda_2 = s

            lam = min(lambda_1, lambda_2)
            result[v] = (lambda_1, lambda_2, lam, c, E_v)

        return result

    def colored_neighbors_within_distance(self, v, max_dist=2):
        res = []
        for u in self.coloring:
            try:
                if nx.shortest_path_length(self.G, source=v, target=u) <= max_dist:
                    res.append(u)
            except nx.NetworkXNoPath:
                pass
        return res

    def eligible_colors_by_neighborhood_share(self, v):
        neighbors = self.full_neighbors[v]
        total = len(neighbors)
        counts = {c: 0 for c in COLOR_LETTERS}
        for u in neighbors:
            if u in self.coloring:
                counts[self.coloring[u]] += 1
        return [c for c in COLOR_LETTERS if counts[c] <= total // 2]

    def final_eligible_colors(self, v):
        v_elig = set(self.vertex_eligible_colors(v))
        share_elig = set(self.eligible_colors_by_neighborhood_share(v))
        return [c for c in COLOR_LETTERS if c in v_elig and c in share_elig]

    def mu_safe_colors(self, v):
        E_v = self.vertex_eligible_colors(v)
        safe = []
        for c in E_v:
            ok = True
            for vi in self.full_neighbors[v]:
                if self.mu[vi][c] <= self.min_mu(vi):
                    ok = False
                    break
            if ok:
                safe.append(c)
        return safe

    def candidate_colors_for_vertex(self, v):
        tier1 = self.mu_safe_colors(v)
        if tier1:
            return tier1

        nearby = self.colored_neighbors_within_distance(v, max_dist=2)
        counts = {c: 0 for c in COLOR_LETTERS}
        for u in nearby:
            counts[self.coloring[u]] += 1

        eligible = self.final_eligible_colors(v)
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

    def get_allowed_moves(self, round_no):
        lambdas = self.compute_lambdas()
        uncolored = [v for v in self.G.nodes() if v not in self.coloring]

        if not uncolored:
            return {}

        is_restricted = (round_no % 2 == 1)

        if is_restricted:
            min_val = min(lambdas[v][2] for v in uncolored)
            min_vets = [v for v in uncolored if lambdas[v][2] == min_val]

            menu = {}
            for v in min_vets:
                cands = self.candidate_colors_for_vertex(v)
                if cands:
                    menu[v] = cands
            return menu
        else:
            menu = {}
            for v in uncolored:
                cands = self.vertex_eligible_colors(v)
                if cands:
                    menu[v] = cands
            return menu

    def make_move(self, v, c, round_no):
        self.state_stack.append((copy.deepcopy(self.coloring), copy.deepcopy(self.mu)))
        self.coloring[v] = c
        self.apply_mu_update(v, c)
        is_restricted = (round_no % 2 == 1)
        self.move_history.append((round_no, v, c, is_restricted))

    def undo_move(self):
        if not self.state_stack or not self.move_history:
            return False

        prev_coloring, prev_mu = self.state_stack.pop()
        self.coloring = prev_coloring
        self.mu = prev_mu
        self.move_history.pop()
        return True


class InteractiveCanvas(FigureCanvas):
    """Interactive Matplotlib Canvas."""
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setParent(parent)

        self.engine = None
        self.pos = {}
        self.selected_node = None
        self.dragged_node = None
        self.round_no = 1
        
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
        self.on_node_selected_callback = None
        self.on_right_click_color_callback = None

    def set_engine(self, engine, layout_name='Kamada-Kawai', round_no=1):
        self.engine = engine
        self.selected_node = None
        self.round_no = round_no
        self.compute_layout(layout_name)
        self.redraw()

    def set_round(self, round_no):
        self.round_no = round_no

    def compute_layout(self, layout_name):
        if not self.engine or self.engine.G.number_of_nodes() == 0:
            return
        
        G = self.engine.G
        try:
            if layout_name == 'Kamada-Kawai':
                self.pos = nx.kamada_kawai_layout(G)
            elif layout_name == 'Spring':
                self.pos = nx.spring_layout(G, seed=42)
            elif layout_name == 'Spectral':
                self.pos = nx.spectral_layout(G)
            elif layout_name == 'Tree':
                if nx.is_tree(G):
                    self.pos = hierarchy_pos(G)
                else:
                    self.pos = nx.spring_layout(G, seed=42)
            else:
                self.pos = nx.spring_layout(G, seed=42)
        except Exception:
            self.pos = nx.spring_layout(G, seed=42)

    def redraw(self):
        self.ax.clear()
        if not self.engine or len(self.pos) == 0:
            self.draw()
            return

        G = self.engine.G
        lambdas = self.engine.compute_lambdas()
        allowed_moves = self.engine.get_allowed_moves(self.round_no)
        is_odd_round = (self.round_no % 2 == 1)

        order = max(1, G.number_of_nodes())
        node_size = max(600, int(2200 - 15 * order))
        font_size = max(9, int(13 - 0.08 * order))

        x_coords = [p[0] for p in self.pos.values()]
        y_coords = [p[1] for p in self.pos.values()]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        dx = max_x - min_x if max_x != min_x else 1.0
        dy = max_y - min_y if max_y != min_y else 1.0

        margin = 0.15
        self.ax.set_xlim(min_x - dx * margin, max_x + dx * margin)
        self.ax.set_ylim(min_y - dy * margin, max_y + dy * margin)

        nx.draw_networkx_edges(G, self.pos, ax=self.ax, edge_color='#b2bec3', width=2.0)

        node_colors = []
        edge_colors = []
        linewidths = []
        labels = {}

        for v in G.nodes():
            if v in self.engine.coloring:
                node_colors.append(COLOR_MAP[self.engine.coloring[v]])
                labels[v] = f"{v}:{self.engine.coloring[v]}"
                edge_colors.append('#2d3436' if v != self.selected_node else '#e67e22')
                linewidths.append(2.0 if v != self.selected_node else 4.0)
            else:
                node_colors.append(UNCOLORED_FILL)
                lam = lambdas.get(v, (None, None, INF))[2]
                lam_str = "inf" if lam == INF else str(lam)
                labels[v] = f"{v}:{lam_str}"

                if v == self.selected_node:
                    edge_colors.append('#e67e22')
                    linewidths.append(4.0)
                elif is_odd_round and v in allowed_moves:
                    edge_colors.append('#f1c40f')
                    linewidths.append(3.5)
                else:
                    edge_colors.append('#2d3436')
                    linewidths.append(2.0)

        nx.draw_networkx_nodes(
            G, self.pos, ax=self.ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors=edge_colors,
            linewidths=linewidths
        )

        for v, (x, y) in self.pos.items():
            self.ax.text(x, y, labels[v], fontsize=font_size, fontweight='bold',
                         ha='center', va='center', color='black')

        title_suffix = " (Gold Ring = Eligible Odd Move)" if is_odd_round else ""
        self.ax.set_title(f"Interactive Graph Canvas (Round {self.round_no}){title_suffix}", fontsize=11, pad=10)
        self.ax.axis('off')
        self.draw()

    def get_node_at_pos(self, event):
        if event.xdata is None or event.ydata is None or not self.pos:
            return None
        
        min_dist = float('inf')
        closest_node = None
        for v, (x, y) in self.pos.items():
            dist = math.hypot(x - event.xdata, y - event.ydata)
            if dist < min_dist:
                min_dist = dist
                closest_node = v

        x_coords = [p[0] for p in self.pos.values()]
        dx = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1.0
        threshold = max(0.1, dx * 0.1)

        if min_dist < threshold:
            return closest_node
        return None

    def on_press(self, event):
        node = self.get_node_at_pos(event)
        
        if event.button == 1:
            if node is not None:
                self.dragged_node = node
                self.selected_node = node
                if self.on_node_selected_callback:
                    self.on_node_selected_callback(node)
                self.redraw()
                
        elif event.button == 3:
            if node is not None:
                self.selected_node = node
                if self.on_node_selected_callback:
                    self.on_node_selected_callback(node)
                self.redraw()
                if self.on_right_click_color_callback:
                    self.on_right_click_color_callback(node)

    def on_motion(self, event):
        if self.dragged_node is not None and event.xdata is not None and event.ydata is not None:
            self.pos[self.dragged_node] = (event.xdata, event.ydata)
            self.redraw()

    def on_release(self, event):
        self.dragged_node = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lambda Vertex-Coloring (v9 Algorithm)")
        self.resize(1200, 800)

        self.round_no = 1
        self.engine = None

        self.init_ui()
        self.setup_shortcuts()
        self.load_default_graph()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- LEFT PANEL ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 1. Graph Config
        graph_box = QGroupBox("1. Graph & Layout Controls")
        gb_layout = QVBoxLayout(graph_box)

        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Number of Vertices (N):"))
        self.spin_vertices = QSpinBox()
        self.spin_vertices.setRange(1, 100)
        self.spin_vertices.setValue(5)
        n_layout.addWidget(self.spin_vertices)
        gb_layout.addLayout(n_layout)

        gb_layout.addWidget(QLabel("Adjacency List (v: neighbors...):"))
        self.adj_input = QTextEdit()
        self.adj_input.setPlaceholderText("Example:\n0: 1 2 3 4\n1: 0 2\n2: 0 1 3\n3: 0 2 4\n4: 0 3")
        self.adj_input.setMaximumHeight(100)
        gb_layout.addWidget(self.adj_input)

        layout_btn_layout = QHBoxLayout()
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(['Kamada-Kawai', 'Spring', 'Spectral', 'Tree'])
        self.layout_combo.currentTextChanged.connect(self.on_layout_change)
        
        btn_build = QPushButton("Build & Initialize Graph")
        btn_build.clicked.connect(self.build_graph)
        layout_btn_layout.addWidget(QLabel("Layout:"))
        layout_btn_layout.addWidget(self.layout_combo)
        layout_btn_layout.addWidget(btn_build)
        gb_layout.addLayout(layout_btn_layout)

        left_layout.addWidget(graph_box)

        # 2. Controls & Moves
        move_box = QGroupBox("2. Move & Color Control")
        mb_layout = QVBoxLayout(move_box)

        self.lbl_round_info = QLabel("Round 1 (ODD - Restricted)")
        self.lbl_round_info.setStyleSheet("font-weight: bold; color: #2c3e50;")
        mb_layout.addWidget(self.lbl_round_info)

        color_selection_layout = QHBoxLayout()
        self.lbl_selected_node = QLabel("Selected Node: None")
        self.combo_colors = QComboBox()
        
        color_selection_layout.addWidget(self.lbl_selected_node)
        color_selection_layout.addWidget(QLabel("Color:"))
        color_selection_layout.addWidget(self.combo_colors)
        mb_layout.addLayout(color_selection_layout)

        btn_layout = QHBoxLayout()
        self.btn_apply_move = QPushButton("Apply Color Move")
        self.btn_apply_move.clicked.connect(self.apply_color_move)
        
        self.btn_undo = QPushButton("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo_move)

        self.btn_save_log = QPushButton("Save Game Log (TXT)")
        self.btn_save_log.clicked.connect(self.save_move_log)
        
        btn_layout.addWidget(self.btn_apply_move)
        btn_layout.addWidget(self.btn_undo)
        btn_layout.addWidget(self.btn_save_log)
        mb_layout.addLayout(btn_layout)

        left_layout.addWidget(move_box)

        # 3. Live Parameter & Mu Vector Tables
        param_box = QGroupBox("3. Live Engine Parameters")
        pb_layout = QVBoxLayout(param_box)

        self.tabs = QTabWidget()

        # Tab 1: Lambda Values
        self.table_lambdas = QTableWidget()
        self.table_lambdas.setColumnCount(5)
        self.table_lambdas.setHorizontalHeaderLabels(['v', 'λ1', 'λ2', 'λ', 'E(v)'])
        self.table_lambdas.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.table_lambdas, "Lambda Values (λ)")

        # Tab 2: Mu Vectors
        self.table_mus = QTableWidget()
        self.table_mus.setColumnCount(5)
        self.table_mus.setHorizontalHeaderLabels(['v', 'μ(r)', 'μ(g)', 'μ(b)', 'min μ'])
        self.table_mus.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.table_mus, "Mu Vectors (μ)")

        pb_layout.addWidget(self.tabs)
        left_layout.addWidget(param_box)

        # --- RIGHT PANEL ---
        self.canvas = InteractiveCanvas(self)
        self.canvas.on_node_selected_callback = self.on_node_selected_from_canvas
        self.canvas.on_right_click_color_callback = self.show_right_click_menu

        splitter.addWidget(left_widget)
        splitter.addWidget(self.canvas)
        splitter.setSizes([480, 720])

    def setup_shortcuts(self):
        self.shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo.activated.connect(self.undo_move)

    def load_default_graph(self):
        sample_adj = "0: 1 2 3 4\n1: 0 2\n2: 0 1 3\n3: 0 2 4\n4: 0 3"
        self.spin_vertices.setValue(5)
        self.adj_input.setText(sample_adj)
        self.build_graph()

    def build_graph(self):
        text = self.adj_input.toPlainText()
        n_val = self.spin_vertices.value()
        self.engine = LambdaColoringEngine(text, num_vertices=n_val)
        
        self.spin_vertices.setValue(self.engine.G.number_of_nodes())
        self.round_no = 1
        self.canvas.set_engine(self.engine, self.layout_combo.currentText(), self.round_no)
        self.update_ui_state()

    def on_layout_change(self, layout_name):
        if self.canvas:
            self.canvas.compute_layout(layout_name)
            self.canvas.redraw()

    def on_node_selected_from_canvas(self, node):
        self.lbl_selected_node.setText(f"Selected Node: {node}")
        self.update_color_dropdown(node)

    def show_right_click_menu(self, node):
        if self.engine is None or node in self.engine.coloring:
            return

        allowed_moves = self.engine.get_allowed_moves(self.round_no)
        if node not in allowed_moves or not allowed_moves[node]:
            return

        menu = QMenu(self)
        color_names = {'r': 'Red (r)', 'g': 'Green (g)', 'b': 'Blue (b)'}

        for c in allowed_moves[node]:
            action = QAction(f"Color: {color_names.get(c, c.upper())}", self)
            action.triggered.connect(lambda checked, col=c: self.commit_move(node, col))
            menu.addAction(action)

        menu.exec_(QCursor.pos())

    def update_color_dropdown(self, node):
        self.combo_colors.clear()
        if self.engine is None or node is None or node in self.engine.coloring:
            return

        allowed_moves = self.engine.get_allowed_moves(self.round_no)
        if node in allowed_moves:
            for c in allowed_moves[node]:
                self.combo_colors.addItem(c.upper(), c)

    def update_ui_state(self):
        if not self.engine:
            return

        is_restricted = (self.round_no % 2 == 1)
        round_type = "ODD - Restricted" if is_restricted else "EVEN - Free Choice"
        self.lbl_round_info.setText(f"Round {self.round_no} ({round_type})")
        self.canvas.set_round(self.round_no)

        fmt = lambda x: "inf" if x == INF else str(x)

        # 1. Refresh Lambdas Table
        lambdas = self.engine.compute_lambdas()
        self.table_lambdas.setRowCount(len(lambdas))
        for i, (v, data) in enumerate(sorted(lambdas.items())):
            l1, l2, lam, c, E_v = data
            self.table_lambdas.setItem(i, 0, QTableWidgetItem(str(v)))
            self.table_lambdas.setItem(i, 1, QTableWidgetItem(fmt(l1)))
            self.table_lambdas.setItem(i, 2, QTableWidgetItem(fmt(l2)))
            self.table_lambdas.setItem(i, 3, QTableWidgetItem(fmt(lam)))
            self.table_lambdas.setItem(i, 4, QTableWidgetItem(",".join(E_v)))

        # 2. Refresh Mu Vectors Table
        nodes = sorted(self.engine.G.nodes())
        self.table_mus.setRowCount(len(nodes))
        for i, v in enumerate(nodes):
            mu_v = self.engine.mu[v]
            min_mu_val = self.engine.min_mu(v)
            
            self.table_mus.setItem(i, 0, QTableWidgetItem(str(v)))
            self.table_mus.setItem(i, 1, QTableWidgetItem(fmt(mu_v['r'])))
            self.table_mus.setItem(i, 2, QTableWidgetItem(fmt(mu_v['g'])))
            self.table_mus.setItem(i, 3, QTableWidgetItem(fmt(mu_v['b'])))
            self.table_mus.setItem(i, 4, QTableWidgetItem(fmt(min_mu_val)))

        if self.canvas.selected_node is not None:
            self.update_color_dropdown(self.canvas.selected_node)

        allowed_moves = self.engine.get_allowed_moves(self.round_no)
        uncolored = [v for v in self.engine.G.nodes() if v not in self.engine.coloring]
        
        if not uncolored:
            QMessageBox.information(self, "Game Finished", "All vertices successfully colored!")
            self.save_move_log()
        elif not allowed_moves:
            QMessageBox.warning(self, "You Win", "No valid vertex/colour combination survives the restrictions this round.")

    def apply_color_move(self):
        node = self.canvas.selected_node
        if node is None:
            QMessageBox.warning(self, "Selection Error", "Please click/select a vertex on the canvas first.")
            return

        color_data = self.combo_colors.currentData()
        if not color_data:
            QMessageBox.warning(self, "Move Error", "No eligible color available for selected vertex.")
            return

        self.commit_move(node, color_data)

    def commit_move(self, node, color_data):
        self.engine.make_move(node, color_data, self.round_no)
        self.round_no += 1

        self.canvas.selected_node = None
        self.lbl_selected_node.setText("Selected Node: None")
        self.canvas.redraw()
        self.update_ui_state()

    def undo_move(self):
        if self.engine and self.engine.undo_move():
            self.round_no -= 1
            self.canvas.selected_node = None
            self.lbl_selected_node.setText("Selected Node: None")
            self.canvas.redraw()
            self.update_ui_state()
        else:
            QMessageBox.information(self, "Undo", "No moves left to undo.")

    def save_move_log(self):
        if not self.engine or not self.engine.move_history:
            QMessageBox.information(self, "Save Log", "No moves recorded yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Game Move Log", "move_log.txt", "Text Files (*.txt)")
        if path:
            with open(path, 'w') as f:
                f.write("=== LAMBDA VERTEX COLORING MOVE LOG ===\n\n")
                for r, v, c, restricted in self.engine.move_history:
                    mode = "Restricted" if restricted else "Free Choice"
                    c_name = "Red (r)" if c == 'r' else ("Green (g)" if c == 'g' else "Blue (b)")
                    f.write(f"Move {r:02d} | Round {r} ({mode}) -> Colored Vertex [{v}] with {c_name}\n")
                f.write("\n=== END OF RECORD ===\n")
            QMessageBox.information(self, "Saved", f"Move log successfully saved to:\n{path}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()