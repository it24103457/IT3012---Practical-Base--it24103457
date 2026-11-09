# agent.py
from collections import deque
import heapq
import math

from logic_engine import KnowledgeBase


class SearchAgent:
    """Problem-solving agent with BFS, DFS, UCS, and A* graph-search algorithms."""

    VALID_ALGOS = ('BFS', 'DFS', 'UCS', 'AStar')

    def __init__(self):
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.plan = []
        # Lab 07 is built on the KB-aware A* planner.
        self.active_algo = 'AStar'

        # Part 3.1: Knowledge Base + safety rules (Horn clauses)
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')           # Rule 1
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')      # Rule 2

    def sense_and_act(self, percept: dict) -> str:
        """Build a plan to the closest food if none is queued, then return the next action."""
        if self.active_algo not in self.VALID_ALGOS:
            raise ValueError(f"Unknown search algorithm: {self.active_algo!r}")

        start = tuple(percept['agent_pos'])

        # Part 3.3: per-tile facts supplied by us (the game does not produce them)
        tile_facts = {
            (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],  # deduces Retreat -> unsafe
            (0, 1): ['TargetVisible', 'HasDust'],                          # SafeToEngage only -> safe
        }

        if not self.plan:
            foods = percept['all_food']
            if foods:
                targets = sorted(foods, key=lambda f: self.manhattan_distance(start, f))
                if self.active_algo == 'AStar':
                    # Try foods nearest-first, so one logically unreachable food does not stall the agent
                    for target in targets:
                        self.plan = self.astar_search(start, target, percept['walls'], percept['grid_size'],
                                                      tile_facts=tile_facts) or []
                        if self.plan:
                            break
                else:
                    method = {
                        'BFS': self.bfs_search,
                        'DFS': self.dfs_search,
                        'UCS': self.ucs_search,
                    }[self.active_algo]
                    self.plan = method(start, targets[0], percept['walls'], percept['grid_size']) or []

        if self.plan:
            return self.plan.pop(0)
        return self._safe_fallback(start, percept['walls'], percept['grid_size'], tile_facts)

    # ------------------------------------------------------------------
    # Heuristic functions
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)"""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def _neighbors(self, pos, walls, grid_size):
        """Yield (action, new_pos) for each valid move from pos."""
        width, height = grid_size
        x, y = pos
        moves = {
            'Up': (x, y + 1),
            'Down': (x, y - 1),
            'Left': (x - 1, y),
            'Right': (x + 1, y),
        }
        for action, (nx, ny) in moves.items():
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    def _reconstruct(self, parent, start, goal):
        """Walk back from goal to start, returning a list of action strings."""
        if start == goal:
            return []
        moves = {
            (0, 1): 'Up',
            (0, -1): 'Down',
            (-1, 0): 'Left',
            (1, 0): 'Right',
        }
        path = []
        cur = goal
        while cur in parent:
            prev = parent[cur]
            dx, dy = cur[0] - prev[0], cur[1] - prev[1]
            path.append(moves[(dx, dy)])
            cur = prev
        path.reverse()
        return path

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-first search using a FIFO queue. Returns optimal action list or None."""
        walls = set(walls)
        frontier = deque([start_pos])
        parent = {}
        visited = {start_pos}

        while frontier:
            current = frontier.popleft()
            if current == goal_pos:
                return self._reconstruct(parent, start_pos, current)

            for action, nxt in self._neighbors(current, walls, grid_size):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = current
                    frontier.append(nxt)
        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-first search using a LIFO stack. Returns an action list or None."""
        walls = set(walls)
        frontier = [start_pos]
        parent = {}
        visited = {start_pos}

        while frontier:
            current = frontier.pop()
            if current == goal_pos:
                return self._reconstruct(parent, start_pos, current)

            for action, nxt in self._neighbors(current, walls, grid_size):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = current
                    frontier.append(nxt)
        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-cost search using a priority queue ordered by path cost g(n)."""
        walls = set(walls)
        frontier = [(0, start_pos)]
        parent = {}
        cost_so_far = {start_pos: 0}

        while frontier:
            cost, current = heapq.heappop(frontier)
            if current == goal_pos:
                return self._reconstruct(parent, start_pos, current)

            for action, nxt in self._neighbors(current, walls, grid_size):
                new_cost = cost + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    parent[nxt] = current
                    heapq.heappush(frontier, (new_cost, nxt))
        return None

    def _is_safe(self, tile, tile_facts):
        """Part 3.2: a tile is feasible only when the KB cannot derive 'Retreat'."""
        self.kb.clear_facts()
        for fact in tile_facts.get(tile, []):
            self.kb.tell_fact(fact)
        self.kb.forward_chain()
        return 'Retreat' not in self.kb.facts

    def _safe_fallback(self, start, walls, grid_size, tile_facts):
        """Return a physically valid, logically safe move when no plan exists."""
        for action, nxt in self._neighbors(start, set(walls), grid_size):
            if self._is_safe(nxt, tile_facts):
                return action
        return 'Up'

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan', tile_facts=None):
        """A* search: f(n) = g(n) + h(n). Returns optimal action list or None.
        Tiles where the KB deduces 'Retreat' are logically infeasible and are never
        expanded (the start tile itself is not checked)."""
        walls = set(walls)
        tile_facts = tile_facts or {}
        heuristic = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance

        frontier = []
        h_start = heuristic(start_pos, goal_pos)
        # tuple format: (f_cost, g_cost, current_pos, path_taken)
        heapq.heappush(frontier, (h_start, 0, start_pos, []))
        reached_states = set()

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for action, nxt in self._neighbors(current_pos, walls, grid_size):
                if nxt not in reached_states:
                    # Part 3.2: reachable is not enough; the tile must be logically feasible.
                    if not self._is_safe(nxt, tile_facts):
                        continue   # infeasible: skip even though physically reachable

                    g_new = g_cost + 1
                    h_new = heuristic(nxt, goal_pos)
                    f_new = g_new + h_new
                    heapq.heappush(frontier, (f_new, g_new, nxt, path_taken + [action]))
        return None


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)