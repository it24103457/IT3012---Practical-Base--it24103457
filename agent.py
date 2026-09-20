# agent.py
from collections import deque
import heapq
import math


class SearchAgent:
    """Problem-solving agent with BFS, DFS, UCS, and A* graph-search algorithms."""

    def __init__(self):
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.plan = []
        self.active_algo = 'BFS'

    def sense_and_act(self, percept: dict) -> str:
        """Build a plan to the closest food if none is queued, then return the next action."""
        if not self.plan:
            start = tuple(percept['agent_pos'])
            foods = percept['all_food']
            if foods:
                target = min(foods, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))
                if self.active_algo == 'BFS':
                    self.plan = self.bfs_search(start, target, percept['walls'], percept['grid_size']) or []
                elif self.active_algo == 'DFS':
                    self.plan = self.dfs_search(start, target, percept['walls'], percept['grid_size']) or []
                elif self.active_algo == 'UCS':
                    self.plan = self.ucs_search(start, target, percept['walls'], percept['grid_size']) or []
                elif self.active_algo == 'AStar':
                    self.plan = self.astar_search(start, target, percept['walls'], percept['grid_size']) or []
        if self.plan:
            return self.plan.pop(0)
        return 'Up'

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

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """A* search: f(n) = g(n) + h(n). Returns optimal action list or None."""
        walls = set(walls)
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