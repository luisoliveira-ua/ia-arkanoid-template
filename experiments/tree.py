"""Generic tree-search utilities used by the Arkanoid planning agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class SearchDomain(ABC):
    """Interface a search domain must implement."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def actions(self, state: Any) -> Iterable[Any]:
        """Return the actions available from ``state``."""

    @abstractmethod
    def result(self, state: Any, action: Any) -> Any:
        """Return the state reached by applying ``action``."""

    @abstractmethod
    def cost(self, state: Any, action: Any) -> float:
        """Return the cost of applying ``action`` in ``state``."""

    @abstractmethod
    def heuristic(self, state: Any, goal: Any) -> float:
        """Estimate the remaining cost from ``state`` to ``goal``."""

    @abstractmethod
    def satisfies(self, state: Any, goal: Any) -> bool:
        """Return whether ``state`` satisfies ``goal``."""


class SearchProblem:
    """A domain together with its initial state and goal."""

    def __init__(self, domain: SearchDomain, initial: Any, goal: Any) -> None:
        self.domain = domain
        self.initial = initial
        self.goal = goal

    def goal_test(self, state: Any) -> bool:
        return self.domain.satisfies(state, self.goal)


class SearchNode:
    """One state in a search tree, linked to its parent for path recovery."""

    __slots__ = ("state", "parent", "profundidade", "custo", "heuristica", "action")

    def __init__(
        self,
        state: Any,
        parent: SearchNode | None,
        profundidade: int,
        custo: float,
        heuristica: float,
        action: Any = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.profundidade = profundidade
        self.custo = custo
        self.heuristica = heuristica
        self.action = action

    def __str__(self) -> str:
        return f"no({self.state},{self.parent},{self.profundidade})"

    __repr__ = __str__


class SearchTree:
    """Tree search supporting breadth, depth, cost, heuristic, A*, and rate.

    ``rate`` is a bounded best-first search: it orders the frontier by
    ``g + h`` and returns the explored state with the highest points/second.
    It is intended for domains that provide ``points_per_second(state)``.
    """

    STRATEGIES = {"breadth", "depth", "uniform", "gulosa", "a*", "rate"}

    def __init__(
        self,
        problem: SearchProblem,
        strategy: str = "breadth",
        limite: int = 100000,
        node_limit: int = 20000,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown search strategy: {strategy!r}")
        if limite < 0:
            raise ValueError("limite must be non-negative")
        if node_limit < 1:
            raise ValueError("node_limit must be positive")

        self.problem = problem
        self.strategy = strategy
        self.limite = limite
        self.node_limit = node_limit
        root = SearchNode(
            problem.initial,
            None,
            0,
            0.0,
            float(problem.domain.heuristic(problem.initial, problem.goal)),
        )
        self.open_nodes: list[SearchNode] = [root]
        self.solution: SearchNode | None = None
        self.nosterm = 0
        self.nosnterm = 0
        self.nosfilhos = 0
        self.nosacu = [root]
        self.prof_acumulada = 0
        self.totalnos = 1

    def get_path(self, node: SearchNode) -> list[Any]:
        """Return states from the root to ``node`` without recursive calls."""
        path: list[Any] = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path

    def get_action_path(self, node: SearchNode) -> list[Any]:
        """Return the actions from the root to ``node``."""
        actions: list[Any] = []
        while node is not None and node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    def search(self) -> list[Any] | None:
        """Search until a goal is found or the frontier/budget is exhausted."""
        best_rate_node: SearchNode | None = None
        best_rate = float("-inf")

        while self.open_nodes:
            node = self.open_nodes.pop(0)
            self.nosnterm += 1

            is_goal = self.problem.goal_test(node.state)
            if is_goal and self.strategy != "rate":
                self.solution = node
                return self.get_path(node)

            if self.strategy == "rate":
                rate = self._points_per_second(node.state)
                if rate > best_rate:
                    best_rate, best_rate_node = rate, node

            # Depth-limited strategies keep nodes at the limit as candidates,
            # but do not generate children beyond it.
            if is_goal or node.profundidade >= self.limite:
                self.nosterm += 1
                continue

            actions = list(self.problem.domain.actions(node.state))
            if not actions:
                self.nosterm += 1
                continue

            children: list[SearchNode] = []
            for action in actions:
                new_state = self.problem.domain.result(node.state, action)
                if self.in_path(node, new_state):
                    continue

                step_cost = float(self.problem.domain.cost(node.state, action))
                heuristic = float(
                    self.problem.domain.heuristic(new_state, self.problem.goal)
                )
                child = SearchNode(
                    new_state,
                    node,
                    node.profundidade + 1,
                    node.custo + step_cost,
                    heuristic,
                    action,
                )
                children.append(child)
                self.totalnos += 1
                self.nosfilhos += 1
                self.prof_acumulada += child.profundidade

            if not children:
                self.nosterm += 1
            self._record_max_cost(children)
            self.add_to_open(children)

            if self.strategy == "rate" and self.totalnos >= self.node_limit:
                break

        if self.strategy == "rate" and best_rate_node is not None:
            self.solution = best_rate_node
            return self.get_path(best_rate_node)
        return None

    def _points_per_second(self, state: Any) -> float:
        scorer = getattr(self.problem.domain, "points_per_second", None)
        if scorer is None:
            raise TypeError("The 'rate' strategy requires points_per_second(state)")
        return float(scorer(state))

    def _record_max_cost(self, nodes: list[SearchNode]) -> None:
        for node in nodes:
            if node.custo > self.nosacu[0].custo:
                self.nosacu = [node]
            elif node.custo == self.nosacu[0].custo:
                self.nosacu.append(node)

    def add_to_open(self, new_nodes: list[SearchNode]) -> None:
        """Add nodes to the frontier according to the selected strategy."""
        if not new_nodes:
            return
        if self.strategy == "breadth":
            self.open_nodes.extend(new_nodes)
        elif self.strategy == "depth":
            self.open_nodes[0:0] = new_nodes
        else:
            self.open_nodes.extend(new_nodes)
            if self.strategy == "uniform":
                key = lambda node: node.custo
            elif self.strategy == "gulosa":
                key = lambda node: node.heuristica
            else:  # a* and rate
                key = lambda node: node.custo + node.heuristica
            self.open_nodes.sort(key=key)

    @staticmethod
    def in_path(node: SearchNode | None, state: Any) -> bool:
        """Check whether ``state`` already occurs among this node's ancestors."""
        while node is not None:
            if node.state == state:
                return True
            node = node.parent
        return False

    def total(self) -> int:
        """Count reachable nodes up to the configured depth and node budget.

        This preserves the historical helper while making it finite for
        domains with cycles or a large branching factor.
        """
        frontier = [self.open_nodes[0]] if self.open_nodes else []
        counted = 0
        while frontier and counted < self.node_limit:
            node = frontier.pop(0)
            counted += 1
            if node.profundidade >= self.limite or self.problem.goal_test(node.state):
                continue
            for action in self.problem.domain.actions(node.state):
                state = self.problem.domain.result(node.state, action)
                if not self.in_path(node, state):
                    frontier.append(
                        SearchNode(state, node, node.profundidade + 1, 0.0, 0.0)
                    )
        return counted

    @property
    def length(self) -> int | None:
        return None if self.solution is None else self.solution.profundidade

    @property
    def avg_branching(self) -> float:
        return self.nosfilhos / self.nosnterm if self.nosnterm else 0.0

    @property
    def custo(self) -> float | None:
        if self.solution is None:
            self.search()
        return None if self.solution is None else self.solution.custo

    @property
    def profmedia(self) -> float:
        return self.prof_acumulada / self.totalnos if self.totalnos else 0.0
