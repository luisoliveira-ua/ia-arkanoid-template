"""Tests for the exact flight replica and the expectimax rebound planner."""

import random
import unittest

from agents.expectimax_agent import ExpectimaxAgent
from agents.fast_physics import Geometry, fly, triangular_quantile
from server.config import load_config
from server.highscores import HighScoreManager
from server.logic import Arkanoid


def _game(level: int, seed: int) -> Arkanoid:
    random.seed(seed)
    config = load_config()
    config.game.level_sequence = [f"level{level}.json"]
    config.game.level_time_limit = 0.0
    return Arkanoid(config, HighScoreManager(None))


def _small_agent(**overrides: object) -> ExpectimaxAgent:
    agent = ExpectimaxAgent("ws://localhost:1/ws", "Test", "test", unlimited_budget=True)
    agent.ROOT_SAMPLES = 8
    agent.GRID_BINS = 12
    agent.GRID_SAMPLES = 3
    agent.NEXT_BOARD_LIMIT = 2
    agent.NEXT_BOARD_BINS = 8
    agent.NEXT_BOARD_SAMPLES = 2
    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent


def _act(agent: ExpectimaxAgent, game: Arkanoid) -> None:
    agent.current_state = game.get_state()
    coroutine = agent.deliberate()
    try:
        coroutine.send(None)
    except StopIteration as done:
        action = done.value
    else:  # pragma: no cover - deliberate never awaits
        raise AssertionError("deliberate() must not suspend")
    if action:
        game.move_paddle(action["direction"])


class TestFastPhysics(unittest.TestCase):
    def test_flight_matches_the_server_engine(self) -> None:
        """Every predicted paddle contact equals the engine's, for random rebound zones."""
        for level in (1, 2, 3):
            game = _game(level, 7)
            geo = Geometry(game.get_state())
            contacts: list[float] = []
            original = game._handle_paddle_bounce

            def record(x: float, original=original, contacts=contacts) -> None:
                contacts.append(x)
                original(x)

            game._handle_paddle_bounce = record  # type: ignore[method-assign]
            checked = 0
            prediction = None
            for _ in range(30 * 120):
                state = game.get_state()
                if prediction is None and state["ball_vy"] < 0:
                    health = geo.health_from(state)
                    prediction = fly(geo, health, state["ball_x"], state["ball_y"],
                                     state["ball_vx"], state["ball_vy"])
                    rel = random.choice((14.0, 42.0, 70.0))
                    game.paddle_x = min(516.0, max(0.0, prediction[3] - rel))
                before = len(contacts)
                game.update(1.0 / 30.0)
                if game.game_over or game.level_idx > 0:
                    break
                if prediction is not None and len(contacts) > before:
                    self.assertAlmostEqual(contacts[-1], prediction[3], places=6)
                    self.assertEqual(geo.health_from(game.get_state()), prediction[4])
                    checked += 1
                    prediction = None
            self.assertGreater(checked, 5)

    def test_triangular_quantile_spans_the_cone(self) -> None:
        self.assertAlmostEqual(triangular_quantile(0.0, -10.0, 10.0, 0.0), -10.0)
        self.assertAlmostEqual(triangular_quantile(0.5, -10.0, 10.0, 0.0), 0.0)
        self.assertAlmostEqual(triangular_quantile(1.0, -10.0, 10.0, 0.0), 10.0)


class TestExpectimaxAgent(unittest.TestCase):
    def test_agent_keeps_every_life_while_scoring(self) -> None:
        game = _game(1, 3)
        agent = _small_agent()
        for _ in range(30 * 40):
            _act(agent, game)
            game.update(1.0 / 30.0)
        self.assertEqual(game.lives, 3)
        self.assertGreater(game.score, 0)
        self.assertTrue(agent.max_depth_reached)

    def test_greedy_only_and_budgeted_modes_play(self) -> None:
        game = _game(2, 5)
        greedy = _small_agent(USE_VALUE_GRID=False)
        budgeted = _small_agent(NEXT_BOARD_TABLES=False)
        budgeted.unlimited_budget = False
        for agent in (greedy, budgeted):
            for _ in range(30 * 15):
                _act(agent, game)
                game.update(1.0 / 30.0)
        self.assertEqual(game.lives, 3)

    def test_grid_flights_are_cached_per_board(self) -> None:
        game = _game(1, 4)
        agent = _small_agent()
        geo = agent._sync_level(game.get_state())
        health = geo.health_from(game.get_state())
        first = list(agent._grid_flights(geo, health, 6, 2))
        self.assertTrue(first)  # simulated: yields once per flight
        self.assertEqual(list(agent._grid_flights(geo, health, 6, 2)), [])  # cached: no flight

    def test_endgame_time_to_clear_last_bricks(self) -> None:
        game = _game(1, 4)
        agent = _small_agent(ENDGAME_BINS=8, ENDGAME_SAMPLES=2)
        geo = agent._sync_level(game.get_state())
        health = geo.health_from(game.get_state())
        # Keep only the first two destructible bricks, one hit each.
        kept = [i for i, h in enumerate(health) if h > 0][:2]
        endgame = [h if h < 0 else (1 if i in kept else 0) for i, h in enumerate(health)]

        def solve(board: list[int]) -> tuple[object, object]:
            run = agent._time_to_clear(geo, board)
            with self.assertRaises(StopIteration) as done:
                while True:
                    next(run)
            return done.exception.value

        xs, seconds = solve(endgame)
        self.assertEqual(len(xs), 8)
        self.assertTrue(all(0.0 < s < 1e6 for s in seconds))
        # Clearing two bricks takes longer than clearing one of them.
        one = [h if h < 0 else (1 if i == kept[0] else 0) for i, h in enumerate(health)]
        self.assertGreater(seconds.mean(), solve(one)[1].mean())

    def test_ignores_finished_and_duplicate_states(self) -> None:
        game = _game(1, 2)
        agent = _small_agent()
        agent.current_state = {"game_over": True}
        coroutine = agent.deliberate()
        with self.assertRaises(StopIteration) as done:
            coroutine.send(None)
        self.assertIsNone(done.exception.value)
        _act(agent, game)
        coroutine = agent.deliberate()  # same state again: stale duplicate
        with self.assertRaises(StopIteration) as done:
            coroutine.send(None)
        self.assertIsNone(done.exception.value)


if __name__ == "__main__":
    unittest.main()
