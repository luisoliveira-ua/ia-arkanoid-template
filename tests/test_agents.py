import asyncio
import contextlib
import io
import json
import unittest
import unittest.mock as mock
from collections.abc import AsyncIterator
from typing import Any

from agents.base_agent import BaseAgent
from agents.dummy_agent import DummyAgent
from agents.manual_agent import receive_loop, send_loop


class ConcreteBaseAgent(BaseAgent):
    """Subclass implementing deliberate for testing BaseAgent."""

    async def deliberate(self) -> dict[str, str] | None:
        return {"action": "move", "direction": "WEST"}


class TestAgents(unittest.IsolatedAsyncioTestCase):
    async def test_base_agent_deliberate_not_implemented(self) -> None:
        agent = BaseAgent()
        self.assertEqual(agent.server_uri, "ws://localhost:8765/ws")
        self.assertIsNone(agent.player_id)
        self.assertIsNone(agent.setup_data)
        with self.assertRaises(NotImplementedError):
            await agent.deliberate()

        concrete = ConcreteBaseAgent()
        concrete_act = await concrete.deliberate()
        self.assertEqual(concrete_act, {"action": "move", "direction": "WEST"})

    async def test_dummy_agent_deliberation(self) -> None:
        agent = DummyAgent()

        # No state
        action = await agent.deliberate()
        self.assertIsNone(action)

        # Game over state
        agent.current_state = {"game_over": True, "valid_actions": [{"action": "move", "direction": "WEST"}]}
        action = await agent.deliberate()
        self.assertIsNone(action)

        # Empty valid actions
        agent.current_state = {"game_over": False, "valid_actions": []}
        action = await agent.deliberate()
        self.assertIsNone(action)

        # State with valid actions
        valid_moves = [
            {"action": "move", "direction": "WEST"},
            {"action": "move", "direction": "EAST"},
        ]
        agent.current_state = {"game_over": False, "valid_actions": valid_moves}

        # Run several times to test that choices come from valid_actions or None
        sampled_actions = set()
        for _ in range(50):
            chosen = await agent.deliberate()
            if chosen is not None:
                self.assertIn(chosen, valid_moves)
                sampled_actions.add(chosen["direction"])
            else:
                sampled_actions.add("NONE")

        # Expect to see valid choices sampled
        self.assertTrue(len(sampled_actions) > 0)

    async def test_base_agent_run_loop(self) -> None:
        agent = ConcreteBaseAgent()
        messages = [
            '{"type": "setup", "player_id": 42}',
            '{"type": "state", "game_over": false, "valid_actions": [{"action": "move", "direction": "WEST"}]}',
        ]
        sent_messages: list[str] = []

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                sent_messages.append(msg)

            async def __aiter__(self):
                for m in messages:
                    yield m

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        import unittest.mock as mock

        with mock.patch("websockets.connect", return_value=MockConnectContext()):
            await agent.run()

        self.assertEqual(agent.player_id, 42)
        self.assertIsNotNone(agent.setup_data)
        assert agent.setup_data is not None
        self.assertEqual(agent.setup_data.get("player_id"), 42)
        # Verify action was sent
        self.assertEqual(len(sent_messages), 2)  # handshake + action

    async def test_base_agent_stops_on_game_over(self) -> None:
        agent = ConcreteBaseAgent()
        messages = [
            '{"type": "setup", "player_id": 1}',
            '{"type": "state", "game_over": true}',
            '{"type": "state", "game_over": false}',  # Should never be reached
        ]
        sent_messages: list[str] = []

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                sent_messages.append(msg)

            async def __aiter__(self):
                for m in messages:
                    yield m

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        import unittest.mock as mock

        with mock.patch("websockets.connect", return_value=MockConnectContext()):
            await agent.run()

        # Only initial handshake sent; no move action after game over
        self.assertEqual(len(sent_messages), 1)
        self.assertIn("agent", sent_messages[0])

    async def test_base_agent_prints_a_summary_when_it_stops(self) -> None:
        import contextlib
        import io
        import unittest.mock as mock

        state = {
            "type": "state", "game_over": True, "timed_out": True, "score": 321, "avg_norm_score": 7.5,
            "normalized_score": 6.25,
            "time_elapsed": 640.0, "level_time_elapsed": 600.0, "level_time_limit": 600.0, "stage": 2, "max_stages": 3,
            "stage_name": "Level 2", "lives": 1, "high_score": 400, "high_score_normalized": 8.0,
        }  # fmt: skip

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                return None

            async def __aiter__(self):
                yield '{"type": "setup", "player_id": 1}'
                yield json.dumps(state)

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        agent = ConcreteBaseAgent(name="Summary Bot")
        out = io.StringIO()
        with mock.patch("websockets.connect", return_value=MockConnectContext()), contextlib.redirect_stdout(out):
            await agent.run()
        text = out.getvalue()
        self.assertIn("AGENT STOPPED", text)
        self.assertIn("Summary Bot", text)
        self.assertIn("time limit reached", text)
        self.assertIn("2 of 3: Level 2", text)
        self.assertIn("7.50 points/s", text)
        self.assertIn("600.0 s of 600 s", text)

    def test_summary_caps_the_level_after_the_last_one_is_cleared(self) -> None:
        agent = ConcreteBaseAgent(name="Won")
        agent.current_state = {"stage": 4, "max_stages": 3, "stage_name": "Stage 3"}
        self.assertIn("3 of 3", agent.summary())

    def test_summary_without_any_state_does_not_fail(self) -> None:
        agent = ConcreteBaseAgent(name="Idle")
        self.assertIn("connection closed", agent.summary())

    async def test_base_agent_stops_on_game_won(self) -> None:
        agent = ConcreteBaseAgent()
        messages = [
            '{"type": "setup", "player_id": 1}',
            '{"type": "state", "game_won": true, "game_over": true}',
        ]
        sent_messages: list[str] = []

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                sent_messages.append(msg)

            async def __aiter__(self):
                for m in messages:
                    yield m

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        import unittest.mock as mock

        with mock.patch("websockets.connect", return_value=MockConnectContext()):
            await agent.run()

        self.assertEqual(len(sent_messages), 1)

    async def test_manual_agent_receive_loop(self) -> None:
        state_json = (
            '{"type": "state", "score": 200, "lives": 3, "high_score": 500, '
            '"paddle_x": 100.0, "ball_x": 120.0, "ball_y": 300.0, '
            '"level": 1, "max_levels": 3, "bricks": []}'
        )

        class MockWebSocket:
            async def __aiter__(self):
                yield '{"type": "setup", "player_id": 10}'
                yield state_json
                yield '{"type": "state", "game_won": true, "game_over": true}'

            async def send(self, msg: str) -> None:
                pass

        stop_event = asyncio.Event()
        await receive_loop(MockWebSocket(), stop_event)
        self.assertTrue(stop_event.is_set())

        # Test send loop exits when stop_event is set
        await send_loop(MockWebSocket(), stop_event)


class TestAgentName(unittest.TestCase):
    def test_name_stored_and_validated(self) -> None:
        self.assertEqual(DummyAgent(name="  ann ").name, "ann")
        with self.assertRaises(ValueError):
            DummyAgent(name="  ")


CLOSED = object()


class FakeSocket:
    """WebSocket whose incoming messages are fed by the test, one at a time."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []

    async def send(self, msg: str) -> None:
        self.sent.append(json.loads(msg))

    def push(self, data: dict[str, Any]) -> None:
        self.incoming.put_nowait(json.dumps(data))

    def close(self) -> None:
        self.incoming.put_nowait(CLOSED)

    async def __aiter__(self) -> AsyncIterator[str]:
        while (m := await self.incoming.get()) is not CLOSED:
            yield m


class GatedAgent(BaseAgent):
    """Records every state it deliberates on; deliberate() blocks until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def deliberate(self) -> dict[str, Any] | None:
        assert self.current_state is not None
        self.seen.append(self.current_state["n"])
        self.started.set()
        await self.gate.wait()
        return {"action": "move", "direction": "WEST", "n": self.current_state["n"]}


def state(n: int, **extra: Any) -> dict[str, Any]:
    return {"type": "state", "n": n, "game_over": False, **extra}


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


class TestBaseAgentLatestState(unittest.IsolatedAsyncioTestCase):
    async def start(self, agent: BaseAgent) -> tuple[FakeSocket, asyncio.Task[None]]:
        sock = FakeSocket()

        class Connect:
            async def __aenter__(self) -> FakeSocket:
                return sock

            async def __aexit__(self, *args: object) -> None:
                return None

        patcher = mock.patch("websockets.connect", return_value=Connect())
        patcher.start()
        self.addCleanup(patcher.stop)
        out = contextlib.redirect_stdout(io.StringIO())
        out.__enter__()
        self.addCleanup(out.__exit__, None, None, None)
        task = asyncio.create_task(agent.run())
        self.addAsyncCleanup(self.stop, task)
        await settle()
        return sock, task

    async def stop(self, task: asyncio.Task[None]) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_fast_deliberation_acts_on_every_state(self) -> None:
        agent = GatedAgent()
        agent.gate.set()
        sock, task = await self.start(agent)
        for n in range(5):
            sock.push(state(n))
            await settle()
        sock.close()
        await asyncio.wait_for(task, 1)
        self.assertEqual(agent.seen, [0, 1, 2, 3, 4])
        self.assertEqual([a["n"] for a in sock.sent[1:]], [0, 1, 2, 3, 4])  # sent[0] is the handshake
        self.assertEqual(agent.stop_reason, "connection closed")

    async def test_slow_deliberation_skips_obsolete_states(self) -> None:
        agent = GatedAgent()
        sock, task = await self.start(agent)
        sock.push(state(0))
        await agent.started.wait()
        for n in range(1, 101):  # the server keeps streaming while deliberate() is busy
            sock.push(state(n))
        await settle()
        agent.gate.set()
        await settle()
        sock.close()
        await asyncio.wait_for(task, 1)
        self.assertEqual(agent.seen, [0, 100])  # the stale 1..99 are dropped, not queued
        self.assertEqual([a["n"] for a in sock.sent[1:]], [0, 100])

    async def test_terminal_state_is_not_lost_while_deliberating(self) -> None:
        for flag, reason in (("game_won", "game won"), ("game_over", "game over")):
            with self.subTest(flag):
                agent = GatedAgent()
                sock, task = await self.start(agent)
                sock.push(state(0))
                await agent.started.wait()
                sock.push(state(1))
                sock.push(state(2, **{flag: True}))
                await settle()
                agent.gate.set()
                await asyncio.wait_for(task, 1)
                self.assertEqual(agent.seen, [0])  # terminal state ends the agent, no further deliberation
                self.assertEqual(agent.stop_reason, reason)
                assert agent.current_state is not None
                self.assertEqual(agent.current_state["n"], 2)

    async def test_setup_is_handled_while_deliberating(self) -> None:
        agent = GatedAgent()
        sock, task = await self.start(agent)
        sock.push(state(0))
        await agent.started.wait()
        sock.push({"type": "setup", "player_id": 7})
        await settle()
        self.assertEqual(agent.player_id, 7)
        agent.gate.set()
        sock.close()
        await asyncio.wait_for(task, 1)

    async def test_dummy_agent_still_plays_through_run(self) -> None:
        agent = DummyAgent()
        sock, task = await self.start(agent)
        moves = [{"action": "move", "direction": "WEST"}, {"action": "move", "direction": "EAST"}]
        sock.push({"type": "setup", "player_id": 1})
        for _ in range(3):
            sock.push({"type": "state", "game_over": False, "valid_actions": moves})
            await settle()
        sock.push({"type": "state", "game_over": True, "timed_out": True, "valid_actions": moves})
        await asyncio.wait_for(task, 1)
        self.assertEqual(agent.player_id, 1)
        self.assertEqual(agent.stop_reason, "time limit reached")
        for action in sock.sent[1:]:
            self.assertIn(action, moves)


if __name__ == "__main__":
    unittest.main()
