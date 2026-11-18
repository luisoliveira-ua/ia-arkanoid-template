import os
import unittest

import aigf.interface as interface

from server.server import ArkanoidGameServer, build_parser, sanitize_name


class TestServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = ArkanoidGameServer()

    async def test_player_lifecycle(self) -> None:
        self.assertEqual(self.server.state, interface.GameState.LOBBY)
        await self.server.on_player_connect(1)
        self.assertEqual(self.server.player_id, 1)
        self.assertEqual(self.server.state, interface.GameState.RUNNING)

        # Extra player connection should not override active player
        await self.server.on_player_connect(2)
        self.assertEqual(self.server.player_id, 1)

        # Disconnect player
        await self.server.on_player_disconnect(1)
        self.assertIsNone(self.server.player_id)
        self.assertEqual(self.server.state, interface.GameState.LOBBY)

        # New player connects after disconnect
        await self.server.on_player_connect(3)
        self.assertEqual(self.server.player_id, 3)
        self.assertEqual(self.server.state, interface.GameState.RUNNING)

    async def test_game_over_transitions_to_lobby_and_awaits_new_player(self) -> None:
        await self.server.on_player_connect(1)
        self.assertEqual(self.server.state, interface.GameState.RUNNING)

        # Simulate game over
        self.server.game.game_over = True
        await self.server.tick(0.02)
        # Server transitions to LOBBY
        self.assertEqual(self.server.state, interface.GameState.LOBBY)

        # Agent disconnects upon receiving game over
        await self.server.on_player_disconnect(1)
        self.assertIsNone(self.server.player_id)
        self.assertEqual(self.server.state, interface.GameState.LOBBY)
        self.assertFalse(self.server.game.game_over)

        # New agent connects and starts playing
        await self.server.on_player_connect(2)
        self.assertEqual(self.server.player_id, 2)
        self.assertEqual(self.server.state, interface.GameState.RUNNING)
        self.assertFalse(self.server.game.game_over)

    async def test_on_start_sim_clears_game_over(self) -> None:
        self.server.game.game_over = True
        await self.server.on_start_sim()
        self.assertEqual(self.server.state, interface.GameState.RUNNING)
        self.assertFalse(self.server.game.game_over)

    async def test_process_action(self) -> None:
        await self.server.on_player_connect(1)
        initial_x = self.server.game.paddle_x

        # Move West
        await self.server.process_action(
            1, {"action": "move", "direction": "WEST"}
        )
        self.assertLess(self.server.game.paddle_x, initial_x)

        # Move East
        await self.server.process_action(
            1, {"action": "move", "direction": "EAST"}
        )
        self.assertAlmostEqual(self.server.game.paddle_x, initial_x)

        # Action from non-active player should be ignored
        await self.server.process_action(
            99, {"action": "move", "direction": "WEST"}
        )
        self.assertAlmostEqual(self.server.game.paddle_x, initial_x)

    async def test_tick_and_state(self) -> None:
        await self.server.on_player_connect(1)
        initial_y = self.server.game.ball_y
        await self.server.tick(0.02)
        state = self.server.get_state()
        self.assertEqual(state["player_id"], 1)
        self.assertNotEqual(state["ball_y"], initial_y)

    async def test_setup_payload_and_reset(self) -> None:
        payload = self.server.get_setup_payload()
        self.assertIn("width", payload)
        self.assertIn("height", payload)
        self.assertIn("paddle_width", payload)

        await self.server.on_player_connect(1)
        await self.server.on_reset_sim()
        self.assertEqual(self.server.state, interface.GameState.LOBBY)


class TestServerOptions(unittest.IsolatedAsyncioTestCase):
    async def test_acceleration_scales_framework_fps_not_sim_step(self) -> None:
        server = ArkanoidGameServer(fps=30, acceleration=4.0)
        self.assertEqual(server.fps, 120)
        self.assertAlmostEqual(server.sim_dt, 1 / 30)

        await server.on_player_connect(1)
        await server.tick(1 / 120)
        self.assertAlmostEqual(server.game.elapsed_time, 1 / 30, places=6)

    async def test_levels_override(self) -> None:
        server = ArkanoidGameServer(levels=["level2.json"])
        self.assertEqual(server.game.level_sequence, ["level2.json"])
        self.assertEqual(server.get_setup_payload()["max_stages"], 1)

    async def test_share_highscore_default_is_memory_only(self) -> None:
        self.assertIsNone(ArkanoidGameServer().game.high_score_manager.filepath)
        shared = ArkanoidGameServer(share_highscore=True)
        self.assertEqual(shared.game.high_score_manager.filepath, shared.app_config.game.highscores_file)
        if os.path.exists("highscores.csv"):
            os.remove("highscores.csv")

    async def test_invalid_acceleration(self) -> None:
        with self.assertRaises(ValueError):
            ArkanoidGameServer(acceleration=0)

    async def test_level_time_limit_override(self) -> None:
        self.assertEqual(ArkanoidGameServer().game.config.game.level_time_limit, 400.0)
        self.assertEqual(ArkanoidGameServer(level_time_limit=120.0).game.config.game.level_time_limit, 120.0)
        with self.assertRaises(ValueError):
            ArkanoidGameServer(level_time_limit=-1)

    def test_unloadable_level_map_is_a_start_up_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            ArkanoidGameServer(levels=["missing_level.json"])
        self.assertIn("missing_level.json", str(ctx.exception))

    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual((args.fps, args.acceleration, args.levels, args.share_highscore), (30, 1.0, None, False))
        self.assertIsNone(args.level_time_limit)
        self.assertIsNone(args.seed)
        args = build_parser().parse_args(["--levels", "1-3", "--share-highscore", "--fps", "60", "--seed", "42"])
        self.assertEqual((args.fps, args.levels, args.share_highscore, args.seed), (60, "1-3", True, 42))

    def test_parser_short_flags(self) -> None:
        args = build_parser().parse_args(
            ["-l", "2", "-S", "-f", "45", "-s", "99", "-a", "2.0", "-t", "300", "-p", "9000"]
        )
        self.assertEqual(args.levels, "2")
        self.assertTrue(args.share_highscore)
        self.assertEqual(args.fps, 45)
        self.assertEqual(args.seed, 99)
        self.assertEqual(args.acceleration, 2.0)
        self.assertEqual(args.level_time_limit, 300.0)
        self.assertEqual(args.port, 9000)


class TestSanitizeName(unittest.TestCase):
    def test_sanitize(self) -> None:
        self.assertEqual(sanitize_name("  ann  "), "ann")
        self.assertEqual(sanitize_name("a\x00b\n\x1b[31mc"), "ab[31mc")
        self.assertEqual(sanitize_name("x" * 100), "x" * 32)
        self.assertEqual(sanitize_name("\x00\x01"), "PLAYER")
        self.assertEqual(sanitize_name("Ünï"), "n")



if __name__ == "__main__":
    unittest.main()
