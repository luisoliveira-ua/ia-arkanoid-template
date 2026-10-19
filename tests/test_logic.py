import math
import random
import unittest
from typing import Any

import numpy as np

from server.logic import Arkanoid, Brick


class TestLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Arkanoid()

    def test_brick_health_and_indestructibility(self) -> None:
        # Destructible brick with 2 health
        b = Brick(
            index=0,
            x=10,
            y=10,
            width=50,
            height=20,
            brick_type="blue",
            health=2,
            indestructible=False,
        )
        self.assertTrue(b.active)
        destroyed = b.hit()
        self.assertFalse(destroyed)
        self.assertEqual(b.health, 1)
        self.assertTrue(b.active)

        destroyed = b.hit()
        self.assertTrue(destroyed)
        self.assertEqual(b.health, 0)
        self.assertFalse(b.active)

        # Indestructible metal brick
        metal = Brick(
            index=1,
            x=70,
            y=10,
            width=50,
            height=20,
            brick_type="metal",
            health=-1,
            indestructible=True,
        )
        self.assertTrue(metal.active)
        destroyed = metal.hit()
        self.assertFalse(destroyed)
        self.assertTrue(metal.active)
        self.assertEqual(metal.health, -1)

    def test_paddle_movement_bounds(self) -> None:
        self.game.paddle_x = 10.0
        self.game.move_paddle("WEST")
        self.assertEqual(self.game.paddle_x, 0.0)
        self.game.move_paddle("WEST")
        self.assertEqual(self.game.paddle_x, 0.0)

        max_x = self.game.width - self.game.paddle_width
        self.game.paddle_x = max_x - 10.0
        self.game.move_paddle("EAST")
        self.assertEqual(self.game.paddle_x, max_x)
        self.game.move_paddle("EAST")
        self.assertEqual(self.game.paddle_x, max_x)

    def test_constant_ball_speed_and_reaching_bricks(self) -> None:
        # Ball moving upward at nominal speed must reach bricks without decelerating
        self.game.ball_x = 300.0
        self.game.ball_y = 700.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -self.game.ball_speed

        dt = 0.05
        for _ in range(25):  # 1.25s of flight
            self.game.update(dt)

        # Ball should travel upward past y=250 easily and reach brick height
        self.assertLess(self.game.ball_y, 250.0)
        current_speed = math.hypot(self.game.ball_vx, self.game.ball_vy)
        self.assertAlmostEqual(current_speed, self.game.ball_speed, places=1)

    def test_wall_bounces(self) -> None:
        r = self.game.ball_radius

        # Left wall
        self.game.ball_x = r + 1.0
        self.game.ball_y = 300.0
        self.game.ball_vx = -200.0
        self.game.ball_vy = 0.0
        self.game.update(0.02)
        self.assertGreater(self.game.ball_vx, 0.0)

        # Right wall
        self.game.ball_x = self.game.width - r - 1.0
        self.game.ball_y = 300.0
        self.game.ball_vx = 200.0
        self.game.ball_vy = 0.0
        self.game.update(0.02)
        self.assertLess(self.game.ball_vx, 0.0)

        # Virtual ceiling below header
        self.game.ball_x = 300.0
        self.game.ball_y = self.game.ceiling_y + r + 1.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -200.0
        self.game.update(0.02)
        self.assertGreater(self.game.ball_vy, 0.0)
        self.assertGreaterEqual(self.game.ball_y, self.game.ceiling_y)

    def test_nord_brick_color_and_hits_remaining(self) -> None:
        # Brick with 3 health starts in Aurora
        b = Brick(
            index=0,
            x=10,
            y=100,
            width=50,
            height=20,
            brick_type="green",
            health=3,
            indestructible=False,
        )
        self.assertEqual(b.hits_remaining, 3)
        self.assertEqual(b.palette, "aurora")
        self.assertEqual(b.dynamic_type, "aurora_yellow")
        self.assertEqual(b.color, "#EBCB8B")

        # After 1st hit -> 2 hits left (Aurora Green)
        b.hit()
        self.assertEqual(b.hits_remaining, 2)
        self.assertEqual(b.palette, "aurora")
        self.assertEqual(b.dynamic_type, "aurora_green")
        self.assertEqual(b.color, "#A3BE8C")

        # After 2nd hit -> 1 hit left (Frost Ice Blue)
        b.hit()
        self.assertEqual(b.hits_remaining, 1)
        self.assertEqual(b.palette, "frost")
        self.assertEqual(b.dynamic_type, "frost")
        self.assertEqual(b.color, "#88C0D0")

        # Verify to_dict shares this in world state
        d = b.to_dict()
        self.assertEqual(d["hits_remaining"], 1)
        self.assertEqual(d["palette"], "frost")
        self.assertEqual(d["type"], "frost")
        self.assertEqual(d["color"], "#88C0D0")

        # Metal brick in Snow Storm
        metal = Brick(
            index=1,
            x=70,
            y=100,
            width=50,
            height=20,
            brick_type="metal",
            health=-1,
            indestructible=True,
        )
        self.assertEqual(metal.hits_remaining, -1)
        self.assertEqual(metal.palette, "snow_storm")
        self.assertEqual(metal.dynamic_type, "metal")
        self.assertEqual(metal.color, "#D8DEE9")

    def test_paddle_3_regions_bounce(self) -> None:
        w = self.game.paddle_width
        left_thresh = w * self.game.config.paddle.left_ratio
        right_thresh = w * (1.0 - self.game.config.paddle.right_ratio)
        cfg = self.game.config.paddle.angles
        self.assertLess(cfg.alpha_deg, cfg.beta_deg)

        cases = [
            ((left_thresh + right_thresh) / 2.0, 0.0, cfg.alpha_deg),  # center: vertical +- alpha
            (left_thresh * 0.5, -cfg.side_deg, cfg.beta_deg),  # left: 45 deg left +- beta
            (right_thresh + 5.0, cfg.side_deg, cfg.beta_deg),  # right: 45 deg right +- beta
        ]
        for rel_x, centre, half in cases:
            angles = [math.degrees(self.game._calculate_paddle_bounce_angle(rel_x)) for _ in range(300)]
            self.assertGreaterEqual(min(angles), centre - half - 1e-9)
            self.assertLessEqual(max(angles), centre + half + 1e-9)
            # cone is actually used (not a fixed angle)
            self.assertGreater(max(angles) - min(angles), half)

    def test_played_games_keep_angles_within_cones(self) -> None:
        """Tracking bots (center and edge-aiming) never produce a paddle bounce beyond side + beta from vertical."""
        cfg = self.game.config.paddle.angles
        limit = cfg.side_deg + cfg.beta_deg
        for offset in (0.0, 0.4, -0.4):
            random.seed(7)
            game = Arkanoid(config=self.game.config, high_score_manager=self.game.high_score_manager)
            angles: list[float] = []
            original = game._handle_paddle_bounce

            def record(hit_x: float, g: Arkanoid = game, orig: Any = original, out: list[float] = angles) -> None:
                orig(hit_x)
                out.append(math.degrees(math.atan2(g.ball_vx, -g.ball_vy)))

            game._handle_paddle_bounce = record  # type: ignore[method-assign]
            for _ in range(4000):
                target = game.ball_x + offset * game.paddle_width
                centre = game.paddle_x + game.paddle_width / 2.0
                if target < centre - 10.0:
                    game.move_paddle("WEST")
                elif target > centre + 10.0:
                    game.move_paddle("EAST")
                game.update(1 / 30)
                if game.game_over or game.game_won:
                    break
            self.assertGreater(len(angles), 20)
            self.assertLessEqual(max(abs(a) for a in angles), limit + 1e-6)

    def test_only_broken_bricks_score(self) -> None:
        self.game.bricks = [
            Brick(0, 100.0, 100.0, 50.0, 20.0, "green", 2, False, 150),
            Brick(1, 200.0, 100.0, 50.0, 20.0, "metal", -1, True, 10),
            Brick(2, 300.0, 100.0, 50.0, 20.0, "red", 1, False, 50),
        ]
        self.game.brick_array = np.array(
            [
                [b.left, b.top, b.right, b.bottom, 1.0, b.index, float(b.indestructible), b.health]
                for b in self.game.bricks
            ],
            dtype=np.float32,
        )
        self.game.score = 0
        self.game._handle_brick_hit(self.game.bricks[0], 0.0, -1.0)  # damages, does not break
        self.game._handle_brick_hit(self.game.bricks[1], 0.0, -1.0)  # metal
        self.assertEqual(self.game.score, 0)
        self.game._handle_brick_hit(self.game.bricks[0], 0.0, -1.0)  # breaks
        self.assertEqual(self.game.score, 150)

    def test_brick_collision_and_destruction(self) -> None:
        # Create a target destructible brick and another brick to prevent level clear
        self.game.bricks = [
            Brick(
                index=0,
                x=280.0,
                y=200.0,
                width=50.0,
                height=20.0,
                brick_type="red",
                health=1,
                points=50,
            ),
            Brick(
                index=1,
                x=100.0,
                y=100.0,
                width=50.0,
                height=20.0,
                brick_type="red",
                health=1,
                points=50,
            ),
        ]
        self.game.brick_array = np.array(
            [
                [280.0, 200.0, 330.0, 220.0, 1.0, 0.0, 0.0, 1.0],
                [100.0, 100.0, 150.0, 120.0, 1.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Position ball moving upwards towards brick 0
        self.game.ball_x = 300.0
        self.game.ball_y = 227.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -200.0

        initial_score = self.game.score
        self.game.update(0.04)

        # Brick should be destroyed
        self.assertFalse(self.game.bricks[0].active)
        self.assertEqual(self.game.brick_array[0, 4], 0.0)
        self.assertGreater(self.game.score, initial_score)
        # Velocity should reflect downwards
        self.assertGreater(self.game.ball_vy, 0.0)

    def test_multi_hit_brick_does_not_break_in_single_update(self) -> None:
        # Green brick with 2 health
        brick = Brick(
            index=0,
            x=280.0,
            y=200.0,
            width=50.0,
            height=20.0,
            brick_type="green",
            health=2,
            points=80,
        )
        self.game.bricks = [brick]
        self.game.brick_array = np.array(
            [[280.0, 200.0, 330.0, 220.0, 1.0, 0.0, 0.0, 2.0]],
            dtype=np.float32,
        )

        # Position ball moving upwards towards bottom edge of brick (y=220)
        # Radius is 6.0. Place ball at y=223 moving up at 400 px/s
        self.game.ball_x = 305.0
        self.game.ball_y = 223.0
        self.game.ball_speed = 400.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -400.0

        # Run update with dt=0.033s (which causes 2-3 physics sub-steps internally)
        self.game.update(0.033)

        # In this single update frame, the brick should only take 1 hit, not break
        self.assertTrue(self.game.bricks[0].active)
        self.assertEqual(self.game.bricks[0].health, 1)
        self.assertEqual(self.game.bricks[0].hits_remaining, 1)
        self.assertEqual(self.game.bricks[0].palette, "frost")
        # Ball should now be moving downwards (reflected)
        self.assertGreater(self.game.ball_vy, 0.0)

    def test_level_progression_when_metal_remains(self) -> None:
        # 1 destructible brick and 1 metal brick
        self.game.bricks = [
            Brick(
                index=0,
                x=100.0,
                y=100.0,
                width=50.0,
                height=20.0,
                brick_type="red",
                health=1,
                indestructible=False,
            ),
            Brick(
                index=1,
                x=200.0,
                y=100.0,
                width=50.0,
                height=20.0,
                brick_type="metal",
                health=-1,
                indestructible=True,
            ),
        ]
        self.game.brick_array = np.array(
            [
                [100.0, 100.0, 150.0, 120.0, 1.0, 0.0, 0.0, 1.0],
                [200.0, 100.0, 250.0, 120.0, 1.0, 1.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        )

        initial_level = self.game.level_idx
        # Destroy the destructible brick
        self.game.bricks[0].active = False
        self.game.brick_array[0, 4] = 0.0

        # Run level check
        self.game._check_level_progression()

        # Should advance to next level even though metal brick was active
        self.assertEqual(self.game.level_idx, initial_level + 1)

    def test_get_state_structure(self) -> None:
        state = self.game.get_state()
        self.assertIn("width", state)
        self.assertIn("height", state)
        self.assertIn("paddle_x", state)
        self.assertIn("ball_x", state)
        self.assertIn("ball_speed", state)
        self.assertIn("gravity", state)
        self.assertIn("lives", state)
        self.assertIn("score", state)
        self.assertIn("timer", state)
        self.assertIn("time_elapsed", state)
        self.assertIn("normalized_score", state)
        self.assertIn("avg_norm_score", state)
        self.assertIn("high_score_normalized", state)
        self.assertIn("bricks", state)
        self.assertIn("actions", state)
        self.assertIn("valid_actions", state)
        self.assertIsInstance(state["actions"], list)

    def test_timer_and_normalized_score_updates(self) -> None:
        self.assertEqual(self.game.elapsed_time, 0.0)
        self.assertEqual(self.game.normalized_score, 0.0)

        # Simulate 10 frames of 0.1s dt
        for _ in range(10):
            self.game.update(0.1)

        self.assertAlmostEqual(self.game.elapsed_time, 1.0, places=2)
        # Give some score
        self.game.score = 500
        self.assertAlmostEqual(self.game.normalized_score, 500.0, places=1)

        # Resetting game resets timer and normalized score
        self.game.reset_game()
        self.assertEqual(self.game.elapsed_time, 0.0)
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.normalized_score, 0.0)

    def _clear_level(self, game: Arkanoid, elapsed: float, points: int) -> None:
        """Play a level fast-forwarded: `points` scored in `elapsed` seconds of its own clock, then every brick gone."""
        game.elapsed_time += elapsed
        game.level_elapsed = elapsed
        game.score += points
        game.bricks.clear()
        game.brick_array[:, 4] = 0.0
        game._check_level_progression()

    def test_avg_norm_score_counts_only_cleared_levels(self) -> None:
        """normalized_score is per level (its clock restarts); avg_norm_score is the sequence mean, uncleared = 0."""
        game = self.game
        game.level_sequence = game.level_sequence[:1] * 3
        game.reset_game()
        game.update(0.1)
        game.elapsed_time = game.level_elapsed = 10.0
        game.score = 100  # level 1 in progress: 100 points in 10 s
        self.assertAlmostEqual(game.normalized_score, 10.0)
        self.assertEqual(game.avg_norm_score, 0.0)  # not cleared yet: nothing earned

        self._clear_level(game, 10.0, 0)
        self.assertEqual(game.level_idx, 1)
        self.assertEqual(game.level_norms, [10.0])
        self.assertEqual(game.level_elapsed, 0.0)
        self.assertEqual(game.normalized_score, 0.0)  # level 2 just started
        self.assertAlmostEqual(game.avg_norm_score, 3.33)  # (10 + 0 + 0) / 3 levels

        game.elapsed_time, game.level_elapsed = 30.0, 20.0
        game.score = 100 + 100  # level 2: 100 points in 20 s
        self.assertAlmostEqual(game.normalized_score, 5.0)
        self.assertAlmostEqual(game.avg_norm_score, 3.33)
        self._clear_level(game, 20.0, 0)
        self.assertEqual(game.level_norms, [10.0, 5.0])
        self.assertAlmostEqual(game.avg_norm_score, 5.0)  # (10 + 5 + 0) / 3
        state = game.get_state()
        self.assertEqual(state["avg_norm_score"], 5.0)
        self.assertEqual(state["score"], 200)  # the raw high score is the sum over the levels

    def test_level_lost_earns_no_normalized_score(self) -> None:
        game = self.game
        game.level_sequence = game.level_sequence[:1] * 3
        game.reset_game()
        self._clear_level(game, 10.0, 100)  # level 1 cleared: 10 points/s
        game.elapsed_time, game.level_elapsed = 15.0, 5.0
        game.score += 400  # level 2: many points quickly, then all lives lost
        self.assertAlmostEqual(game.normalized_score, 80.0)
        game.lives = 1
        game._handle_bottom_miss()
        self.assertTrue(game.game_over)
        self.assertFalse(game.game_won)
        self.assertEqual(game.normalized_score, 0.0)  # the level ended uncleared
        self.assertAlmostEqual(game.avg_norm_score, 3.33)  # only level 1 counts: (10 + 0 + 0) / 3
        self.assertEqual(game.score, 500)  # the raw score keeps everything, it sorts the ranking

    def test_levels_label_compresses_runs(self) -> None:
        game = self.game
        for files, label in (
            (["level1.json", "level2.json", "level3.json"], "1-3"),
            (["level2.json"], "2"),
            (["level1.json", "level3.json"], "1,3"),
            (["level1.json", "level2.json", "level4.json"], "1-2,4"),
            (["level3.json", "level1.json", "level2.json"], "1-3"),  # order does not matter
            (["level1.json", "level1.json"], "1"),  # nor repeats
            (["custom.json"], "custom.json"),
        ):
            game.level_sequence = files
            self.assertEqual(game.levels_label(), label)

    def test_never_clearing_agent_has_zero_avg_norm_score(self) -> None:
        game = self.game
        game.reset_game()
        game.elapsed_time = game.level_elapsed = 5.0
        game.score = 2000
        game.lives = 1
        game._handle_bottom_miss()
        self.assertTrue(game.game_over)
        self.assertEqual(game.avg_norm_score, 0.0)
        self.assertEqual(game.normalized_score, 0.0)
        self.assertEqual(game.score, 2000)

    def test_won_sequence_averages_every_level(self) -> None:
        game = self.game
        game.level_sequence = game.level_sequence[:1] * 2
        game.reset_game()
        self._clear_level(game, 10.0, 100)  # 10 points/s
        self._clear_level(game, 20.0, 100)  # 5 points/s
        self.assertTrue(game.game_won)
        self.assertTrue(game.game_over)
        self.assertEqual(game.level_norms, [10.0, 5.0])
        self.assertAlmostEqual(game.avg_norm_score, 7.5)  # mean of the levels, not 200 / 30
        self.assertAlmostEqual(game.normalized_score, 5.0)  # last level, cleared

    def test_level_time_limit_ends_game_without_advancing(self) -> None:
        self.game.config.game.level_time_limit = 5.0
        self.game.reset_game()
        for _ in range(int(5.0 / 0.1) + 5):
            self.game.ball_y = 400.0  # keep the ball in the open field: no miss, no brick
            self.game.ball_x = 300.0
            self.game.ball_vx, self.game.ball_vy = 0.0, 0.0
            self.game.update(0.1)
        self.assertTrue(self.game.game_over)
        self.assertTrue(self.game.timed_out)
        self.assertFalse(self.game.game_won)
        self.assertEqual(self.game.level_idx, 0)
        state = self.game.get_state()
        self.assertTrue(state["timed_out"])
        self.assertEqual(state["level_time_limit"], 5.0)
        self.assertGreaterEqual(state["level_time_elapsed"], 5.0)
        self.assertLess(state["level_time_elapsed"], 5.2)

    def test_level_time_limit_resets_on_level_clear_and_can_be_disabled(self) -> None:
        self.game.config.game.level_time_limit = 5.0
        self.game.reset_game()
        self.game.level_elapsed = 4.0
        self.game.load_level(1)
        self.assertEqual(self.game.level_elapsed, 0.0)
        self.game.config.game.level_time_limit = 0.0
        self.game.level_elapsed = 10_000.0
        self.game.update(0.1)
        self.assertFalse(self.game.timed_out)


if __name__ == "__main__":
    unittest.main()
