import os
import unittest

from server.config import load_config, load_map_file, parse_levels


class TestConfig(unittest.TestCase):
    def test_load_default_config(self) -> None:
        cfg = load_config("non_existent_config.json")
        self.assertEqual(cfg.screen.width, 600)
        self.assertEqual(cfg.screen.height, 800)
        self.assertEqual(cfg.paddle.width, 84.0)
        self.assertAlmostEqual(cfg.ball.speed, 400.0)

    def test_load_real_config(self) -> None:
        cfg = load_config("config.json")
        self.assertEqual(cfg.screen.width, 600)
        self.assertEqual(cfg.screen.height, 800)
        self.assertIn("metal", cfg.bricks.types)
        self.assertTrue(cfg.bricks.types["metal"].indestructible)
        self.assertEqual(cfg.bricks.types["metal"].points, 0)
        self.assertEqual(cfg.ball.speed, 400.0)

    def test_consts_module(self) -> None:
        import consts
        self.assertEqual(consts.SCREEN_WIDTH, 600)
        self.assertEqual(consts.SCREEN_HEIGHT, 800)
        self.assertEqual(consts.BRICK_TYPES["metal"]["points"], 0)
        self.assertTrue(consts.BRICK_TYPES["metal"]["indestructible"])

    def test_load_level_maps(self) -> None:
        for lvl in ["maps/level1.json", "maps/level2.json", "maps/level3.json"]:
            self.assertTrue(os.path.exists(lvl), f"Level map missing: {lvl}")
            map_data = load_map_file(lvl)
            self.assertIn("bricks", map_data)
            self.assertGreater(len(map_data["bricks"]), 0)
            # Verify each brick has required fields
            for brick in map_data["bricks"]:
                self.assertIn("x", brick)
                self.assertIn("y", brick)
                self.assertIn("type", brick)
                self.assertIn("health", brick)
                self.assertIn("indestructible", brick)

    def test_parse_levels(self) -> None:
        self.assertEqual(parse_levels("1"), ["level1.json"])
        self.assertEqual(parse_levels(2), ["level2.json"])
        self.assertEqual(parse_levels("1,3"), ["level1.json", "level3.json"])
        self.assertEqual(parse_levels("1-3"), ["level1.json", "level2.json", "level3.json"])
        self.assertEqual(parse_levels("3,1-2"), ["level3.json", "level1.json", "level2.json"])
        self.assertEqual(parse_levels([1, 2]), ["level1.json", "level2.json"])
        self.assertEqual(parse_levels("level2"), ["level2.json"])

    def test_parse_levels_errors(self) -> None:
        for bad in ("", "9", "3-1", "1,99"):
            with self.assertRaises(ValueError):
                parse_levels(bad)


    def test_invalid_paddle_angles_rejected(self) -> None:
        import json
        import tempfile

        for angles in ({"alpha_deg": 20, "beta_deg": 10}, {"side_deg": 80, "beta_deg": 15}):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"paddle": {"angles": angles}}, f)
            try:
                with self.assertRaises(ValueError):
                    load_config(f.name)
            finally:
                os.remove(f.name)


    def test_load_simplified_grid_map(self) -> None:
        import json
        import tempfile
        content = {
            "name": "Grid Level",
            "grid": {"margin_x": 35.0, "margin_y": 120.0, "gap_x": 10.0, "gap_y": 5.0},
            "bricks": [
                {"row": 0, "col": 0, "type": "red"},
                {"row": 1, "col": 1, "health": 2},
                {"row": 2, "col": 3, "health": -1},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(content, f)
        try:
            data = load_map_file(f.name)
            self.assertEqual(len(data["bricks"]), 3)
            # Red brick: row 0, col 0
            self.assertEqual(data["bricks"][0]["x"], 35.0)
            self.assertEqual(data["bricks"][0]["y"], 120.0)
            self.assertEqual(data["bricks"][0]["health"], 1)
            self.assertFalse(data["bricks"][0]["indestructible"])
            # Health 2 -> blue brick
            self.assertEqual(data["bricks"][1]["type"], "blue")
            self.assertEqual(data["bricks"][1]["x"], 35.0 + 60.0)
            self.assertEqual(data["bricks"][1]["y"], 120.0 + 25.0)
            # Health -1 -> metal brick
            self.assertEqual(data["bricks"][2]["type"], "metal")
            self.assertTrue(data["bricks"][2]["indestructible"])
            self.assertEqual(data["bricks"][2]["points"], 0)
        finally:
            os.remove(f.name)


if __name__ == "__main__":
    unittest.main()
