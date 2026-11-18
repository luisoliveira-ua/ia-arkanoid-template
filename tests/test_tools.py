"""Unit tests for level designer tool and map converters."""

import json
import os
import tempfile
import unittest

from server.config import load_map_file
from tools.level_designer import (
    ascii_to_map,
    build_parser,
    convert_map_to_grid,
    map_to_ascii,
    validate_map_winnable,
)


class TestLevelDesigner(unittest.TestCase):
    def test_ascii_to_map_and_back(self) -> None:
        ascii_lines = [
            "R.B.G.Y.M",
            "1.2.3.4.X",
        ]
        map_data = ascii_to_map("Test Map", ascii_lines)
        self.assertEqual(map_data["name"], "Test Map")
        self.assertEqual(len(map_data["bricks"]), 10)
        self.assertEqual(map_data["bricks"][0], {"row": 0, "col": 0, "type": "red"})
        self.assertEqual(map_data["bricks"][1], {"row": 0, "col": 2, "type": "blue"})
        self.assertEqual(map_data["bricks"][4], {"row": 0, "col": 8, "type": "metal"})
        self.assertEqual(map_data["bricks"][5], {"row": 1, "col": 0, "type": "red"})

        # Map back to ASCII
        rendered = map_to_ascii(map_data)
        lines = rendered.splitlines()
        self.assertEqual(lines[0], "R.B.G.Y.M")
        self.assertEqual(lines[1], "R.B.G.Y.M")

    def test_convert_legacy_map_to_grid(self) -> None:
        legacy_data = {
            "name": "Legacy",
            "brick_width": 50.0,
            "brick_height": 20.0,
            "bricks": [
                {"x": 35.0, "y": 120.0, "type": "red", "health": 1, "indestructible": False},
                {"x": 155.0, "y": 145.0, "type": "blue", "health": 2, "indestructible": False},
            ],
        }
        grid_data = convert_map_to_grid(legacy_data)
        self.assertEqual(grid_data["name"], "Legacy")
        self.assertEqual(len(grid_data["bricks"]), 2)
        self.assertEqual(grid_data["bricks"][0], {"row": 0, "col": 0, "type": "red"})
        self.assertEqual(grid_data["bricks"][1], {"row": 1, "col": 2, "type": "blue"})

    def test_convert_keeps_official_map_positions(self) -> None:
        """Converting an official map to grid form and loading it back leaves every brick where it was."""
        for name in ("level1.json", "level2.json", "level3.json"):
            original = load_map_file(f"maps/{name}")
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, name)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(convert_map_to_grid(original), f)
                converted = load_map_file(path)
            before = sorted((b["x"], b["y"], b["type"]) for b in original["bricks"])
            after = sorted((b["x"], b["y"], b["type"]) for b in converted["bricks"])
            self.assertEqual(before, after, name)

    def test_convert_rejects_off_grid_map(self) -> None:
        """A brick that does not sit on the grid is an error, not a brick moved to the nearest cell."""
        off_grid = {"bricks": [{"x": 35.0, "y": 120.0, "type": "red"}, {"x": 40.0, "y": 145.0, "type": "red"}]}
        with self.assertRaises(ValueError):
            convert_map_to_grid(off_grid)

    def test_validate_map_winnable(self) -> None:
        unreachable = validate_map_winnable("maps/level1.json")
        self.assertEqual(unreachable, [])

    def test_parser_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-o", "out.json", "-i", "in.json", "-c", "-v", "-a", "art.txt", "-g"])
        self.assertEqual(args.output, "out.json")
        self.assertEqual(args.input, "in.json")
        self.assertTrue(args.convert)
        self.assertTrue(args.validate)
        self.assertEqual(args.ascii, "art.txt")
        self.assertTrue(args.gui)


if __name__ == "__main__":
    unittest.main()
