"""Tests for the level designer: official maps, CLI and the headless editor."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import pygame

from server.config import load_map_file
from tools import level_designer as ld


def _run_main(*argv: str) -> tuple[int, str]:
    """Run `ld.main()` with argv; return (exit code, stdout)."""
    out = io.StringIO()
    code = 0
    with mock.patch.object(sys, "argv", ["level_designer", *argv]), contextlib.redirect_stdout(out):
        try:
            ld.main()
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, out.getvalue()


class TestOfficialMaps(unittest.TestCase):
    def test_convert_keeps_every_brick_position(self) -> None:
        """Converting an official map must not move or drop any brick."""
        for n in (1, 2, 3):
            with open(f"maps/level{n}.json", encoding="utf-8") as f:
                src = json.load(f)
            conv = ld.convert_map_to_grid(src)
            g = conv["grid"]
            cells = {(b["row"], b["col"]) for b in conv["bricks"]}
            self.assertEqual(len(cells), len(src["bricks"]))
            px, py = src["brick_width"] + g["gap_x"], src["brick_height"] + g["gap_y"]
            back = {(round(g["margin_x"] + c * px, 3), round(g["margin_y"] + r * py, 3)) for r, c in cells}
            self.assertEqual(back, {(round(b["x"], 3), round(b["y"], 3)) for b in src["bricks"]})

    def test_ascii_view_shows_every_brick(self) -> None:
        """The editor text view shows all bricks of the official maps (none dropped or merged)."""
        for n in (1, 2, 3):
            with open(f"maps/level{n}.json", encoding="utf-8") as f:
                src = json.load(f)
            art = ld.map_to_ascii(src)
            self.assertEqual(sum(ch not in ".\n" for ch in art), len(src["bricks"]))

    def test_infer_grid_default_pitch_and_explicit_keys(self) -> None:
        """Aligned bricks keep the default pitch; explicit grid keys win; empty maps use defaults."""
        aligned = {"bricks": [{"x": 35.0, "y": 120.0}, {"x": 155.0, "y": 145.0}]}
        geo = ld.infer_grid(aligned)
        self.assertEqual((geo["gap_x"], geo["gap_y"]), (ld.GRID_GAP_X, ld.GRID_GAP_Y))
        explicit = {"grid": {"margin_y": 50.0}, "bricks": [{"x": 35.0, "y": 120.0}]}
        self.assertEqual(ld.infer_grid(explicit)["margin_y"], 50.0)
        self.assertEqual(ld.infer_grid({})["margin_y"], ld.GRID_MARGIN_Y)

    def test_infer_grid_irregular_pitch(self) -> None:
        """Rows spaced 30 px (level 3) give gap_y 10; the origin is the top-left brick."""
        geo = ld.infer_grid(load_map_file("maps/level3.json") | {"grid": {}})
        self.assertEqual((geo["margin_y"], geo["gap_y"]), (90.0, 10.0))


class TestValidate(unittest.TestCase):
    def test_official_maps_are_winnable(self) -> None:
        for n in (1, 2, 3):
            self.assertEqual(ld.validate_map_winnable(f"maps/level{n}.json"), [])

    def test_walled_off_brick_is_reported(self) -> None:
        """A destructible brick sealed inside a metal ring cannot be reached."""
        bricks = [{"x": 275.0, "y": 300.0, "type": "red", "indestructible": False}]
        for dx in (-60.0, 0.0, 60.0):
            for dy in (-25.0, 25.0):
                bricks.append({"x": 275.0 + dx, "y": 300.0 + dy, "type": "metal", "indestructible": True})
        for dx in (-60.0, 60.0):
            bricks.append({"x": 275.0 + dx, "y": 300.0, "type": "metal", "indestructible": True})
        self.assertEqual(ld.validate_map_winnable({"bricks": bricks}), [(275.0, 300.0)])


class TestCli(unittest.TestCase):
    def test_validate_ok_and_failure(self) -> None:
        code, out = _run_main("-v", "-i", "maps/level1.json")
        self.assertEqual(code, 0)
        self.assertIn("SUCCESS", out)
        with mock.patch.object(ld, "validate_map_winnable", return_value=[(1.0, 2.0)]):
            code, out = _run_main("-v", "-i", "maps/level1.json")
        self.assertEqual(code, 1)
        self.assertIn("FAILED", out)

    def test_ascii_and_convert_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            art = os.path.join(tmp, "a.txt")
            with open(art, "w", encoding="utf-8") as f:
                f.write("R.B\n.G.\n")
            out_a = os.path.join(tmp, "a.json")
            _run_main("-a", art, "-o", out_a)
            with open(out_a, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)["bricks"]), 3)
            out_c = os.path.join(tmp, "c.json")
            _, out = _run_main("-c", "-i", "maps/level3.json", "-o", out_c)
            self.assertIn("Converted", out)
            with open(out_c, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)["bricks"]), len(load_map_file("maps/level3.json")["bricks"]))

    def test_no_action_prints_help(self) -> None:
        with mock.patch.object(ld, "run_gui") as gui:
            _, out = _run_main("-i", "maps/level1.json")
            gui.assert_not_called()
            self.assertIn("usage", out)
            _run_main("-g", "-i", "maps/level1.json", "-o", "x.json")
            gui.assert_called_once_with(output_path="x.json", initial_map="maps/level1.json")


class TestGui(unittest.TestCase):
    """Drive the pygame editor headless with scripted events."""

    def _play(self, keys: list[int], initial: str | None, out_path: str, mouse: tuple[int, int] | None = None) -> None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        batches: list[list[pygame.event.Event]] = []
        for k in keys:
            batches.append([pygame.event.Event(pygame.KEYDOWN, key=k)])
        if mouse:
            batches.insert(1, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)])
            batches.insert(2, [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1)])
            batches.insert(3, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3)])
        batches.append([pygame.event.Event(pygame.QUIT)])
        pos = mouse or (0, 0)
        with (
            mock.patch("pygame.event.get", side_effect=batches),
            mock.patch("pygame.mouse.get_pos", return_value=pos),
        ):
            ld.run_gui(output_path=out_path, initial_map=initial)

    def test_open_official_map_and_save_keeps_all_bricks(self) -> None:
        """Opening L3, saving and reloading must keep all 35 bricks (top row included)."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "sub", "l3.json")
            self._play([pygame.K_r, pygame.K_s], "maps/level3.json", out)
            with open(out, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(len(saved["bricks"]), len(load_map_file("maps/level3.json")["bricks"]))
            self.assertEqual(saved["grid"]["margin_y"], 90.0)

    def test_tools_paint_and_clear(self) -> None:
        """All tool keys are accepted; painting then erasing under the pointer leaves the cell empty."""
        keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_m, pygame.K_0, pygame.K_1, pygame.K_r]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "new.json")
            self._play(keys + [pygame.K_s], None, out, mouse=(40, 125))
            with open(out, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["bricks"], [])


if __name__ == "__main__":
    unittest.main()
