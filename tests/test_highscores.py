"""Tests for the CSV high score manager and Arkanoid integration."""

import csv
import os
import unittest

from server.config import AppConfig
from server.highscores import DEFAULT_SCORES, HighScoreManager
from server.logic import Arkanoid


class TestHighScores(unittest.TestCase):
    def setUp(self) -> None:
        self.test_file = "test_highscores.csv"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self) -> None:
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_memory_only_mode_writes_nothing(self) -> None:
        manager = HighScoreManager(filepath=None)
        self.assertTrue(manager.add_score(50000, player="MEM"))
        self.assertEqual(manager.get_highest_score(), 50000)
        self.assertFalse(os.path.exists(self.test_file))

    def test_defaults_without_file_and_csv_log(self) -> None:
        manager = HighScoreManager(filepath=self.test_file)
        self.assertFalse(os.path.exists(self.test_file))
        self.assertEqual(len(manager.scores), 10)
        self.assertEqual(manager.get_highest_score(), DEFAULT_SCORES[0].score)

        manager.add_score(500, player="LOW", agent_id="a1", time_elapsed=10.0)  # not top 10, still logged
        manager.add_score(15000, player="ann, \"the\" bot", agent_id="a2", time_elapsed=100.0, avg_norm_score=150.0)
        with open(self.test_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["id"] for r in rows], ["a1", "a2"])
        self.assertEqual(rows[1]["player"], 'ann, "the" bot')
        self.assertEqual(rows[1]["score"], "15000")
        self.assertEqual(rows[1]["avg_norm_score"], "150.0")

    def test_add_score_top_10(self) -> None:
        manager = HighScoreManager(filepath=self.test_file)

        # Score lower than rank 10 should not enter
        added = manager.add_score(500, player="LOW")
        self.assertFalse(added)
        self.assertEqual(len(manager.scores), 10)

        # Score qualifying for rank 1
        added = manager.add_score(15000, player="TOP_PLAYER")
        self.assertTrue(added)
        self.assertEqual(len(manager.scores), 10)
        self.assertEqual(manager.get_highest_score(), 15000)
        self.assertEqual(manager.scores[0].player, "TOP_PLAYER")
        self.assertEqual(manager.scores[0].score, 15000)

        # Score of 0 or negative should be rejected
        self.assertFalse(manager.add_score(0))
        self.assertFalse(manager.add_score(-100))

        # Re-load in a separate manager instance to verify persistence
        manager2 = HighScoreManager(filepath=self.test_file)
        self.assertEqual(manager2.get_highest_score(), 15000)
        self.assertEqual(manager2.scores[0].player, "TOP_PLAYER")

    def test_corrupt_rows_skipped(self) -> None:
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("date,id,player,score,time_elapsed,avg_norm_score\n")
            f.write("2026-01-01,a1,GOOD,20000,10,2000\n")
            f.write("2026-01-01,a2,BAD,not-a-number,10,5\n")
            f.write("only,two\n")

        manager = HighScoreManager(filepath=self.test_file)
        self.assertEqual(len(manager.scores), 10)
        self.assertEqual(manager.scores[0].player, "GOOD")
        self.assertEqual(manager.scores[0].agent_id, "a1")
        self.assertNotIn("BAD", [e.player for e in manager.scores])

    def test_binary_garbage_file_ignored(self) -> None:
        with open(self.test_file, "wb") as f:
            f.write(b"\xff\xfe\x00garbage")
        self.assertEqual(HighScoreManager(filepath=self.test_file).get_highest_score(), DEFAULT_SCORES[0].score)

    def test_arkanoid_logic_integration(self) -> None:
        manager = HighScoreManager(filepath=self.test_file)
        cfg = AppConfig()
        game = Arkanoid(config=cfg, high_score_manager=manager)
        game.player_name = "AI_AGENT"

        self.assertEqual(game.high_score, manager.get_highest_score())

        # Simulate game play and scoring
        game.score = 25000
        game.lives = 1
        # Ball lost
        game._handle_bottom_miss()
        self.assertTrue(game.game_over)
        self.assertEqual(game.high_score, 25000)
        self.assertEqual(manager.get_highest_score(), 25000)
        self.assertEqual(manager.scores[0].player, "AI_AGENT")

        # Check get_state includes leaderboard
        state = game.get_state()
        self.assertEqual(state["high_score"], 25000)
        self.assertIn("high_scores", state)
        self.assertEqual(len(state["high_scores"]), 10)
        self.assertEqual(state["high_scores"][0]["score"], 25000)
        self.assertIn("high_scores_normalized", state)
        self.assertEqual(len(state["high_scores_normalized"]), 10)
        self.assertIn("timer", state)
        self.assertIn("avg_norm_score", state)

    def test_uncleared_game_has_no_avg_norm_score(self) -> None:
        """No avg_norm_score (no level cleared): no rate from score / time; raw board yes, normalized board no."""
        manager = HighScoreManager(filepath=None)
        self.assertTrue(manager.add_score(20000, player="BURST", time_elapsed=5.0))  # 4000 as score / time
        self.assertEqual(manager.get_highest_score(), 20000)
        self.assertEqual(manager.get_highest_normalized_score(), 100.0)  # still the arcade default
        self.assertNotIn("BURST", [e.player for e in manager.normalized_scores])

    def test_dual_leaderboard_and_normalized_ranking(self) -> None:
        manager = HighScoreManager(filepath=self.test_file)

        # FAST player: moderate raw score (3000) in very fast time (10s) -> 300 pts/sec
        added_fast = manager.add_score(3000, player="SPEEDRUNNER", time_elapsed=10.0, avg_norm_score=300.0)
        self.assertTrue(added_fast)
        self.assertEqual(manager.get_highest_normalized_score(), 300.0)
        # Fast player doesn't top raw score leaderboard (Taito default has 10000)
        self.assertEqual(manager.get_highest_score(), 10000)

        # SLOW player: massive raw score (50000) over long time (2000s) -> 25 pts/sec
        added_slow = manager.add_score(50000, player="GRINDER", time_elapsed=2000.0, avg_norm_score=25.0)
        self.assertTrue(added_slow)
        # Grinder tops raw leaderboard
        self.assertEqual(manager.get_highest_score(), 50000)
        self.assertEqual(manager.raw_scores[0].player, "GRINDER")
        # Speedrunner still tops normalized leaderboard
        self.assertEqual(manager.get_highest_normalized_score(), 300.0)
        self.assertEqual(manager.normalized_scores[0].player, "SPEEDRUNNER")

        # Verify persistence across new manager instance
        manager2 = HighScoreManager(filepath=self.test_file)
        self.assertEqual(manager2.get_highest_score(), 50000)
        self.assertEqual(manager2.get_highest_normalized_score(), 300.0)
        self.assertEqual(manager2.raw_scores[0].player, "GRINDER")
        self.assertEqual(manager2.normalized_scores[0].player, "SPEEDRUNNER")


if __name__ == "__main__":
    unittest.main()
