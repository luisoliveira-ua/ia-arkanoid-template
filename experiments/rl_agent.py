"""Online tabular Q-learning agent for Arkanoid rebound decisions.

The agent learns from observed score, elapsed time, and lost lives. Its Q table
is saved between runs; until a state has enough experience, it uses the proven
strategic controller as a safe fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.strategic_agent import StrategicAgent
from experiments.arkanoid_domain import ArkanoidSearchDomain, ArkanoidState


DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data
DEFAULT_Q_PATH = DATA_DIR / "rl_q_values.json"


@dataclass(frozen=True)
class _Experience:
    state: str
    action: str
    score: int
    elapsed: float
    lives: int
    level: int


class ReinforcementAgent(StrategicAgent):
    """Learn a tabular rebound policy online with epsilon-greedy Q-learning."""

    ACTIONS = (
        ArkanoidSearchDomain.LEFT,
        ArkanoidSearchDomain.CENTER,
        ArkanoidSearchDomain.RIGHT,
    )
    ALPHA = 0.2
    GAMMA = 0.85
    DEFAULT_EPSILON = 0.05
    MIN_ACTION_VISITS = 3
    SAVE_EVERY_UPDATES = 16
    LIFE_LOSS_PENALTY = 150.0
    REFERENCE_RATE_BY_LEVEL = {1: 18.17, 2: 12.59, 3: 9.17}

    def __init__(
        self,
        *args: Any,
        epsilon: float = DEFAULT_EPSILON,
        q_path: str | Path | None = None,
        learn: bool = True,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.learning_enabled = learn
        self.q_path = Path(q_path) if q_path is not None else DEFAULT_Q_PATH
        self._q_values, self._visits = self._load_q_table(self.q_path)
        self._pending: tuple[str, str, tuple[str, ...]] | None = None
        self._active_experience: _Experience | None = None
        self._updates_since_save = 0

    @staticmethod
    def _load_q_table(
        path: Path,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, {}
        if not isinstance(raw, dict) or raw.get("version") != 1:
            return {}, {}

        raw_values = raw.get("q_values", {})
        raw_visits = raw.get("visits", {})
        if not isinstance(raw_values, dict):
            raw_values = {}
        if not isinstance(raw_visits, dict):
            raw_visits = {}

        values: dict[str, dict[str, float]] = {}
        visits: dict[str, dict[str, int]] = {}
        for state, actions in raw_values.items():
            if isinstance(actions, dict):
                state_values: dict[str, float] = {}
                for action, value in actions.items():
                    try:
                        if str(action) in ReinforcementAgent.ACTIONS:
                            state_values[str(action)] = float(value)
                    except (TypeError, ValueError):
                        continue
                values[str(state)] = state_values
        for state, actions in raw_visits.items():
            if isinstance(actions, dict):
                state_visits: dict[str, int] = {}
                for action, count in actions.items():
                    try:
                        if str(action) in ReinforcementAgent.ACTIONS:
                            state_visits[str(action)] = max(0, int(count))
                    except (TypeError, ValueError):
                        continue
                visits[str(state)] = state_visits
        return values, visits

    @staticmethod
    def _score_and_time(observation: dict[str, Any]) -> tuple[int, float, int]:
        score = int(observation.get("score", observation.get("level_score", 0)))
        elapsed = float(
            observation.get("time_elapsed", observation.get("level_time_elapsed", 0.0))
        )
        lives = int(observation.get("lives", 3))
        return score, elapsed, lives

    def _state_key(self, domain: ArkanoidSearchDomain, state: ArkanoidState) -> str:
        landing_x, _ = domain.predict_landing(state)
        landing_bucket = min(2, max(0, int(3.0 * landing_x / domain.width)))
        horizontal = state.ball_vx / max(1e-6, domain.ball_speed)
        direction = "L" if horizontal < -0.2 else "R" if horizontal > 0.2 else "C"
        density = self._brick_bucket(state.remaining_bricks)
        return f"{domain.level_number}|{density}|{landing_bucket}|{direction}"

    def _choose_target(
        self,
        domain: ArkanoidSearchDomain,
        state: ArkanoidState,
        observation: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        if state.ball_vy <= 0:
            if (
                observation is not None
                and self._was_descending
                and state.ball_y
                >= domain.paddle_y - 4.0 * domain.ball_speed / domain.fps
            ):
                self._observe_paddle_contact(domain, observation)
            return super()._choose_target(domain, state, observation)

        baseline_action, baseline_target = super()._choose_target(
            domain, state, observation
        )
        landing_x, inbound_seconds = domain.predict_landing(state)
        safe_targets = {
            action: self._zone_target(domain, state, landing_x, inbound_seconds, action)
            for action in domain.ACTIONS
        }
        safe_actions = [
            action for action in domain.ACTIONS if safe_targets[action] is not None
        ]
        if not safe_actions:
            self._pending = None
            return baseline_action, baseline_target

        state_key = self._state_key(domain, state)
        visits = self._visits.get(state_key, {})
        if self._rng.random() < self.epsilon:
            selected_action = self._rng.choice(safe_actions)
        elif (
            safe_actions
            and all(
                visits.get(action, 0) >= self.MIN_ACTION_VISITS
                for action in safe_actions
            )
        ):
            values = self._q_values.get(state_key, {})
            highest = max(values.get(action, 0.0) for action in safe_actions)
            tied = [
                action
                for action in safe_actions
                if values.get(action, 0.0) >= highest - 1e-9
            ]
            selected_action = baseline_action if baseline_action in tied else tied[0]
        else:
            selected_action = (
                baseline_action if baseline_action in safe_actions else safe_actions[0]
            )

        target_choice = safe_targets[selected_action]
        if target_choice is None:
            self._pending = None
            return baseline_action, baseline_target
        self._pending = (state_key, selected_action, tuple(safe_actions))
        return selected_action, target_choice[0]

    def _observe_paddle_contact(
        self,
        domain: ArkanoidSearchDomain,
        observation: dict[str, Any],
    ) -> None:
        if not self.learning_enabled:
            self._active_experience = None
            self._pending = None
            return

        score, elapsed, lives = self._score_and_time(observation)
        game_finished = bool(
            observation.get("game_over") or observation.get("game_won")
        )
        pending = self._pending
        previous = self._active_experience
        life_lost = previous is not None and lives < previous.lives
        if previous is not None:
            self._update_q(
                previous,
                observation,
                next_state=(
                    pending[0]
                    if pending is not None and not life_lost and not game_finished
                    else None
                ),
                next_actions=(
                    pending[2]
                    if pending is not None and not life_lost and not game_finished
                    else None
                ),
                terminal=game_finished or life_lost,
            )

        if life_lost:
            # The selected paddle zone missed; the serve after losing a life
            # does not execute that rebound action.
            self._active_experience = None
        elif pending is not None and not game_finished:
            self._active_experience = _Experience(
                state=pending[0],
                action=pending[1],
                score=score,
                elapsed=elapsed,
                lives=lives,
                level=domain.level_number,
            )
        else:
            self._active_experience = None
        self._pending = None

    def _update_q(
        self,
        experience: _Experience,
        observation: dict[str, Any],
        *,
        next_state: str | None,
        next_actions: tuple[str, ...] | None = None,
        terminal: bool,
    ) -> None:
        score, elapsed, lives = self._score_and_time(observation)
        points_earned = max(0, score - experience.score)
        seconds = max(0.05, elapsed - experience.elapsed)
        reference_rate = self.REFERENCE_RATE_BY_LEVEL.get(experience.level, 12.0)
        life_loss = max(0, experience.lives - lives)
        reward = (
            points_earned
            - reference_rate * seconds
            - self.LIFE_LOSS_PENALTY * life_loss
        )

        if bool(observation.get("timed_out")):
            terminal = True
        next_value = 0.0
        if next_state is not None and not terminal:
            available_actions = next_actions or self.ACTIONS
            next_value = max(
                self._q_values.get(next_state, {}).get(action, 0.0)
                for action in available_actions
            )
        target = reward + (self.GAMMA * next_value if not terminal else 0.0)
        state_values = self._q_values.setdefault(experience.state, {})
        old_value = state_values.get(experience.action, 0.0)
        state_values[experience.action] = old_value + self.ALPHA * (target - old_value)
        state_visits = self._visits.setdefault(experience.state, {})
        state_visits[experience.action] = state_visits.get(experience.action, 0) + 1
        self._updates_since_save += 1
        if self._updates_since_save >= self.SAVE_EVERY_UPDATES:
            self._save_q_table()

    def _save_q_table(self) -> None:
        payload = {
            "version": 1,
            "algorithm": "tabular-q-learning",
            "alpha": self.ALPHA,
            "gamma": self.GAMMA,
            "q_values": self._q_values,
            "visits": self._visits,
        }
        temporary_path = self.q_path.with_suffix(self.q_path.suffix + ".tmp")
        try:
            self.q_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            temporary_path.replace(self.q_path)
        except OSError as error:
            logging.warning("Could not save RL Q table to %s: %s", self.q_path, error)
        else:
            self._updates_since_save = 0

    async def deliberate(self) -> dict[str, Any] | None:
        observation = self.current_state
        if (
            observation
            and not observation.get("game_over")
            and not observation.get("game_won")
            and self._active_experience is not None
        ):
            domain, _ = ArkanoidSearchDomain.from_observation(observation)
            if self._active_experience.level != domain.level_number:
                # A level transition resets the ball without another paddle
                # contact, so close the old rebound before the new free serve.
                self._update_q(
                    self._active_experience,
                    observation,
                    next_state=None,
                    terminal=True,
                )
                self._active_experience = None
                self._pending = None
        return await super().deliberate()

    def report(self) -> None:
        if not self.learning_enabled:
            super().report()
            return
        observation = self.current_state
        if observation and (
            observation.get("game_over") or observation.get("game_won")
        ):
            if self._active_experience is not None:
                self._update_q(
                    self._active_experience,
                    observation,
                    next_state=None,
                    terminal=True,
                )
                self._active_experience = None
        self._save_q_table()
        super().report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid online Q-learning agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Arkanoid RL Agent")
    parser.add_argument("--id", default=None)
    parser.add_argument(
        "--epsilon", type=float, default=ReinforcementAgent.DEFAULT_EPSILON
    )
    parser.add_argument("--q-table", type=Path, default=DEFAULT_Q_PATH)
    args = parser.parse_args()
    asyncio.run(
        ReinforcementAgent(
            args.server,
            args.name,
            args.id,
            epsilon=args.epsilon,
            q_path=args.q_table,
        ).run()
    )


if __name__ == "__main__":
    main()
