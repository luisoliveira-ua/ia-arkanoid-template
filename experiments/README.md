# Experiments

Everything here was tried while building the submission agent and is kept for the record. The
submission (`student.py`, `agents/expectimax_agent.py`, `agents/fast_physics.py`) imports nothing
from this folder. Run the scripts as modules from the repository root, for example
`python -m experiments.tune_expectimax`.

Measured results of each idea against the expectimax agent are in the main README
("Things that did not help").

| File | What it is |
|---|---|
| `rollout_agent.py` | Expectimax plus paired rollouts of whole rebound sequences |
| `deep_agent.py` | Expectimax plus an exact second rebound under every root outcome |
| `value_agent.py`, `value_model.py` | Learned time-to-clear network (numpy MLP) used at the root |
| `tail_agent.py`, `tail_weights.py` | Hits weighted by how often each brick is left for the end |
| `channel_agent.py` | Hits on the outermost columns weighted up (level 1 ceiling loop) |
| `train_value.py` | Fitted policy iteration for the value network |
| `train_expectimax_value.py` | Value network trained on expectimax games with TD(lambda) targets |
| `tune_expectimax.py` | Paired coordinate search over the expectimax parameters |
| `strategic_agent.py`, `search_agent.py`, `rl_agent.py` | Earlier agents (sampled strategic planning, tree search, tabular Q-learning) |
| `arkanoid_domain.py`, `tree.py` | Search domain and tree search used by the earlier agents |
| `agente2.py` | Early agent prototype |
| `evaluate_student.py`, `sim_arkanoid.py` | Evaluation harnesses of the earlier agents |
| `train_agent.py`, `learn_policy.py`, `train_rl_agent.py` | Training of the earlier agents |
| `strategy_baseline.md` | Best configuration of the earlier strategic agent |
| `results/` | Evaluation reports of the earlier agents |
| `training_data/` | Models, tuning logs, recorded games and rollout datasets |
