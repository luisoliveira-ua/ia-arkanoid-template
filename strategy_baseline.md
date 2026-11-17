# Student strategy baseline

This records the best measured strategy configuration to date. The active code remains in `agents/strategic_agent.py` and `arkanoid_domain.py`.

## Best measured configuration

- Angle samples per paddle zone: `16`
- Safety reserve: `2` ticks
- Minimum robust rebound margin: `2.0`
- Scoring-loop threshold: `0.99`
- Upper-channel threshold: `0.99`
- Upper-channel rate slack: `0.03`
- Early top-route setup window: `30` seconds
- Learned-rate prior weight: `0.0`
- Tree lookahead: disabled (`LOOKAHEAD_LEVELS = frozenset()`)
- Tunnel gaps: calculated from the brick layout at the start of each simulated shot

## Benchmark reference

`evaluation_final_logic.json`: 20 seeded games per level, seeds `20261116`–`20261135`, using the server's logic engine.

| Level | Mean normalized score | Clear rate | Mean lives lost |
|---|---:|---:|---:|
| 1 | 18.17 points/s | 20/20 | 0 |
| 2 | 12.59 points/s | 20/20 | 0 |
| 3 | 9.17 points/s | 20/20 | 0 |

## Next candidate (not validated)

Keep the baseline values and use a level-specific upper-channel threshold:

```python
UPPER_CHANNEL_SEEK_PROBABILITY_BY_LEVEL = {
    1: 0.99,
    2: 15 / 16,
    3: 15 / 16,
}
```

With 16 deterministic samples, `0.99` requires all 16 samples to enter the channel. `15/16` would allow one sample to miss on levels 2 and 3, where the measured rates are below level 1, while retaining the existing 3% rate guard. This is a sample threshold, not a calibrated probability guarantee. The comparison needs to be rerun after disk space is available; the earlier attempt could not save its report.
