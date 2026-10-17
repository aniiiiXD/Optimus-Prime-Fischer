# Opponent-bias diagnostic candidate

This isolated copy of `onevone_baseline` tests two observed decision gaps:

- a bounded, deterministic search for a guaranteed win before ending a turn or
  buying an uncertain development card at 8 or 9 VP;
- exact Monopoly scoring from the opponent's public resource hand, instead of
  a production-based estimate.

The search never resolves hidden dice, development draws, or robber theft. It
uses `obs.cpu_left` to leave clock headroom. The official adapter remains
`agent.py`; all imports are inside this folder and the pinned kit.

See [the measured experiment](../OPPONENT_BIAS_RESULT.md). This is a diagnostic
copy derived from pre-event private work, **not a competition submission**.
