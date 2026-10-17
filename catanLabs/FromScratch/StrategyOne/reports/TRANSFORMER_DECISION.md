# Transformer decision — defer for now

## Decision

Do **not** train a Transformer yet. Keep the compact legal-action MLP as the current policy baseline and fix its measured failures first.

## Evidence

The MLP's main failure is already visible in the current state. During 50 MLP-vs-Value-Base games, it chose `END_TURN` while the stronger bot chose `BUY_DEVELOPMENT_CARD` 435 times. In the 20% dataset check, it correctly selected `BUY_DEVELOPMENT_CARD` in 0 of 173 teacher-labelled cases.

This is an action-ranking and data-coverage problem, not a demonstrated long-history problem. The dataset has only 1,758 development-card-purchase labels out of 156,471 decisions (1.12%). A larger sequence model trained on the same labels would likely reproduce this bias, cost more inference compute, and make the failure harder to inspect.

## Where a Transformer could earn an edge later

- Process full action history to model opponent habits and trade behaviour.
- Learn interactions among board locations, roads, ports, production, and the VP race.
- Act as a win-value critic: estimate which resulting state is more likely to win several turns ahead.
- Supply scores to short search rather than replace the fast policy ranker.

## Gate before revisiting

Revisit a small Transformer only after the MLP has a measured repair for development-card conversion, territory targets, and robber targets, and we have a win-based/value target plus substantially more diverse self-play or external-game data. It must improve held-out **win rate**, not just teacher-action accuracy.
