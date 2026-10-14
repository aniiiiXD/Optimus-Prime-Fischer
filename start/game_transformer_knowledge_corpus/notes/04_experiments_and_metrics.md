# Experiments and metrics

## Training strategies and their equations

| Strategy | One-line mathematical definition | What it answers |
|---|---|---|
| next-move imitation | minimize (L_{BC}=-\log\pi_\theta(a_{expert}\mid s)) | can the model copy the data policy? |
| legal-move multitask | add binary cross-entropy over every action's legality | has it learned the rules? |
| value learning | minimize (L_V=(V_\theta(s)-r)^2) | can it predict the eventual result? |
| outcome weighting | use (L=-w(r)\log\pi_\theta(a\mid s)) | do winner moves deserve more influence? |
| advantage weighting | use (L=-\exp(A/\beta)\log\pi_\theta(a\mid s)) with clipped weights | can better-than-baseline moves dominate without online RL? |
| minimax distillation | target (a^*=\arg\max_a[-V_T(T(s,a))]) | can a shallow teacher transfer lookahead? |
| MCTS distillation | minimize cross-entropy to normalized visit counts (N(s,a)^{1/\tau}) | can search produce richer policy targets? |
| symmetry augmentation | train on all valid transforms (g(s),g(a)) | can known invariances reduce data needs? |
| hard-example replay | sample errors with probability proportional to loss or regret | can training focus on discovered weaknesses? |
| entropy bonus | subtract (\lambda H(\pi_\theta(\cdot\mid s))) | can early collapse be prevented? |

## Strength metrics

- **Win/draw/loss:** primary outcome against fixed baselines from both starting sides.
- **Elo:** summarize pairwise results, with uncertainty and opponent pool stated.
- **Illegal-move rate:** must be zero after engine masking; raw unmasked rate diagnoses rule learning.
- **Top-1/top-3 agreement:** compare policy to teacher moves without pretending all alternatives are equally bad.
- **Policy regret:** (Q(s,a^*)-Q(s,\hat a)), more informative than exact-match when several moves are good.
- **Value error/calibration:** MSE plus reliability bins between predicted result and observed result.
- **Exploit test:** replay a catalog of known tactical traps and opening biases.

## CPU metrics

- median, p90, and p99 latency for batch 1;
- positions or candidate actions per second for batches and search leaves;
- peak and steady-state RSS;
- estimated FLOPs and achieved GFLOP/s;
- model bytes read and arithmetic intensity (I=\text{FLOPs}/\text{bytes});
- thread scaling efficiency;
- cache hit rate and time saved;
- playing strength per second and per joule if power counters are available.

## Minimum trustworthy comparison

1. Freeze dataset split, opponent versions, opening set, seeds, time/search budget, and hardware configuration.
2. Change one factor.
3. Run both colors and enough games to report a confidence interval.
4. Check training loss, held-out policy/value metrics, and tournament strength.
5. Profile the full move pipeline, not only a matrix multiplication.
6. Keep a golden suite proving optimized and reference engines choose the same legal moves within the allowed tolerance.

## First ablation table to produce

| Run | Data target | Value head | Search | Params | p50 ms | RSS MB | Win rate vs heuristic | Win rate vs minimax |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A | random legal move | no | no | | | | | |
| B | minimax move | no | no | | | | | |
| C | minimax move | yes | no | | | | | |
| D | minimax move | yes | shallow alpha-beta | | | | | |

This table tells us whether better targets, a value head, or inference-time search buys the largest strength increase per CPU cost.
