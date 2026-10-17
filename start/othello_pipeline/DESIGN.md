# Othello Transformer: experiment design, not a strength claim

## What we are building

```text
Othello rules + position sampler
            │
            ├── opening diversity / hard positions / pass + endgame cases
            ▼
Neural Reversi teacher (fixed weights, level, threads, hash, time)
            │  scores for every legal move; independent Edax is held out
            ▼
versioned JSONL + canonical-state cache + group-isolated split
            ▼
64 square tokens + [STATE] ── 2 bidirectional attention blocks
            │
            ├── shared candidate scorer → policy logits, Q(s,a), legality
            └── state scorer            → V(s)
            ▼
legal mask → direct move (later: fixed-budget search)
            ▼
teacher regret + paired games + whole-move CPU profile
```

This student is **not OthelloGPT**. OthelloGPT reads move history and predicts a plausible next move. We show the current board explicitly and train to prefer teacher-valued legal actions. The neural teacher itself is not a Transformer; its evaluator plus search supplies targets.

## Perspectives that must stay separate

| Perspective | Question | Initial choice | Failure to watch |
|---|---|---|---|
| Game | Is every action/flip/pass correct? | Reuse `gamebench` adapter; compare against teacher GTP | coordinate/color mismatch |
| Openings | Are early states diverse? | randomized openings, then balanced phase sampling | one book line memorized |
| Teacher | Are scores trustworthy? | fixed version and search budget, record all legal moves | unstable or perspective-flipped scores |
| Data | Is the holdout independent? | canonical D4 board key plus opening-group split | symmetric/trajectory leakage |
| Student | Can attention learn spatial patterns? | relative board tokens, row/column embeddings, shared candidate head | perfect training agreement but weak play |
| Training | What changed the weights? | policy + Q + optional outcome value + legality losses, ablated | multitask loss hurting policy |
| Playing | Can it win? | direct policy first; later optional search | hiding a weak network behind strong search |
| CPU | Is it fast *per useful decision*? | full move latency, batch throughput, thread sweep | optimizing GEMM while teacher/search dominates |

## Input and output contract

- Board: 64 row-major cells, each encoded `empty=0, own=1, opponent=2` relative to side-to-move; also supply absolute side (black=0, white=1). A learned `[STATE]` token is appended, so attention sees **65 tokens**.
- Candidates: square IDs `0..63`, pass `64`; padding `65` is never a move. In training we score all 65 candidates to learn legality; in inference we can score only the legal list. A hard engine mask remains mandatory.
- Outputs in one forward pass: policy logit and Q value per candidate, raw legality logit per candidate, scalar state value. Attention is bidirectional over current-board squares; there is **no causal/history mask**.
- Starter model in `model.py`: width 64, 4 heads (16 dimensions/head), 2 blocks, MLP width 128, FP32. Its actual parameter count and latency must be measured, never inferred from a paper's much larger model.

For board tokens `X∈R^{65×d}`, each head computes `softmax(QKᵀ/√d_h)V`; the shared candidate head scores `concat(contextual_square, state_summary, action_embedding)`. A pass uses the state summary and its own action embedding, not a fake square. This is intentionally smaller than the corpus's 4-block, width-128 multi-game proposal; we can ablate width **after** this baseline is sound.

The implementation has **110,212 parameters**. A rough FP32 forward count is about 11 million arithmetic FLOPs with nine legal candidates: per block, projection FLOPs scale as `8Td²`, attention matmuls as `4T²d`, and MLP as `4Td·d_ff`, with `T=65`, `d=64`, `d_ff=128`, two blocks. This excludes norms, activations, and runtime overhead. It is a shape estimate, not a hardware-counter reading; the tiny batch-1 workload can be dispatch-limited even when a textbook roofline would suggest compute headroom.

## Teacher-labeled dataset

One JSONL row per **nonterminal** position:

```json
{"board":[-1,0,1],"player":0,"legal":[19,26],"teacher_scores":[3.5,1.0],"result":1,"group":"opening-abc","teacher_id":"neural-reversi-v6.3.0-e6bbc4f6-L12"}
```

The `board` example is abbreviated; real rows have 64 cells. `teacher_scores` align with `legal` and must be from the *current player's* perspective. `result` is optional final outcome from that perspective (`-1,0,+1`). `group` identifies an opening family/game lineage. Preserve raw teacher scores, search level, time, engine/weight hashes, and any timeout in provenance; never silently treat an incomplete query as an exact oracle. Check the engine score sign empirically on known endgames before training.

Sampling: include early/middle/late phases, corners and near-corners, low/high mobility, forced pass, tactical mistakes, and losing as well as winning positions. Use stochastic rollouts or teacher top-k perturbations so the student learns to recover off the main line. Deduplicate positions under the eight dihedral board symmetries; transform actions/scores along with boards. Cache labels by canonical board + side + exact teacher settings. Do **not** count symmetric copies as independent holdout examples. Early shared root positions may need removal from validation even with opening-group splits.

An opening book can seed diversity, but none is downloaded in this workspace yet; do not confuse the OthelloGPT theory notebook with a move book. The teacher's `datagen/opening.rs` is a generation tool, not a local pretrained opening book.

## Training objective and first ablations

Given teacher move scores `q_T(a)`, set a soft policy target `p_T(a)=softmax(q_T(a)/τ)` over **legal** moves. Start with `τ=2` only as a tunable baseline; inspect score scale first. Q targets start as `tanh(q_T/16)`; this scale is provisional and must be checked by game phase. An outcome result, if available, supervises `V`.

`L = 1·CE(p_T, πθ) + 0.5·Huber(Qθ,q_T) + 0.25·Huber(Vθ,z) + 0.1·BCE(legalθ,legal)`.

The legality BCE needs a positive-class weight because only roughly nine of 65 actions are legal in the pilot. We tested weight 1 versus 8. Weight 8 improved raw legal recall but did **not** improve teacher regret; the hard rules mask remains required. The pilot has no outcome labels, so `V` received zero supervised gradient and must not be presented as trained.

The policy is always hard-masked by real rules before softmax; legality BCE measures learned rule awareness, not permission to make illegal moves. Run separate policy-only, +Q, +value, +symmetry, +hard-position replay trials; change one factor at a time. FP32 AdamW, learning rate `3e-4`, weight decay `0.01`, batch `64`, fixed seeds, early stopping on group-held-out regret are starting knobs, not results. Save each checkpoint plus full config, data fingerprint, epoch metrics, and teacher provenance.

## Evidence gate: what would count as good?

1. Rules parity: opening, random midgame, forced pass, and endgame states agree with the teacher engine. One failed parity test blocks labeling.
2. Offline holdout: teacher top-1/top-3 agreement, **score regret** (`max_a q_T(a)-q_T(a_model)`), corner/tactical/phase breakdown, Q error, value calibration, raw legality recall. Use distinct opening groups and canonical board keys.
3. Online: paired games with colors swapped versus random, greedy, minimax, and independent Edax at pinned budgets. Report W/D/L and uncertainty; direct student and student+search are separate entries.
4. Systems: p50/p95 whole-move latency, positions/s at batch 1/32/128, RSS, CPU threads, cache hits, teacher-label cost, and strength per second. A small forward time alone is not a winning result.

## CPU plan from our measured Connect Four lessons

- Do not assume more threads are faster. The Connect Four batch-1 model measured 0.296 ms with 1 thread versus 0.389 ms with 4; batch-128 favored 4 threads. Othello must get its own sweep on the same Mac.
- Reuse PyTorch's vectorized matrix kernels first. Our NEON demo proves SIMD exists, **not** that a custom kernel will beat PyTorch on these small shapes. Apple AMX/SME dispatch cannot be inferred from a feature flag or timing alone.
- Profile rules, board encoding, forward, masking, optional search, and output separately. Othello move generation/flip scanning may become the hot path; only then compare Python arrays with bitboards/native code under exact move-parity tests.
- Batch teacher positions and training examples; keep one long-lived teacher process and a persistent position cache. Worker/process counts should be benchmarked because short queries can lose to startup and IPC (as the Connect Four Pons pilot did).
- Quantization, `torch.compile`, ONNX, and custom kernels are **later controlled experiments**. Promote only if legal choices and held-out regret remain acceptable, not solely on latency.

## What is ready versus missing

Ready locally: rules adapter, OBF/GTP conversion, pinned compiled teacher and checksum-verified weights, opening/pass/exact-score parity, cached label generator, student architecture, group-aware trainer, 600-position pilot dataset, two pilot checkpoints, baseline tournaments, CPU forward profiler.

Missing before a strength claim: broad/high-quality teacher positions, independent opening-family holdout beyond one small pilot, outcome-value targets, stronger-search teacher ablations, paired diverse opening games, and an independent Edax tournament. The current student weights are a **plumbing pilot**, not a competition candidate.

## Knowledge sources used

- `../game_transformer_knowledge_corpus/notes/03_cpu_first_architecture.md` — board/candidate interface and size constraints.
- `../game_transformer_knowledge_corpus/notes/04_experiments_and_metrics.md` — loss choices, regret and paired-match gates.
- `../game_transformer_knowledge_corpus/notes/06_mav_takeaway.md` — policy/Q/world-model distinction and multi-action targets.
- `../connect_four_transformer/CPU_OPTIMIZATION_REPORT.md` — measured thread, batching, search-hotpath, and holdout lessons on this machine.
- `../transformer_games_lab/SYSTEMS_LAB.md` — SIMD/NEON/AMX terminology and honest kernel profiling.
