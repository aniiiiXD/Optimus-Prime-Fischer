# Frozen v3 policy on its own visited states

The source Transformer checkpoint was scored with the legal-action mask against fair-teacher labels on **60,000 unique decisions from 98 completed Transformer self-play games** (board seeds `20271600`–`20271697`). The comparison column uses only the **3,060 rows from ten audited validation games** in the original v3 corpus. The sealed final games were not read or scored. These are action-agreement diagnostics, not game-strength or action-regret measurements.

| Phase | Self-play rows | Self-play choice agreement | Original validation rows | Validation choice agreement |
|---|---:|---:|---:|---:|
| Initial settlement | 784 | 261/784 (33.3%) | 80 | 10/80 (12.5%) |
| Initial road | 784 | 439/784 (56.0%) | 80 | 41/80 (51.2%) |
| Move robber | 2,768 | 281/2,768 (10.2%) | 165 | 30/165 (18.2%) |
| Main turn | 42,284 | 8,968/19,970 (44.9%) | 2,275 | 551/938 (58.7%) |
| Discard | 13,380 | 4,604/12,005 (38.4%) | 460 | 210/460 (45.7%) |
| **All phases** | **60,000** | **14,553/36,311 (40.1%)** | **3,060** | **842/1,723 (48.9%)** |

Overall agreement including forced actions is 38,242/60,000 (63.7%) on self-play states versus 2,179/3,060 (71.2%) on original validation. The self-play corpus has 23,689 forced rows (39.5%); the validation set has 1,337 (43.7%). These forced choices inflate the overall percentages. The 8.8 percentage-point choice-agreement gap is descriptive: the two corpora have different position and phase distributions, and the original validation set was used for checkpoint selection.

## Where choices diverge

- **First action of each self-play game:** the model matches the teacher on 24/98 opening settlement placements (24.5%). Across all eight opening settlements per game it matches 261/784 (33.3%). Example, game 0 row 0: teacher places on node 16, model on node 8; 54 placements were legal. This records a preference difference, not which node is better.
- **Robber:** 2,487/2,768 choices disagree. In 2,183 disagreements the target tile differs; in 304 the tile matches but the victim differs. Example, game 0 row 37: teacher chooses tile 14 and relative victim 3, model chooses tile 8 and relative victim 1; 21 robber actions were legal.
- **Main turn:** 11,002/19,970 nonforced choices disagree. Of these, 5,959 retain the same action type but choose a different target or trade, while 5,043 change action type. The largest type pairs are teacher maritime trade → model end turn (3,281), teacher end turn → model maritime trade (1,390), and both maritime trade with different terms (3,472). Example, game 0 row 38: teacher ends the turn; model trades four sheep for brick. This points to a trade/turn-termination disagreement to investigate, not a proven strategic error.

## Scope and provenance

The frozen checkpoint is `gate3_v3_choice/best.pt` (`0c792198d7624a723193b42b429f99dff53c891010c6c4661c1b37b3d5f8be85`). The self-play dataset SHA-256 is `ea6d717b8eb085b369870707e9a670b27d3c3f8ac41adf38bc851dd7e227e517`; the original v3 dataset is `5e1939e12f36353700dc4b94180e225eb0418153215df2b786d6bbf412661bea`; and its split file is `fab3be26faaff4f470ce437f6a25bbc67ed4984c272925cb57079599ae423126`. Self-play and original collection board seeds do not overlap. The JSON companion contains exact counts, phase summaries, action-type pairs, and examples.

The model's choices generate the self-play distribution, so this is a useful distribution-shift screen. Teacher agreement alone cannot establish that the teacher's preferred action wins more games. Decisions within each game are correlated, and 98 games do not provide independent action-level samples for a tight uncertainty claim. No retraining or tournament was run for this diagnostic.
