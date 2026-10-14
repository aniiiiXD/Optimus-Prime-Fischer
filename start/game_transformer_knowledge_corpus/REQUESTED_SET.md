# Exact requested set

This is the focused collection requested on 2026-10-13: Laya, eight board-game Transformer repositories, and the Multi-Action-Value Transformer paper. The listed files are stored on the original machine, but `repos/` and `papers/` are excluded from the Hyde Git repository; use the [fetch instructions](README.md#rebuild-or-search-the-corpus) after cloning.

## Decision-model reference

| Project | Local copy | Pinned source revision |
|---|---|---|
| Laya | [`repos/laya`](repos/laya) | `c73908d96ff73c0062c340a11bf7c54ec7d8bc4d` |

## Board-game repositories

| Project | Local copy | Pinned source revision |
|---|---|---|
| Othello World | [`repos/othello_world`](repos/othello_world) | `f23bb5696cf30b93bd8af8a391ee33fc3aac417e` |
| Othello Mamba | [`repos/othello_mamba`](repos/othello_mamba) | `29cf393c4d11ea93cc1a9bb14057c74a469d1096` |
| ChessTransformer | [`repos/ChessTransformer`](repos/ChessTransformer) | `0398779d8a8ec88d4c787a68c94b0e2778bd29e2` |
| Chess Transformers | [`repos/chess-transformers`](repos/chess-transformers) | `efe16dbf163c0c515a39afd79564e6449396d2eb` |
| Chessformer | [`repos/chessformer`](repos/chessformer) | `e24db0a5df8fc67ff996b34074ac59f56c6aaf86` |
| Chessformers | [`repos/chessformers`](repos/chessformers) | `818159477f7ef34d3d919dc474564e6cfecc7a47` |
| Connect 4 | [`repos/connect-4`](repos/connect-4) | `e468aae3eed212eb20e8c54b2d2277fd9a07fa8e` |
| Emergent World Representations — Othello | [`repos/Emergent-World-Representations-Othello`](repos/Emergent-World-Representations-Othello) | `e52217b4b756d22579b28f24c7b1b2355c8f8914` |

## Research paper

**Mastering Board Games by External and Internal Planning with Language Models**

Multi-Action-Value Transformer; Chess, Chess960, Connect Four, and Hex.

- Optional local PDF after fetching: `papers/01_mastering_board_games_external_internal_planning.pdf`
- Written takeaway kept in Git: [`notes/06_mav_takeaway.md`](notes/06_mav_takeaway.md)
- Primary source: <https://arxiv.org/abs/2412.12119>
- Verified: PDF 1.5, 70 pages, 1,396,698 bytes

## Important download boundary

The revisions above record the original local download, but the source trees are not included in Hyde Git. The downloader follows current branches, so a new run may get different commits. Files hosted separately by repository authors—multi-gigabyte datasets, Hugging Face/Google Drive checkpoints, and Git LFS payloads—are not automatically equivalent to the GitHub source repository. The locally present and externally available weights are itemized in [`docs/checkpoint_catalog.md`](docs/checkpoint_catalog.md).
