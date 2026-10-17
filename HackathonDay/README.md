# Hackathon Day — Catan agent lab

Open [`index.html`](index.html) for the live report. It is a standalone HTML
file with no server, package install, external font or network dependency.

`report.json` is the editable source. It currently holds the common 1v1 grid
of 20 playable profiles and 31 documented experiments, including unsuccessful
attempts. Every result has an evidence link. The two-player and four-player
arenas stay separate because their win rates are not interchangeable.

## Update after a run

1. Save the raw game log and record the rules, opponent, board seeds, both
   seats, checkpoint/configuration, wins, caps and errors.
2. Add one object to `experiments` in `report.json`. Give it a unique `id`,
   `track` (`1v1` or `4-player`), `kind`, `status`, `title`, `metric`,
   `comparison`, `finding`, `decision` and relative `source` path. Include a
   failed or inconclusive result as readily as a win.
3. Update the current-agent verdict only after a fresh paired test supports
   promotion. Change `as_of` when the report is revised.
4. Rebuild and verify:

   ```sh
   python3 HackathonDay/render.py
   python3 HackathonDay/render.py --check
   ```

The renderer escapes report text and embeds everything into `index.html`, so
opening the file directly works. The HTML links to local raw evidence; those
links resolve while the workspace and the two experiment worktrees remain in
their current locations.

## Reading the numbers

- The 1,028-board profile table uses dataset board seeds from both seats. It
  is a broad diagnostic against the pinned Catanatron web bot.
- Fresh confirmation, held-out, mixed-bot, native-engine and direct
  head-to-head results are stated with their own denominators and controls.
- Teacher-action accuracy measures imitation. Promotion is based on legal,
  completed game wins on new boards.
