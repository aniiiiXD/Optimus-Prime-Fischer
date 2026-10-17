# Dataset plan

Status: ready to implement; generator and encoder are not built yet.

1. **Configure the game.** Start with provisional two-player, 10-VP Catanatron
   games. Record the engine commit and rules version. Confirm trading and other
   rule differences against the official competition specification.
2. **Prepare the teacher.** Use `AlphaBetaPlayer` with `depth=1`,
   `value_fn="base"`, default weights, `prunning=False`, and `epsilon=None`.
   Before collecting labels, prevent access to actual hidden cards and future
   randomness, fix unsupported trading decisions, and isolate search from the
   live game's random generator. Depth one is a shallow teacher whose strength
   still needs measurement.
3. **Generate varied games.** Allocate equal numbers of board-and-seat pairs to
   teacher versus teacher, `ValueFunctionPlayer` (`base`, no exploration),
   `WeightedRandomPlayer`, and `RandomPlayer`. Swap seats on each board. Save all
   decisions; initially use only teacher decisions as imitation targets.
4. **Store one decision per JSONL line.** Capture the observation immediately
   before the move; attach its result after the game ends. Use this structure:

   ```text
   schema_version, game_id, decision_id, player, bot_version, rules_version
   observation:
     board: terrain, actual dice numbers, adjacency, ports, roads, buildings, robber
     own_hand, own_dev_cards_and_playability, opponents_public
     turn_owner, acting_player, phase, pending_effects, trade_state
     bank_information_if_observable, visible_event_history, legal_actions
   chosen_action, outcome, termination_reason
   ```

   Preserve actual dice numbers: equal probabilities do not imply shared rolls.
   Include only information visible to this player. Keep seeds and engine
   provenance in a separate game manifest, outside model inputs. Mark capped or
   failed games explicitly; do not label them as completed wins or losses.
5. **Encode for training.** Convert observations into tile, intersection, road,
   and game-context arrays with stable IDs and player-relative ownership.
   Encode each legal action and mask illegal choices. Keep raw JSONL so the
   encoding can change without regenerating games.
6. **Validate a small pilot.** Check state round trips, card-play timing, legal
   moves, outcome labels, and every decision phase including trades. Verify
   that changing hidden simulator information leaves observations and teacher
   decisions unchanged when visible history and teacher randomness are fixed.
   Verify search leaves the live state and randomness unchanged.
7. **Scale after checks pass.** Split by board seed, keeping both seats and all
   related games together. Keep evaluation seeds untouched by training or
   tuning. Measure teacher win rate, runtime, failures, and decision coverage
   before expanding the dataset.
