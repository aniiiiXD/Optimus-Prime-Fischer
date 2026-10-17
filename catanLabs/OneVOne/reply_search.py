"""Compare two choices through the opponent's turn and our next roll."""

import random

import numpy as np
from catan_kit.obs import Obs
from my_team.nano import Agent as Nano, AT


def next_turn_value(game, action, color, config, seed):
    rng_state = random.getstate()
    try:
        rng = random.Random(seed)
        for _ in range(128):
            world = Obs(game, color).sample_world(rng)
            if world.winning_color() is None:
                break
        else:
            return None
        start = world.state.num_turns
        # shortcut: cheap Nano replies approximate the opponent; test more styles if useful.
        preferences = dict(config.get("nano", {}), search_depth=1, beam_width=4)
        policies = {c: Nano(preferences) for c in world.state.colors}

        def execute(chosen):
            if chosen.action_type == AT.ROLL:
                random.seed(seed * 10007 + world.state.num_turns)
            world.execute(chosen)

        random.seed(seed)
        execute(action)
        for _ in range(24):
            winner = world.winning_color()
            if winner is not None:
                return 10000. if winner == color else -10000.
            current = world.state.current_color()
            prefix = f"P{world.state.color_to_index[color]}_"
            if (world.state.num_turns > start and current == color
                    and world.state.current_prompt.name == "PLAY_TURN"
                    and world.state.player_state[prefix + "HAS_ROLLED"]):
                scorer = policies[color]
                scorer.me, scorer.enemy = color, next(c for c in world.state.colors if c != color)
                scorer.mk = prefix
                scorer.ek = f"P{world.state.color_to_index[scorer.enemy]}_"
                scorer._init_map(world.state.board)
                scorer._set_belief(Obs(world, color))
                scorer.deck_left = len(world.state.development_listdeck)
                return scorer._value(world) - 5.5 * scorer._threat(world)
            actions = world.playable_actions
            state = random.getstate()
            view = Obs(world, current)
            index = policies[current].decide(view, actions) if len(actions) > 1 else 0
            random.setstate(state)
            execute(actions[index])
        return None
    finally:
        random.setstate(rng_state)


def check_replies(game, plan, incumbent, color, config):
    baseline = int(np.argmax(plan.heuristic))
    challenger = incumbent if incumbent != baseline else max(
        (i for i in range(len(plan.actions)) if i != baseline), key=lambda i: plan.heuristic[i])
    values = [[next_turn_value(game, plan.actions[i], color, config,
                               701 + game.state.num_turns * 1009 + k)
               for k in range(2)] for i in (baseline, challenger)]
    complete = all(value is not None for arm in values for value in arm)
    chosen = incumbent
    if complete:
        differences = np.asarray(values[1]) - values[0]
        if np.all(differences > .25):
            chosen = challenger
        elif np.all(differences < -.25):
            chosen = baseline
    return chosen, {"reply_checked": True, "reply_complete": complete,
                    "reply_roots": [baseline, challenger], "reply_values": values,
                    "reply_changed": chosen != incumbent}
