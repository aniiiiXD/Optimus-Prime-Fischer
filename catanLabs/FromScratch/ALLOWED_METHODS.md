# Allowed methods and why they help

The organizer explicitly permits these methods. The objective is consistent
match wins under limited compute, with a strategy we can explain clearly.

| Method | Use in our Catan agent | Benefit |
| --- | --- | --- |
| Search | Simulate legal candidate actions and plausible dice/opponent outcomes before committing. | Finds moves whose value appears only after future turns. |
| Self-play | Play current and past versions of our agent against each other. | Produces fresh training games and exposes strategies that beat earlier policies. |
| Learned value functions | Train a model to estimate the quality or win chance of a visible game state. | Scores positions quickly and guides search beyond immediate VP gains. |
| Imitation learning | Train a policy on decisions from a stronger teacher. | Gives the model a capable, stable starting policy before RL. |
| Handcrafted heuristics | Score production, resource bottlenecks, settlement expansion, VP routes, robber impact, and trades with explicit formulas. | Provides a fast baseline, an explainable strategy, and useful teacher signals. |
| Other ideas | Add opponent resource estimates, trade accept/reject logic, curriculum learning, or a small specialist model for particular phases. | Lets us adapt to the official rules and exploit useful structure in 1v1 play. |
| AI coding assistants | Use them to implement, test, profile, and analyze experiments. | Speeds up development; it is not part of the game-playing policy. |

## Judging criteria

| Criterion | What it means for us |
| --- | --- |
| Consistent wins | Evaluate on held-out board seeds, both seats, reference bots, and a frozen opponent league. |
| Clear strategy explanation | Keep each component interpretable and preserve match evidence for every claimed improvement. |
