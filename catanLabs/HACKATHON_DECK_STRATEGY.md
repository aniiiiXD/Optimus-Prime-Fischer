# Catan hackathon: strategy ideas from the rules deck

**Status: pre-spec research plan, 16 October 2026.** The [11-slide Catan rules
deck](https://drive.google.com/file/d/1ftEFhs8wcwiJ5NvM5524NWubRgbnT_OE/view?usp=sharing)
was read from the downloaded `Hyde@IITB.pdf`. The organizer announcement
confirms Catan, limited compute, bot-versus-bot play, and an evaluation based
on wins plus a clear strategy explanation. The full rules, bot interface, and
evaluation details arrive on hackathon day. This document separates what the
deck shows from ideas we still need to test.

## What the deck actually tells us

| Slides | Shown in the deck | Consequence for a bot |
|---|---|---|
| 2 | Two players compete. | Prepare a two-seat evaluation in addition to our existing four-seat lab. A four-player win rate is not a two-player result. |
| 3–5 | Hex tiles produce wood, brick, sheep, wheat, or ore when their number is rolled; settlements and cities collect. | Value locations by expected production **and** by the build paths those resources enable. |
| 6–7 | Road costs wood + brick; settlement costs wood + brick + sheep + wheat; city costs 3 ore + 2 wheat; development card costs ore + wheat + sheep. Roads connect, settlements collect, cities produce twice as much. | Assess a build by the time to afford it, the VP gained, and what it opens next. A city is worth 2 VP in total, so upgrading a 1-VP settlement adds 1 VP. |
| 8 | Knight, Road Building, Year of Plenty, and hidden Victory Point cards are illustrated. | Use cards to complete concrete plans: army, robber pressure, two free roads, or the exact resources for a build. The slide is illustrative; it does not establish the full deck composition. |
| 9 | A 7 activates the robber; its tile stops producing and an opponent may lose a resource. | Robber placement can deny production while improving our own position. The exact steal and discard rules still need the event spec. |
| 10 | Bank exchange is 4:1, or 3:1 and 2:1 at ports. | A port matters when our production repeatedly creates a surplus of the resource it discounts. The deck does not define player-to-player offer rules. |
| 11 | First to 10 VP wins: settlements 1, cities 2, VP cards 1, Largest Army 2 after 3 Knights, Longest Road 2 after 5 connected segments. | Search explicitly for point swings and immediate wins; do not value an extra road or Knight only by its resource cost. Award, tie, and interruption details await the spec. |

The organizer message also mentions **trade and bluffing**. That is a reason to
prepare for opponent offers, not proof of their exact action format or whether
they are present in every match. Standard Catan mechanics absent from these
slides, such as Monopoly, must also be checked against the event rules rather
than inferred away.

## Similar public strategy resources

These are human strategy ideas, not evidence that a bot will win. The event's
final rules can also change which ideas apply.

| Source | Useful idea for our bot |
|---|---|
| [The Settlers of Catan Strategy Guide (six-page PDF)](https://funboardgames.org/wp-content/uploads/2012/04/CatanStrategyGuide.pdf) | A compact, deck-like guide covering initial sites and roads, trading, robber, cities, awards, and development cards. It favors production while considering resource mix and expansion. |
| [Colonist starting strategies](https://blog.colonist.io/guide-to-catan-starting-strategies/) | Four opening plans: road/settlement expansion, balanced expansion, ore/wheat/sheep cities and cards, and concentrated production with a matching 2:1 port. Score the initial two sites jointly. |
| [Colonist beginner-to-advanced strategy](https://blog.colonist.io/catan-strategies-beginner-intermediate-advanced/) | Production pips, resource access, opponent strength, and trade timing. Its leader estimate suggests features for robber targeting. |
| [Reddit: how to read the board](https://www.reddit.com/r/Catan/comments/18eobl5/) | Community discussion of opening archetypes, wheat/ore synergy, road direction, and the risk of being cut off. |
| [Reddit: first pair of settlements](https://www.reddit.com/r/Catan/comments/1u0da36/how_does_one_learn_the_arts_of_first_pair_of/) | Players disagree on how much to value ports, pips, and all-five-resource coverage. That disagreement is a reason to run an ablation, not hard-code one ranking. |
| [Reddit: analysis of 50,000 games](https://www.reddit.com/r/Catan/comments/1qed7tu/i_analyzed_50000_games_of_catan_and_built_a_site/) | Possible hypotheses about early cities and card timing. Treat observational win rates cautiously: a strong move can correlate with already winning. |

I also searched public X/Twitter results for Catan opening and resource
strategy. The indexed results did not yield a substantive, verifiable strategy
thread worth citing, so no X claim is being used here.

**Tests suggested by these sources:** (1) joint opening pair selection versus
single-site pip maximization; (2) board-dependent ore/wheat/sheep versus
wood/brick expansion plans; (3) port value conditional on matching production;
(4) city-now versus settlement-first by expected turns to the next VP; and
(5) initial road direction weighted by reachable open sites and likely
opponent blocking. Run each on held-out seeds with the same opponents and
latency budget; keep only changes that improve wins.

## The playing strategy to try first

1. **Plan the opening as a pair.** Score the two initial settlements together:
   likely dice production, coverage of the five resources, access to ports,
   and strong sites reachable by the initial roads. A great first hex cluster
   can be a poor opening if the second placement leaves no way to build a
   settlement or city. In two-player games, include the value of occupying a
   scarce site before the opponent can take it.
2. **Choose a build target, then price actions against it.** Estimate the
   turns needed to reach the next settlement, city, development card, or
   Longest Road claim. Wood and brick are especially useful for early
   expansion; ore and wheat drive cities. Recompute scarcity as the board and
   hand change. A road that reaches a productive settlement site is valuable;
   an isolated road may only be useful in a Longest Road race.
3. **Check for a win before using the general value score.** After rolling,
   inspect legal one- and two-action sequences such as bank trade → city,
   trade → settlement, Year of Plenty → build, and road → Longest Road.
   Continue Road Building through both free roads before judging it. Search
   deterministic, known outcomes only; use expected values for hidden card
   draws and future dice instead of reading the simulator's secrets.
4. **Treat development cards as options with timing.** A Knight can move the
   robber and advance Largest Army; Year of Plenty can finish an immediate
   build; Road Building can open a site or claim Longest Road. A hidden VP
   card can end a close game, but its identity is unknown when buying.
   Compare the expected benefit of a purchase with the city or settlement
   that the same ore, wheat, and sheep could fund.
5. **Use the robber for a two-sided swing.** Block the opponent's valuable
   production while avoiding our own. Prefer a target that slows the
   opponent's next likely build or threatens a resource we need. Infer likely
   opponent holdings only from public rolls, builds, trades, and discards;
   never use hidden engine fields. In a two-player match, denying one rival
   matters more than spreading harm across several rivals.
6. **Trade for completed plans, not raw card count.** Use 4:1 or a port when
   the exchange immediately unlocks a valuable build or prevents a costly
   stall. If the event permits direct offers, price each offer by **our gain
   minus the opponent's gain**. Reject trades that hand the rival a likely
   winning build. Track public offers and actions to estimate their needs;
   any bluffing tactic must stay within the actual legal action interface.
7. **Switch from growth to the VP race.** Near 10 points, compare how soon
   each player can win. Take a certain win, defend against the rival's next
   plausible win, and value Largest Army and Longest Road as 2-point swings.
   Building extra production late is useful only if it improves the race.

One practical decision order is: **legal immediate win → legal short winning
line → prevent an opponent's near win → best long-term value action**. Each
step must return an action from the engine's legal-action list.

## What we already have, and what it proves

The promoted local policy is [TacticalVP with `tactical_min_vp=0`](Competitive/TACTICAL_VP_MIN0_CONFIRM.md).
It uses FairValue's public-information value score plus bounded deterministic
two-action VP search. In provisional **four-player** Catanatron games, its
fresh paired tests improved win rate over FairValue by 9.75 points in a
diverse opponent mix and 10.25 points in a stronger mix. The event format
has not yet been measured against that policy.

The [77,569-parameter Transformer distillation](Transformer/runs/gate6_tactical_distill/DISTILLATION.md)
reached 81.0% held-out action agreement in a 590-second CPU run, then lost
substantial playing strength in fresh games. Its opening settlement, robber,
and road decisions were especially weak. A 6× phase sampling attempt made
game results worse. This points to a safer ML role: let a small model rank
candidate search branches or estimate leaf value while the proven legal
policy remains the fallback. Compare that helper with an MLP using the same
data, parameters, and CPU time. Promote it only on game outcomes.

## Ranked experiments

| Priority | Experiment | Evidence that would justify keeping it |
|---|---|---|
| 0 | Build a two-player match harness from the event contract; run the existing legal baseline and TacticalVP against each reference bot. | Complete games, zero illegal actions, measured CPU latency, and seed/seat-swapped win rates. |
| 1 | Joint opening placement and initial-road planner. | Better opening production/resource coverage **and** higher held-out wins than TacticalVP; inspect turns to first expansion and first city. |
| 2 | Extend the bounded VP scan to card and port continuations that the official rules permit. | New winning lines found without hidden-state access; a paired win gain that survives fresh seeds and the latency cap. |
| 3 | Opponent-aware robber and trade module. | More opponent production denied or useful trades completed, with higher match win rate. Direct trading waits for the official offer/response rules. |
| 4 | Small action-ranker or value model to order TacticalVP's legal search branches. | Faster search or stronger play at the same decision budget; the fallback still handles low-confidence or unsupported states. |
| 5 | Self-play outcome/value learning or RL from a strong warm start. | Only after we measure enough completed games per CPU minute and a useful gain over the non-RL baseline. Avoid treating offline imitation accuracy as the objective. |

For datasets, collect **player-visible** observations, legal actions,
teacher choices, public outcomes, and final winners across varied maps,
seats, opponents, and decision phases. Deduplicate visible positions. Keep
whole board seeds and game trajectories together across train/validation/test
so near-identical states do not leak into evaluation. Deliberately include
openings, robber choices, card continuations, trades, and near-win positions;
check that added rare-phase data does not destroy ordinary-turn decisions.

For a quick screen, use paired board seeds and swap the two seats. Report
completed games, wins, VP, illegal actions, per-decision p95 latency, and a
confidence interval for the paired win difference. Confirm promising changes
on untouched seeds and several opponent types. A symmetric two-player match
starts at 50% against an equally strong opponent; that is not a benchmark
against an arbitrary reference bot.

## Hackathon-day contract check

Before making an event-strength claim, fill the [rules manifest](rules_manifest.json)
from the actual problem statement and record the differences from our local
four-player setup:

- Player count; 10-VP target; board, port, and number layouts; setup order.
- Road/settlement distance rules, Army/Road award and tie rules, development
  deck and card timing, robber and discard behavior.
- Whether player-to-player trades exist, their legal offers/responses, and
  what a player can observe about an opponent's hand or past actions.
- Bot input/output format, legal-action list, invalid-action behavior,
  persistent memory, allowed search/state copies, and CPU/memory/time limits.
- Reference bots, seat policy, board distribution, scoring, and match caps.

Then run one complete legal game through the official interface, measure
whole-decision latency on the target CPU, adapt the two-player tournament,
and compare the baseline before changing strategy weights or training a new
model. Keep the deck's introductory rules and the event's executable rules
distinct in every report.
