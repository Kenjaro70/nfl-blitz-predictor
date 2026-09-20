# How I Built the NFL Blitz Predictor

A writeup of the decisions and the dead ends — including a finding I had to retract, a bug
that quietly made a football feature meaningless, and probabilities that turned out to be
overstated by 2x.

## The problem

Given pre-snap offensive context (down, distance, formation, personnel, score, clock), predict whether the defense is going to blitz (5+ pass rushers). This is the kind of model an offensive coordinator's analyst would want at the line of scrimmage — but it's also genuinely hard, because *defenses try to disguise blitzes*. That's a real ceiling: if a model is too good, something is leaking.

## Why this dataset (and not the obvious one)

The obvious first instinct was `nfl_data_py` (nflverse play-by-play). It has 20+ years of plays, easy install, well documented. But it has no per-play blitz label. The closest proxies are `qb_hit` or `sack` — those measure outcomes ("the QB got pressured"), not the defensive *call*. A model trained on those would really be predicting "did the OL break down," which is a different problem.

The NFL Big Data Bowl 2023 dataset (Kaggle) was a better fit. It includes a `pff_role` column with per-player labels — `Pass Rush`, `Pass Coverage`, `Pass Block`, etc. That gives a clean per-play count of rushers, and the standard NFL definition `rushers >= 5` becomes the binary label.

The tradeoff: only 2021 weeks 1–8 (~8.5k labeled pass plays vs. ~1M plays in nflverse). I made the call that label quality > sample size for a first model. If I were extending this, I'd combine: use BDB to train, then validate on a separate season's PBP with a heuristic label, and see how the model degrades.

## Features

All pre-snap, all from `plays.csv` (the offense's view of the world before the ball is snapped):

| Group | Features |
|---|---|
| Situation | down, yardsToGo, yards_to_goal, quarter, gameClock, score |
| Engineered | score_differential, game_seconds_remaining, is_two_minute, is_red_zone, is_goal_to_go |
| Teams | possessionTeam, defensiveTeam, is_home_offense |
| Pre-snap look | offenseFormation, personnelO, personnelD, **defendersInBox** |

`defendersInBox` ended up being the single most important feature (~19% of total importance) — which is football-obvious in hindsight: when the offense lines up and there are 7 guys near the line of scrimmage, the defense has *fewer* coverage players left, so a blitz is more likely.

### What counts as "pre-snap" — and why `defendersInBox` is allowed but coverage isn't

I excluded `pff_passCoverage` and `dropBackType` while keeping `defendersInBox`, and that
needs a stated standard, because my first attempt at one was wrong. I had written that the
coverage fields were out because "PFF tags them after watching the play." That reason
doesn't survive contact: *everything* in this dataset was recorded after the fact, including
the label itself. Provenance is not the test.

The test I actually apply is two questions:

1. **Temporal:** is the value fixed at the instant before the snap?
2. **Circularity:** is it mechanically entangled with the rusher count — i.e. is it the
   label wearing a different hat?

A feature has to pass both. Measured against the data:

| Field | Fixed pre-snap? | Entangled with the label? | Verdict |
|---|---|---|---|
| `defendersInBox` | Yes — a count of bodies in an alignment | No — 6 in the box still blitzes only 21.9% | **keep** |
| `pff_passCoverage` | No — coverage is disguised and rotates after the snap | Yes, mechanically | drop |
| `dropBackType` | No — 12% of values are `SCRAMBLE`, a post-snap event | — | drop |

The circularity on coverage isn't a judgement call, it's arithmetic. A defense has 11
players; everyone not rushing is covering. So the coverage call is close to the complement
of the rusher count, and it shows up in the data exactly that way:

| `pff_passCoverage` | n | Blitz rate | Mean rushers |
|---|---|---|---|
| Cover-0 | 270 | **83.7%** | 5.64 |
| Cover-1 | 2,011 | 40.9% | 4.45 |
| Cover-3 | 2,665 | 26.6% | 4.25 |
| Quarters | 1,033 | 11.0% | 4.07 |
| Cover-2 | 1,084 | 8.6% | 3.98 |
| Cover-6 | 805 | 4.8% | 4.02 |

Cover-0 means zero deep safeties — which is to say those players are rushing. Handing the
model that column would be handing it half the answer: Cover-0 and Cover-1 plays account
for **49.8% of every blitz in the dataset**. The model would score well and would have
learned nothing an offense can use, because the offense doesn't know the coverage call
pre-snap. That's the whole reason defenses disguise it.

`defendersInBox` is a different kind of thing. It's a physical count of pre-snap alignment,
visible to the quarterback and center, and the relationship to the label is informative but
nowhere near deterministic — it runs from 4.0% blitz at 3 in the box to 75.0% at 9, with the
modal 6-in-the-box sitting at 21.9%, barely off the 24.6% base rate. That is what a
legitimate predictive feature looks like: real signal, no shortcut.

Two honest caveats I'd raise before being asked:

**It's a charting convention, not a measurement.** "In the box" is a judgement about where
the box ends. Two charters could disagree at the margin, and my top feature (19% of
importance) inherits that noise. Worth noting it is *not* a PFF field — it has no `pff_`
prefix, it comes from the NFL's own game data in `plays.csv` — but that makes it a different
organization's convention, not an objective one.

**Deployment needs its own source for it.** A live system can't read `defendersInBox` out of
a CSV; someone or something has to count defenders in real time, from a human in the booth
or computer vision. Any noise in that source degrades the model's most important input. That
is an operational risk rather than a leakage risk, and I'd keep the two separate — the
feature is legitimate, but the pipeline that would feed it in production doesn't exist yet.

## The leakage I thought I found

First run used `train_test_split(stratify=y, random_state=42)` — the textbook default. Got accuracy **0.701, AUC 0.704**. Felt reasonable.

Then I realized: a random play-level split puts plays from the same game in both train and test. And defensive scheme is consistent within a game — the same DC is calling the same plays from the same playbook. So the model was learning a team's *Sunday-specific tendency* and being tested on more plays from that same Sunday.

Fixed it with `GroupShuffleSplit(test_size=0.2, ...)` grouping by `gameId`. That guarantees no game appears in both sets: 97 train games / 25 test games, no overlap. The honest numbers:

| Metric | Random split (leaky) | By-game split (honest) | Δ |
|---|---|---|---|
| ROC AUC | 0.704 | 0.689 | −0.015 |
| Accuracy | 0.701 | 0.682 | −1.9 pts |

I wrote that up as a leakage discovery. **Then I measured it properly and it went away.**

### The measurement that killed my own finding

One 80/20 split of 122 games leaves ~25 games in test, and which 25 you draw moves the
metrics a lot. So I re-ran the entire comparison over 10 seeds — by-game split and
play-level split, same model, same seed, paired:

| | Mean ± SD over 10 splits |
|---|---|
| By-game AUC (honest) | 0.711 ± 0.017 |
| Play-level AUC (leaky) | 0.708 ± 0.013 |
| **Paired difference (leaky − honest)** | **−0.0035 ± 0.0068 (SE)** |

The leaky split scored *higher* in **5 of 10 seeds** — exactly what you'd expect from a
coin flip. The point estimate is slightly negative. There is no detectable leakage effect
at this sample size.

The −0.015 I originally reported was one draw from a distribution with a standard
deviation of 0.017. I'd measured noise and told a story about it. The story was
plausible — defensive scheme *is* consistent within a game, the mechanism is real — which
is exactly why I believed a single number that happened to point the way I expected.

I'm keeping the by-game split, because it's still the correct estimator: it answers "will
this work on a game we haven't seen," which is the question, and it can't be *worse* than
the alternative. But the honest claim is "I used the right split," not "I caught and
quantified leakage." Those are very different sentences and only one of them is supported.

The wider lesson is the one I'd actually want to be asked about: **the single-split
measurement wasn't precise enough to support any conclusion I drew from it**, in either
direction. Every headline number in this writeup now carries error bars for that reason,
and a couple of other conclusions didn't survive the same treatment (see the random forest
vs. logistic regression comparison below).

### What grouping by game still doesn't fix

Grouping by `gameId` doesn't remove *team* leakage: the same defensive coordinator appears
in both train and test games, and `defensiveTeam` is a feature, so the model is partly
memorizing coaching tendency. To claim generalization to an unseen defense I'd group by
`defensiveTeam`; to claim generalization forward in time I'd split weeks 1–6 vs. 7–8.
Those are different claims and I haven't measured either.

## The bug that mattered more than the leakage

While writing this up I went back to the BDB data dictionary for `absoluteYardlineNumber`,
the column I'd been feeding the model as field position. I had it documented in my own API
as "yards from the offense's goal line." It isn't.

It's a raw coordinate in the tracking system's frame: range **11–109**, covering a 120-yard
field with both end zones. Critically, its orientation depends on which direction the
offense happens to be driving. I checked every play in the dataset:

- `absoluteYardlineNumber == yards_to_goal + 10` on 4,480 plays
- `absoluteYardlineNumber == 110 - yards_to_goal` on 4,201 plays
- one or the other on **8,556 of 8,556** — i.e. always, roughly half each way

So the same physical spot on the field encoded as two different numbers depending on which
end zone the offense was headed for. Any distance-to-goal signal was smeared across the
whole range.

Two engineered features were built on top of it, and both were close to inert:

| Feature | Fired (buggy) | Fired (fixed) | Truth |
|---|---|---|---|
| `is_red_zone` | 385 plays (4.5%) | 1,182 plays (13.8%) | missed 885 real red-zone plays |
| `is_goal_to_go` | 5 plays (0.06%) | 447 plays (5.2%) | effectively a dead column |

Worse than under-firing: of the 385 plays the old `is_red_zone` did flag, **88 were the
offense backed up inside its own 20** — the opposite of a scoring situation, labeled as one.

The fix is to ignore `absoluteYardlineNumber` entirely and derive distance from
`yardlineSide` + `yardlineNumber`, which are direction-independent:

```python
yards_to_goal = np.where(
    df["yardlineSide"] == df["possessionTeam"],
    100 - df["yardlineNumber"],  # own half: have to cross midfield first
    df["yardlineNumber"],        # opponent half: the number *is* the distance
)
```

(`yardlineSide` is null exactly at the 50, where both branches give 50 anyway — so the
null case needs no special handling.)

**And it bought almost nothing: AUC 0.6889 → 0.6892.**

I want to be straight about that, because I expected a real gain and didn't get one. Three
reasons it barely moved, and I think the third is the actual lesson:

1. A random forest can partially route around a scrambled feature. It can still split on
   "near either end zone," which carries some signal regardless of direction.
2. Field position is a weak blitz predictor to begin with — `yards_to_goal` lands at 8.4%
   importance, well behind `defendersInBox` at 19%.
3. **Correctness and performance are different axes.** The old model was getting an okay
   score partly by accident, off a feature that meant nothing and two flags that almost never
   fired. Any conclusion I drew about red-zone blitz behavior from that model would have been
   fiction. The metric barely moved; the model's claim to describe football moved a lot.

If anything the fix is more useful as a demonstration that my feature pipeline now matches
the data dictionary — and that I check.

## Why I don't lead with accuracy

The first draft of this README opened with "Accuracy 0.682." That was a mistake, and
it's a worse one than the leakage.

Blitzes are **26.1% of the held-out plays**. So a model that prints "no blitz" and
nothing else — no features, no fitting, no thought — scores **0.739 accuracy** and beats
mine by almost six points. Leading with accuracy means leading with a number where the
dumbest possible baseline wins.

The full comparison, all on the same by-game split:

| Model | ROC AUC | Accuracy | Blitz precision | Blitz recall |
|---|---|---|---|---|
| Always predict "no blitz" | 0.500 | **0.739** | — | 0.00 |
| Logistic regression | 0.682 | 0.655 | — | — |
| **Random forest** | **0.689** | 0.619 | 0.37 | 0.66 |

Three things I take from this:

**The accuracy gap is a choice, not a bug.** The model is an alert, and the threshold is
tuned to a stated 3:1 cost on missed blitzes (next section), which spends accuracy to buy
recall. It catches **66% of blitzes** and is right **37%** of the time when it fires,
against a 26.1% base rate. If I wanted the accuracy number to look good I'd move the
threshold to 0.5 and catch fewer blitzes, which would be optimizing the writeup instead of
the decision.

**AUC is the honest headline** because it's threshold-free and base-rate-independent.
0.711 ± 0.017 against a 0.500 no-skill floor is modest but real — consistent with defenses
actively disguising blitzes, which is what I'd expect.

**The random forest does not beat logistic regression.** On the single split it's +0.007
AUC. Across 10 splits it's **+0.003 ± 0.010, winning 6 of 10** — a coin flip. This was the
second conclusion that didn't survive the error bars. The signal in these features is
essentially linear, and I can't justify the ensemble on performance. If I were shipping
this for real I'd use the logistic regression: same accuracy, and I could hand a coach a
list of coefficients instead of 300 trees.

I'd rather walk into a room with a defensible 0.711 ± 0.017 AUC and a clear statement of
the base rate than a 0.682 accuracy that the first person to do the subtraction takes apart.

## The decision: turning a probability into a call

A probability isn't a decision, and for most of this project the decision was sklearn's
default 0.5 — a number I never chose. Fixing that turned out to require fixing something
else first.

### The probabilities were lying

`class_weight='balanced'` does its job by over-weighting the minority class, and a side
effect is that the predicted probabilities come out inflated. On the held-out games the
raw model predicted a **mean blitz probability of 0.434** against an **observed rate of
0.261**. It was overstating blitz risk by nearly 2x, and the API was serving those numbers
to three decimal places.

Isotonic regression, fit on out-of-fold training predictions, fixes it:

| | Brier | Mean predicted | Observed |
|---|---|---|---|
| Raw | 0.2057 | 0.434 | 0.261 |
| Isotonic | **0.1752** | **0.247** | 0.261 |

Reliability by bucket after calibration — predicted 0.29 now really does hit 31%:

| Bucket | n | Predicted | Observed |
|---|---|---|---|
| 0.00–0.15 | 511 | 0.100 | 0.137 |
| 0.15–0.25 | 430 | 0.205 | 0.202 |
| 0.25–0.35 | 612 | 0.294 | 0.307 |
| 0.45–0.60 | 153 | 0.517 | 0.542 |
| 0.60–1.00 | 40 | 0.811 | 0.650 |

Isotonic is monotone, so AUC is unchanged. This fixes what the numbers mean, not how well
they rank — which is exactly the distinction I'd missed when I called the old 53% output
"appropriately uncertain." It wasn't uncertain, it was wrong.

### The cost model

Now the threshold. The consumer is an offensive analyst deciding whether to keep extra
protection in:

- **Missed blitz** → unblocked rusher, free run at the QB. Sack or hurry.
- **False alarm** → a back or TE blocks instead of running a route. One fewer target.

A missed blitz is worse, but not catastrophically so. I put it at **3:1** — and then tried
to measure it rather than leave it asserted. That attempt is its own section below; the
short version is that it failed, and 3:1 remains an assumption I can't defend with data.

So it's a named constant, it's in the metadata with its provenance, and the training run
prints what happens if you disagree with me:

| FN:FP | Threshold | Precision | Recall | Alert rate |
|---|---|---|---|---|
| 1:1 | 0.50 | 0.56 | 0.22 | 0.10 |
| 2:1 | 0.33 | 0.48 | 0.44 | 0.24 |
| **3:1** | **0.25** | **0.37** | **0.66** | **0.46** |
| 5:1 | 0.17 | 0.32 | 0.82 | 0.67 |
| 8:1 | 0.11 | 0.28 | 0.95 | 0.89 |

The nice part: once probabilities are calibrated, the cost-optimal threshold has a closed
form. Alert whenever `p · C_FN > (1−p) · C_FP`, i.e. `p > C_FP/(C_FP + C_FN)` — so 3:1
gives **0.25**, no tuning required. I grid-searched it on out-of-fold training predictions
as a check and the grid independently picked 0.25. That agreement is really a calibration
test: the closed form only works if the probabilities mean what they say.

The threshold is chosen on training folds only. Picking it on the test set would be the
same error as tuning a hyperparameter there, and it's an easy one to make because
threshold selection doesn't *feel* like fitting.

### And it didn't reduce cost

At 3:1 the tuned threshold and the old 0.5 default produce the **same** expected cost on
this split — 0.5606 per play, from genuinely different confusion matrices (157 missed
blitzes vs. 214). A coincidence, and I only noticed because I printed both.

So this didn't make the model cheaper. What it did:

1. The operating point is now a stated decision traceable to an assumption someone can
   argue with, rather than a library default.
2. It moves correctly when the assumption changes — the sensitivity table is the actual
   deliverable, not the single row I picked.
3. The probabilities are honest, so `0.27` can be reported to an analyst as "27%" without
   it being a lie.

That's three things I can defend, and zero improvement in the headline metric. I think
that's the normal shape of this kind of work and I'd rather show it than dress it up.

## Trying to ground 3:1 in EPA — and failing

An asserted cost ratio bothered me, because everything downstream of it is rigorous and it
isn't. So I went to measure it. `training/analyze_cost_ratio.py` is that attempt.

The idea: EPA isn't in the Big Data Bowl data, but nflverse play-by-play has it, and BDB's
`gameId`/`playId` map onto nflverse's `old_game_id`/`play_id`. That join comes back **8,557
of 8,557 — 100%**. So every play gets a real EPA, and I can look at the four cells of
{did the defense blitz} × {did the offense keep extra protection}:

- cost of a missed blitz = `EPA(blitz, heavy protection) − EPA(blitz, light)`
- cost of a false alarm = `EPA(no blitz, light) − EPA(no blitz, heavy)`

| Cell | n | EPA | Sack% | Routes | Air yds | Play-action |
|---|---|---|---|---|---|---|
| 4-man rush / light | 4,599 | +0.045 | 6.0% | 5.00 | 8.98 | 12% |
| 4-man rush / heavy | 1,140 | **+0.201** | 6.6% | 3.69 | 12.85 | **61%** |
| blitz / light | 866 | +0.017 | 7.2% | 5.00 | 8.02 | 15% |
| blitz / heavy | 1,044 | +0.060 | 10.2% | 3.66 | 10.56 | 39% |

Which gives cost_FN = **+0.043** and cost_FP = **−0.156** — a ratio of **−0.27:1**. The
false-alarm cost is *negative*: on this measurement, "over-protecting" against a four-man
rush is associated with EPA **four times better** than not.

That's not a surprising finding, it's a broken measurement, and the last three columns say
why. The heavy-protection cells are **61% play-action with 12.9 mean air yards**, against
12% and 9.0 for light. Those aren't protection decisions — they're max-protect deep shots.
The extra blocker is a *marker of aggressive play design*, and plays designed to throw deep
off play-action have high EPA when they work.

The root cause is a data-definition problem, the same species as the yardline bug earlier:
**`pff_role == 'Pass Block'` is assigned from what a player did *after* the snap.** A back
who stays in because he read blitz gets counted as a blocker. So my "protection" variable is
partly an effect of the blitz I'm trying to warn about, and partly a proxy for the play call.
The offense's pre-snap protection *intent* — the thing the decision is actually about — is
not in this dataset at all.

Cluster-bootstrapped by game, cost_FN's 95% CI is [−0.070, +0.158] — it spans zero, so I
can't even establish that a missed blitz is costly by this route. cost_FP is reliably
negative at [−0.231, −0.082], i.e. reliably the wrong sign.

**Verdict: the cost ratio is not identified by this data.** 3:1 stays, labeled as an
assumption in the code, the metadata, and the README. The sensitivity table is the honest
deliverable; the row I picked is a guess.

### What the attempt did establish

Two things worth more than the number I was chasing.

**A sack is enormously expensive: −2.06 EPA** (sack plays average −1.849, non-sack pass
plays +0.207). And a blitz does raise sack risk, from **6.1% to 8.8%**. Multiply those out
and the sack channel alone is worth −0.055 EPA per blitz.

**But the net effect of a blitz on offensive EPA is only −0.036**, 95% CI [−0.094, +0.025],
with an 87% posterior that it hurts the offense at all. The net is *smaller* than the sack
channel, because a blitz also gives up coverage — fewer defenders in the secondary, bigger
plays when the protection holds. Blitzing was close to EPA-neutral in 2021.

That last number is the one I'd actually lead with, and it's slightly uncomfortable: **it
caps what this entire project can be worth.** If a blitz only costs the offense ~0.036 EPA
on average, then a perfect blitz oracle is worth a fraction of that, and my 0.711-AUC model
is worth a fraction of the fraction. I'd rather know that and say it than ship a model with
a good-looking ROC curve and no idea whether anyone should care.

It also reframes the use case. The value isn't in average EPA — it's in the tail, the
Cover-0 all-out pressure where the sack probability spikes and a free rusher ends the drive.
A model aimed at *that* would be a different model: a rarer, more extreme label, probably
higher rusher thresholds, and evaluated on the plays that actually swing games rather than
on all 8,549 pass plays equally.

### What would identify the ratio

The pre-snap tracking data, which is sitting unused in this repo. `week1.csv` … `week8.csv`
give every player's position at 10Hz, including before the snap. Backfield alignment in the
final pre-snap frame is a measure of protection *intent*, independent of what happened after
the ball moved. That's the variable this analysis needed and the experiment I'd run next.

## Sanity check: are the predictions football-sensible?

Four scenarios against the deployed API, on calibrated probabilities. `lift` is the
probability divided by the 24.6% base rate — the number I'd actually put in front of an
analyst, because "1.1x an average pass play" is more useful than a bare percentage:

| Situation | Blitz prob | Lift | Alert? |
|---|---|---|---|
| 1st & 10 own 25, 5-man box, opening drive | 14.7% | 0.60x | no |
| 3rd & 8, midfield, 6-man box | 27.0% | 1.10x | yes |
| 4th & 1 goal-to-go, 9-man box, late Q4 | 51.7% | 2.10x | yes |
| 3rd & 2, 8-man box, tied Q4 | 59.4% | 2.41x | yes |

The direction is right across the spectrum and the spread is football-sensible: a heavy
box on a short third down is ~2.4x more blitz-prone than an opening-drive vanilla look.

Worth noting that the first three of these used to read 33.1% / 53.1% / 67.7% before
calibration. Same model, same ranking — but those numbers were inflated by about 2x, and
I had written "appropriately uncertain" about a 53% that should have been 27%. The ranking
was fine the whole time; the numbers were not, and I was reading meaning into them anyway.

Two fields also came *out* of the API here. It used to return a `confidence` label derived
from distance-to-0.5, and a `num_pass_rushers_estimate` that mapped the probability onto
a rusher count. The model is binary and never predicts a count, and a confidence band
computed from a miscalibrated probability is decoration. I removed both rather than dress
them up, and there's a test asserting they don't come back.

## Limitations I'd call out in an interview

1. **One season, 8 weeks.** Defensive coordinators change scheme between seasons. The model would degrade fast on 2022+ data without retraining.
2. **No player-tracking features.** The `week*.csv` files contain 10Hz tracking for every player on every play — that's where the next big AUC gains live (defender alignment depth, motion response, walk-up timing). Skipped for scope; it would roughly triple the project size.
3. **The 3:1 cost ratio is asserted, not measured.** The threshold follows rigorously from it, but the ratio itself is my guess at how an offensive coordinator trades a sack risk against a lost route. The right version measures both sides in EPA from play-by-play data. Until then, the sensitivity table is the honest deliverable and the single chosen row is an assumption.
4. **A random forest I can't justify.** It ties logistic regression across 10 splits. It's still here because swapping it is a bigger change than this pass allowed, but "we picked the more complex model and it performed identically" is a fair thing to be challenged on.
5. **Personnel strings aren't standardized.** "1 RB, 1 TE, 3 WR" is treated as a categorical token. A better approach would parse it into integer columns (n_rb, n_te, n_wr, n_db).

## What I'd do next

In order of expected payoff:

1. **Ship the logistic regression** — it matches the random forest across 10 splits and is interpretable. The burden of proof is on the ensemble and it hasn't met it.
2. **Measure protection intent from the pre-snap tracking data** — backfield alignment in the final pre-snap frame of `week*.csv`, which is what the EPA cost-ratio analysis needed and couldn't get from post-snap role labels.
3. **Re-aim the label at the tail** — the net EPA cost of an average blitz is only −0.036, so the value is in all-out pressure, not in the mean. A 6+-rusher label on a smaller, more extreme population is probably the more useful model.
4. **Add team historical blitz rate as a feature** — compute defensive team's training-set blitz rate, join in. Captures coaching tendency without leakage.
5. **Parse personnel into integer columns** — frees the model from learning that "1 RB, 1 TE, 3 WR" and "1RB 1TE 3WR" are the same.
6. **Try LightGBM / XGBoost** — but with error bars this time; a claimed +1–3 AUC points would need to clear a ±0.017 split-to-split SD before I'd believe it.
7. **Add player-tracking features** — defender alignment depth, motion response. This is the real ceiling-mover but is its own project.
8. **Group by `defensiveTeam`, and split by week** — two different generalization claims I currently can't make: performance against an unseen defensive coordinator, and performance forward in time.

## Stack

- **Data:** NFL Big Data Bowl 2023 (Kaggle)
- **Modeling:** scikit-learn — RandomForestClassifier, GroupShuffleSplit + GroupKFold
  (by-game splits and out-of-fold predictions), IsotonicRegression (probability
  calibration), LabelEncoder
- **Serving:** FastAPI + uvicorn on Python 3.11
- **Containerization:** Docker — separate training and serving images, train produces `.joblib` artifacts baked into the serving image
- **Distribution:** `kenjaro/nfl-blitz-predictor` on Docker Hub
