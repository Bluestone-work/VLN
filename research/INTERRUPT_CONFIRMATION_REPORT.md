# Independent event-timing confirmation

2026-10-08. **NO-GO for learning an event-triggered interrupt policy.** The
confirmation reproduces a bounded recovery opportunity, but event timing has no
route-level advantage over outcome-blind timing within the same native option and
causes concrete harm on native-successful routes.

## Frozen population and execution

The route sample was frozen before outcomes: 128 R2R-CE training routes, eight
scenes, 16 routes per scene, with exact route identity exclusion from all prior
registered cohorts and validation routes. Capture, control, graph probe and timing
schedule were frozen before any confirmation return was read. The schedule has 43
phase-known event routes and three same-option uniform interior cuts per event
route, 172 cuts total.

The experiment ran 128 native controls plus two arms at every event/uniform cut:
matched sensing then finish, and sensing followed by one unchanged-navigator
replan with native ghost consumption. This is **472 complete rollouts**. The
release checkpoint, seed 100, sensors, sliding, controller, STOP and 15-decision
budget were unchanged. Routes without a phase-known event remain native in the
128-route population denominator.

## Main result

Quality rescue means SR rescue with nDTW nondegradation; cost-capped rescue also
requires no increase in primitive actions.

| Outcome among 7 exposed native failures / 36 exposed native successes | Event cut | Uniform same-option cut |
| --- | ---: | ---: |
| Failed routes with any SR rescue | 3/7 | 3/7 |
| Failed routes with quality rescue | **3/7, 3 scenes** | **3/7, 3 scenes** |
| Failed routes with cost-capped quality rescue | 1/7 | 2/7 |
| Successful routes with any SR loss | **2/36** | **2/36** |
| Successful routes with any nDTW loss | 15/36 | 25/36 |

Event quality rescues are routes **7411, 9186, 6154**. Uniform quality rescues are
**7411, 5660, 6154**. The two sets overlap on 7411 and 6154; event-only route 9186
and uniform-only route 5660 show that recovery is not uniquely tied to the detected
collision. Both arms therefore provide the same route-level rescue count and scene
coverage, while event timing does not pass the required no-success-loss gate.

The event arm loses native SR on routes **1271** and **4049**; uniform timing loses
SR on **5307** and **4049**. Any-cut outcomes are correlated within route and are
not independent policy performance. Successful-route nDTW and primitive-cost
degradation remain visible even when binary SR is unchanged.

## Verification

The sampling route and scene exclusions are hashed and metadata-only. Capture has
1,029 native decisions, 12,464 forward branches and 3,038 reverse branches; all
sentinels, action masks and sensor checks pass. Native/control noninterference is
exact for 128 episodes, 2,304 episode metrics and 18 aggregate metrics.

An independent verifier passes **472 rollouts, 172 interrupted prefixes, 172 cut
sensor pairs, 3,776 reconstructed metric components and 4,171 path-continuity
checks**. No physics is rewound. The first smoke attempt failed only because of a
schedule-key filter bug before intervention arms; it is retained separately and
was followed by a successful 132-rollout smoke verification. The full result has
no discarded simulator or validity failures.

## Decision

The previous timing oracle established that mid-option replanning can sometimes
recover a failed route. This independent confirmation rejects the stronger claim
that the first collision is a useful event-specific trigger: its quality rescue is
matched by outcome-blind within-option timing, and successful-route harm remains.
Do not train an event classifier, PPO policy or VLM trigger from these results.
Do not change the waypoint predictor or revive the adaptive coarse/default/fine
selector. The mechanism is currently best described as an additional replanning
boundary with state-dependent but unresolved timing value.

Further work would require a separately justified mechanism that predicts when
replanning is safe, with abstention and successful-route protection. That is a new
question; this confirmation does not authorize it automatically.

Artifacts: `research/results/interrupt_confirmation_train128/` contains the frozen
sample, capture/control/probe/integrity records, schedule, complete 472-case
returns, independent verification, logs and source hashes.
