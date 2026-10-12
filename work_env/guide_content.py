# guide_content.py — Chapters for the in-app Guide tab.
#
# Separate module so the prose does not bloat dashboard_ui.html. The
# /api/guide endpoint reads GUIDE and returns it as JSON. Edit any
# chapter here; no other file needs to change.
#
# Schema per chapter:
#   {
#     "id":     str,             # short slug, used as <details> id
#     "title":  str,             # displayed heading
#     "intro":  list[str],       # plain-English paragraphs
#     "tech":   list[str],       # optional technical paragraphs
#     "bullets": list[str],      # optional bullet list (plain-English)
#   }

GUIDE = [
    {
        "id": "welcome",
        "title": "What this platform is",
        "intro": [
            "32k-XAI is a research platform that simulates a society of 32,000 cognitive agents. "
            "Each agent has its own personality, mood, and decision rules, and every rule is a "
            "transparent mathematical function rather than a neural network.",
            "The purpose is to study how macro-level social phenomena — cooperation, trust, "
            "inequality, rejection of unfair offers — emerge from individual decisions that can be "
            "traced to specific psychological parameters. Unlike deep-learning agents, every "
            "decision here can be interrogated: given one agent at one moment, you can read every "
            "number that drove its choice and know where each number came from.",
            "The population is not invented. Agents are drawn from a Vine Copula estimated on real "
            "human psychometric data — the IPIP-FFM for Big Five personality, the SD3 for Dark "
            "Triad traits, and the DASS for distress. Every trait carries a provenance record: "
            "grounded directly in a human sample, grounded via a proxy, derived from other "
            "columns, or assigned as a synthetic prior with a documented flag.",
        ],
        "tech": [
            "The decision loop is: temporal-difference learning over an MDP with a two-step "
            "Bellman backup, an arbitration weight w_MB that blends model-based planning with "
            "model-free habit, Fehr–Schmidt inequity aversion on the accept decision, and softmax "
            "action selection. No neural networks or large language models are involved.",
            "The population engine is JAX-vectorised: all 32,000 agents' Q-tables live in a "
            "single tensor and are stepped in parallel. On the macOS M5 the full 32k x 500 run "
            "takes roughly 2.9 s on Metal and 3.7 s on CPU; on an NVIDIA GTX 1660 Ti under "
            "Windows it takes 0.26 s. Runs are deterministic given the same seed, N, T, and "
            "shock configuration.",
            "Five human benchmarks are checked after each run: ultimatum mean offer, ultimatum "
            "rejection rate, trust-game mean send, trust-game mean return, and wealth Gini. PASS "
            "means the simulated value lands within one standard error of the published estimate.",
        ],
    },
    {
        "id": "population",
        "title": "The population",
        "intro": [
            "Every experiment in this platform runs on the same population: 32,000 agents, each "
            "with a personality, a current mood, a learning rate, a fairness parameter, and a "
            "dozen other characteristics drawn jointly from human covariance data.",
            "The Crowd tab is where you inspect this population. You can see the distribution of "
            "any trait, filter to a subgroup, and compare the subgroup to the rest of the "
            "population — with effect sizes, not just differences.",
            "Crucially, every trait in the Crowd tab carries a provenance badge: GROUNDED (green) "
            "means the column came from a direct empirical sample; PROXY (amber) means it is a "
            "defensible but weaker stand-in; DERIVED (blue) means it is computed from other "
            "columns; SYNTHETIC (red) means it is a literature-derived prior flagged in every "
            "trace. This is not decoration — it tells you how much weight to put on any given "
            "trait when you read a result.",
        ],
        "tech": [
            "The copula is a D-Vine or R-Vine estimated with pair-copula family selection per "
            "edge (Gaussian, Student-t, Clayton, Gumbel, Frank, Joe) via BIC. The current "
            "calibrated copula (P6d) has 26 columns after retiring 4 inert parameters.",
            "Grounded columns and their sources: O/C/E/A/N from IPIP-FFM (N=603k); S_M/S_P/S_N "
            "from SD3 (N=17k); DASS_Dep/DASS_Anx/DASS_Str from DASS-21 (N=40k); alpha_R, "
            "beta_choice, phi_per, w_MB from the Daw two-step task (N=149); k_DD from a "
            "delay-discounting sample (N=476); a_inhib from SSRT (N=264); lambda_ineq from an "
            "ultimatum-game sample (N=111); m_t from PANAS (N=329).",
            "Proxy columns: theta_T from HCP EMOTION FACE_MEDIAN_RT (N=482); u_U from HCP "
            "GAMBLING PUNISH_PROP_LARGER (N=491); omega_social from a trust-game investment "
            "measure (N=113). Synthetic columns: eta_vol and gamma_sal, both literature priors.",
            "The four retired columns (lambda_norm, s_spite, tau_sens, kappa_ctx) are present in "
            "the engine's PARAM_NAMES list but never read by the calibrated decision loop; they "
            "are marked INERT and were removed from the copula in P6d step 1.",
        ],
    },
    {
        "id": "scenarios",
        "title": "Running a scenario",
        "intro": [
            "The Scenarios tab contains thirteen pre-designed experiments. Each is a named shock "
            "package with a documented magnitude, an intervention window, and a target population.",
            "Open any scenario to configure three numbers — how many agents, how many ticks, and "
            "the random seed — and click Run simulation. The dashboard runs the engine, compares "
            "the result against the five human benchmarks, and shows the PASS/FAIL table plus a "
            "time-series view of how behaviour evolved.",
            "The Scenarios tab is the fastest way to see the platform work. For a first run, "
            "choose Baseline (no shocks) or Resource Crisis (a large economic shock).",
        ],
        "tech": [
            "Scenarios run through POST /api/run/{name}. The endpoint builds a simulator with "
            "the scenario's shock list, loads the calibrated theta from p6c_results.json, and "
            "returns platform values, per-seed results, timeseries, and subgroup splits if the "
            "shock was targeted.",
            "Shock types: resource_scarcity (multiplies the per-agent reward modifier by "
            "1 - magnitude), distress_injection (adds magnitude to the baseline distress term), "
            "network_thinning (reduces the effective neighbour count by magnitude), and "
            "wealth_redistribution (pulls each agent's wealth toward the mean by magnitude per "
            "tick).",
            "Each run produces a run_id which is a SHA-256 hash of (copula_ops, N, seed, shocks). "
            "Because the hash is deterministic, an evicted run can be rebuilt byte-for-byte from "
            "its configuration; the 3D views use this to resurrect runs whose cache entry expired.",
        ],
    },
    {
        "id": "design",
        "title": "Designing an experiment",
        "intro": [
            "Design experiments is where you build your own study rather than pick a pre-built "
            "scenario. You compose three ingredients: a shift to the population's psychology, "
            "one or more interventions, and a measurement plan.",
            "Population psychology. Any of the seventeen primary traits can have its mean shifted "
            "(e.g. make everyone slightly more neurotic) or its standard deviation scaled (e.g. "
            "widen the spread of Machiavellianism). The change applies to the sampled agents "
            "before the simulation starts.",
            "Interventions. Four types, same as the pre-built scenarios, but here you choose the "
            "magnitude, the start tick, the duration, and the target — everyone, a random "
            "percentage, or a subgroup defined by a trait (e.g. the top 30% on Agreeableness).",
            "Measurement. Set N, T, and how many seeds to average across. Averaging across seeds "
            "gives a mean and a standard error the way a real experimental design does.",
        ],
        "tech": [
            "The design payload is posted to /api/design and consists of copula_ops (a list of "
            "{op, trait, value} operations: shift_mean, scale_sd, set, clip_pct), shocks, N, T, "
            "and seeds. The engine samples N agents from the copula, applies the copula_ops, "
            "resolves each shock's target_where clause to a boolean mask, and runs the simulator "
            "once per seed.",
            "When the design returns, the server also registers the exact agent matrix it used "
            "under a run_id derived from (copula_ops, N, first seed, shocks). This is what makes "
            "the 3D views and the joint-distribution scatter show the ACTUAL agents of your "
            "design rather than a fresh random sample.",
            "Subgroup effects are reported as a treatment-vs-control table (targeted agents "
            "versus non-targeted) with two windows: the full simulation and the intervention "
            "window only. The latter isolates the direct effect from any pre-existing group "
            "differences.",
        ],
    },
    {
        "id": "trace",
        "title": "Reading a decision trace",
        "intro": [
            "The trace is the platform's signature feature. Given one agent and one tick, it "
            "shows every internal number that contributed to the decision that was made.",
            "You reach it from any result card: click 'Trace a single decision' and choose an "
            "agent and a tick. The page shows a narrative of what happened, a flow diagram of "
            "the decision as it unfolded (habit values, plan values, blend weight, action, "
            "outcome), and a full table of the fifty-six fields that make up one tick.",
            "Every field is clickable: clicking a label opens a plain-English definition with "
            "the formula where useful. The population-context block below the trace tells you "
            "how rare interactions are in this engine so you can interpret the trace correctly.",
        ],
        "tech": [
            "The trace is emitted directly from the calibrated engine (p6c_v5.py) with "
            "trace_agent_i set. It captures, in order: stage-1 Q-values (habit, plan, "
            "arbitrated), the arbitration weight w_mb_eff and softmax temperature tau, the "
            "engage/avoid logits, the transition outcome s2, per-head chosen actions and "
            "Q-values for offer/send/accept/return, Fehr-Schmidt utility and its two inequity "
            "components, the four reward components, the partner's actions and parameters, and "
            "context (distress, baseline distress, wealth).",
            "Determinism is exact: same seed, same N, same T, same shocks produce a "
            "bit-identical trace. The canonical trace hash is reported in the audit block of "
            "every run.",
            "One caveat the platform is explicit about: the reward scale is not identical to "
            "the human-data scale in the ultimatum and trust games — rewards are clipped and "
            "modified by omega_social, salience, and mood. This is documented as "
            "[REWARD-SCALE-DRIFT] and appears in the flags list on every trace.",
        ],
    },
    {
        "id": "crowd",
        "title": "The crowd tab — inspecting the population",
        "intro": [
            "The Crowd tab is where you look at the population every experiment runs on. It shows you who is in the society, how traits are distributed, how they correlate, and how any subgroup differs from the rest.",
            "Pick a trait from the chip strip at the top. You will see its histogram, its descriptive statistics, and — crucially — a provenance panel that tells you where that column came from and how much weight it deserves. Green dots mean the trait is grounded directly in a human sample; amber means a defended proxy; red means a literature prior.",
            "Below the histogram is the joint distribution: choose any two traits and see where agents sit in that 2D space. If you have a filter active, the subgroup is highlighted in blue over the full population in grey.",
            "Below that is the factor structure panel — the confirmatory factor analyses that justify the bifactor Dark Triad and bifactor DASS structures the specification mandates.",
            "Then the APA-ready paragraphs card, which produces a methods paragraph and a results paragraph you can paste into a manuscript draft.",
            "Finally the filter bar. Apply a subgroup filter and the table below the filter compares that subgroup to the rest of the population across all seventeen traits, with Cohen's d and a 95% CI on the mean difference. This is the tool to use when you want to know what actually distinguishes a subgroup.",
        ],
        "tech": [
            "Provenance statuses come from the TRAIT_PROVENANCE table in dashboard.py and are served through /api/traits. The reference distribution is the P6d copula (32,000 agents x 26 columns after retiring four inert parameters).",
            "The subgroup analysis is served by /api/subgroup-analysis. The subgroup is compared against the complement of the filter mask (not the population including the subgroup), which is the standard independent-samples formulation. Cohen's d uses the pooled SD. The 95% CI on the mean difference uses a normal approximation: (m_sub - m_rest) +/- 1.96 * sqrt(s_sub^2/n_sub + s_rest^2/n_rest).",
            "At N = 32,000 the p-values are dominated by sample size and are not decision-relevant. The panel says this explicitly and directs you to Cohen's d instead. A Bonferroni threshold for the family of 17 tests is shown under the table if you want it.",
            "The joint distribution is served by /api/scatter2d. Pearson r is computed with a Fisher z 95% CI; Spearman rho is computed on average-tied ranks via scipy.stats.rankdata. Both are computed on the full columns, not on the display subsample.",
            "The factor structure panel reads from /api/factor-structure, whose payload is embedded in dashboard.py rather than read from work_g3/ or work_g4/, because the .app bundle ships work_env and work_g2 but not the CFA directories. Regenerate the payload by re-running cfa_sd3.py and cfa_dass.py and replacing the raw string.",
            "The APA paragraphs are generated client-side from the provenance cache and the last run record. They are a starting draft, not a substitute for writing the methods section yourself.",
        ],
    },
    {
        "id": "view3d",
        "title": "3D views — agents in trait space",
        "intro": [
            "The 3D views tab renders the actual agents of a specific run in three dimensions. You pick a run, choose which three traits form the axes, and rotate the cloud with the mouse. A fourth trait drives the colour.",
            "The list of runs is only the runs you have actually done. If a run has been evicted from the backend's in-memory cache, it is marked with a rebuild glyph and rebuilt on demand when you load it — the reconstruction is byte-identical because the engine is deterministic.",
            "Compare mode shows two runs side by side with shared rotation and shared z-scoring. When both runs used the same seed and N, they drew the same agents from the copula — the two clouds coincide dot-for-dot, and the difference is what happened to those agents, not who they were. The banner above the clouds states this explicitly.",
            "The agent profile panel shows a single agent as a radial bar chart, one bar per primary trait, z-scored against the run's own population.",
        ],
        "tech": [
            "The projection is orthographic with yaw-then-pitch rotation, implemented client-side in about twenty lines of JavaScript. No WebGL, no external libraries.",
            "The scatter is served by /api/scatter3d. Axes are z-scored against a reference run when one is provided so that two clouds share a coordinate system in compare mode. Colour scale is also shared.",
            "Agent coordinates are sampled from the run's cached agent matrix (up to max_points points, deterministic seed). The highlight agent is force-included in the sample. The radial profile is served by /api/agent-profile.",
            "Because /api/scatter3d and /api/agent-profile both look up a run_id in the backend's _RUN_CACHE (5-slot, cleared on restart), the frontend calls ensureRunLive() first. If the run has been evicted, ensureRunLive re-runs it from its stored configuration and adopts the returned id.",
        ],
    },
    {
        "id": "library",
        "title": "Library — saving and re-opening experiments",
        "intro": [
            "Every experiment you run is auto-saved to the Library. Each entry records the configuration (scenario or design, N, T, seed or seed list), the full result, and a timestamp.",
            "You can rename entries, duplicate a configuration to run again with a different seed, delete, and export the whole library as JSON.",
            "Selecting two entries and clicking Compare opens a paired table. If both used the same seed and N, the header reports a paired counterfactual — same agents, different conditions. Otherwise it reports an independent-samples comparison.",
        ],
        "tech": [
            "The library lives in localStorage under xai32k.session.v2. Auto-saved entries are capped at 40; manually named entries are never purged. A save within 3 s of the previous entry with the same name is skipped.",
            "The library is a client-side record only. The backend's _RUN_CACHE is separate and holds 5 runs at a time. Selecting a library entry re-runs or rebuilds as needed.",
            "Export produces a JSON array of entries; import merges new entries, de-duplicating on id.",
        ],
    },
    {
        "id": "compare",
        "title": "Compare — counterfactuals and controls",
        "intro": [
            "Compare is where the platform does what a real experiment cannot: two runs that used the same seed and the same N drew the same agents, and you can compare their outcomes directly.",
            "Select two or more scenarios, click Run comparison, and the dashboard runs each and lays the results side by side.",
            "The paired-versus-independent distinction matters statistically. A paired comparison (same seed and N) has much more power because the between-agent variance is held constant. An independent comparison does not have that advantage.",
        ],
        "tech": [
            "The comparison runs each scenario through POST /api/run/{name} with the same N, T, and seed, then draws a table with platform values and PASS/FAIL pills.",
            "In the Library comparison the per-benchmark z is computed as (B - A) / se_human. This is a rough significance indicator, not a paired test on the difference. The honest reading of the table is by effect size and by which benchmarks move, not by whether the z crosses a threshold.",
            "Multi-seed runs give you a per-seed SD, which is the correct variance to use for a paired test. Design experiments support up to 10 seeds.",
        ],
    },
    {
        "id": "report",
        "title": "Generating a scientific report",
        "intro": [
            "The report feature produces a self-contained printable document from any run: cover page, executive summary, full benchmark table, benchmark chart, time-series panels, delta-from-baseline table, population statistics, one or more decision traces, mechanism analysis, audit block, active flags, data provenance, and a citation list.",
            "You reach it from any completed run. The output is a single HTML file that opens in Safari or Preview and prints cleanly to PDF.",
            "The report is designed to be paste-ready for a supplementary appendix. Every page footer carries the run_id so a printed page can be traced back to the exact configuration that produced it.",
        ],
        "tech": [
            "The report is assembled entirely from data the app already holds: the run response, the provenance table, the audit block, and any traces that have been fetched during the session.",
            "Trace fields are drawn from the 56-field schema emitted by the calibrated engine. Each field is documented in the Appendix — Data dictionary section.",
            "Active flags propagated with every trace include [SYNTHETIC-PRIOR-PARAMS], [REWARD-SCALE-DRIFT], [COLLAPSED-LR], [ENVY-ONLY], and others; the report lists them verbatim so a reader can see the caveats without having to consult the source.",
        ],
    },
    {
        "id": "foundations",
        "title": "Scientific foundations",
        "intro": [
            "This platform rests on three empirical pillars: the Dark Triad bifactor structure, the DASS bifactor structure, and the recovery of learning and social-preference parameters from behavioural tasks.",
            "The Dark Triad is modelled as a general Dark Core plus three specific factors (Machiavellianism, Psychopathy, Narcissism). This structure replicates in the SD3 (N = 17,740) with a bifactor model outperforming three correlated factors by CFI +0.053.",
            "The DASS is modelled the same way: general distress plus three specifics. In the DASS-42 sample (N = 39,775) the bifactor model wins by CFI +0.034.",
            "Learning and social parameters are recovered from task data using hierarchical fitting: w_MB from the Daw two-step task (rho = 0.841, N = 149), k_DD from a delay-discounting sample (N = 476), a_inhib from SSRT (N = 264), lambda_ineq from an ultimatum sample (N = 111).",
            "Every parameter carries a documented reliability flag and a recovery metric. Two parameters — eta_vol and gamma_sal — remain synthetic priors and are flagged accordingly in every trace.",
        ],
        "tech": [
            "The bifactor CFAs were run with semopy maximum likelihood on the raw Likert responses treated as continuous. The strictly correct treatment for ordinal data is DWLS/WLSMV; this is documented as [CONTINUOUS-ML-CFA] and does not change the direction of the model comparison.",
            "The DASS CFA was run on DASS-42 rather than the specification's DASS-21. The bifactor structure replicates in both. The general-distress construction is unaffected.",
            "The reference implementation includes a model-reduction note: alpha_R+ and alpha_P- were collapsed into a single learning rate alpha_R because the seven-parameter model failed MCMC convergence. All downstream traces carry [COLLAPSED-LR].",
            "The Fehr-Schmidt model uses an envy-only simplification: alpha = D + lambda_ineq, beta = a proxy derived from Agreeableness. This is documented as [ENVY-ONLY] and [SIMPLIFIED MODEL] in every trace.",
            "The specification is at version 2.0 (PDF on disk) and v1.0 (markdown on disk). A cited v2.1 was never produced; the sources of truth are the PDF, the markdown, and the current code.",
        ],
    },
    {
        "id": "glossary",
        "title": "Glossary",
        "intro": [
            "Agent — one simulated person. The default run uses 8,000; the platform supports up to 32,000.",
            "Tick — one round of the simulation. At each tick every agent chooses a partner, decides whether to engage, plays both games with that partner, and updates its internal estimates.",
            "Q-value — an agent's estimate of the reward from taking a particular action. Higher Q means the agent expects more reward.",
            "Model-free value (Q_MF) — habit. The value the agent has learned from recent experience by trial and error.",
            "Model-based value (Q_MB) — plan. The value the agent computes by simulating the future using its model of how the environment behaves.",
            "Arbitration weight (w_mb_eff) — the blend between habit and plan. Near 0 means mostly habit, near 1 means mostly plan. Distress reduces this weight, which is the mechanism that models choking under pressure.",
            "Softmax — the function that turns Q-values into probabilities. Higher-valued actions are more likely to be chosen, but the choice is not deterministic.",
            "Fehr-Schmidt utility — the fairness-adjusted value of accepting an offer, subtracting penalties for disadvantageous and advantageous inequality.",
            "Distress — an internal state that rises when an agent's offers are rejected and decays slowly. High distress depresses planning.",
            "Inverse temperature (tau) — how sharply the softmax prefers the highest-Q action. High tau means the best option is almost always chosen; low tau means choices are more random.",
            "Gini coefficient — a single number from 0 (everyone equal) to 1 (one agent owns everything) summarising wealth inequality.",
            "Paired counterfactual — two runs with the same seed and N, hence the same agents. The only difference is the intervention. This is the comparison a real experiment cannot make.",
        ],
        "tech": [],
    },
    {
        "id": "references",
        "title": "References",
        "intro": [
            "The five human benchmarks against which every run is validated are drawn from the following published work.",
        ],
        "tech": [],
        "bullets": [
            "Fehr, E., & Schmidt, K. M. (1999). A theory of fairness, competition, and cooperation. Quarterly Journal of Economics, 114(3), 817-868.",
            "Oosterbeek, H., Sloof, R., & Van De Kuilen, G. (2004). Cultural differences in ultimatum game experiments: Evidence from a meta-analysis. Experimental Economics, 7(2), 171-188.",
            "Johnson, N. D., & Mislin, A. A. (2011). Trust games: A meta-analysis. Journal of Economic Psychology, 32(5), 865-889.",
            "Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. Neuron, 69(6), 1204-1215.",
            "Zanon, C., et al. (2020). Examining the dimensionality, reliability, and invariance of the DASS-21 across eight countries. Assessment, 28(2), 1-14.",
            "Knitter, L. A., Hoffmann, J., Eid, M., & Koch, T. (2025). Measuring the dark triad: A meta-analytical SEM study of two prominent short scales. Frontiers in Psychology, 15, 1469970.",
            "Khorshidi, S., & Mansouri Jamshidi, K. (2026). The Dark Triad as a Multi-Layered Structure: A Triangulation of Classical Psychometrics and Explainable Machine Learning. SSRN Electronic Journal.",
        ],
    },
]
