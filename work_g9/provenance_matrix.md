# G9 - Data Provenance Matrix

**Total parameters:** 31

**With empirical evidence:** 20

**Synthetic priors (no dataset):** 10

**Not identifiable:** 1

**Unaccounted:** 0

| Parameter | Construct | Work dir | Raw dataset | License / DUA | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| O | Openness | work_ipip | IPIP-FFM-data-8Nov2018 | Open Psychometrics public | score_ipip.py; 603,322 IPC=1 subjects |
| C | Conscientiousness | work_ipip | IPIP-FFM-data-8Nov2018 | Open Psychometrics public | score_ipip.py; 50-item Big Five |
| E | Extraversion | work_ipip | IPIP-FFM-data-8Nov2018 | Open Psychometrics public | score_ipip.py |
| A | Agreeableness | work_ipip | IPIP-FFM-data-8Nov2018 | Open Psychometrics public | score_ipip.py |
| N | Neuroticism | work_ipip | IPIP-FFM-data-8Nov2018 | Open Psychometrics public | score_ipip.py; anti-double-count with DASS subscale applies |
| D | General Dark Core | work_sd3 | SD3 | Open Psychometrics public | score_sd3.py + work_g3/cfa_sd3.py (bifactor) |
| S_M | Machiavellianism | work_sd3 | SD3 | Open Psychometrics public | score_sd3.py; alpha=0.861 |
| S_P | Psychopathy | work_sd3 | SD3 | Open Psychometrics public | score_sd3.py; alpha=0.798 |
| S_N | Narcissism | work_sd3 | SD3 | Open Psychometrics public | score_sd3.py; alpha=0.805 |
| general_distress | DASS general factor | work_g4 | XAI_Mental_Health_DASS | Open Psychometrics public | cfa_dass.py (bifactor); 39,775 subjects |
| DASS_Dep | DASS depression specific | work_dass | XAI_Mental_Health_DASS | Open Psychometrics public | score_dass.py; secondary modulator |
| DASS_Anx | DASS anxiety specific | work_dass | XAI_Mental_Health_DASS | Open Psychometrics public | score_dass.py; secondary modulator |
| DASS_Str | DASS stress specific | work_dass | XAI_Mental_Health_DASS | Open Psychometrics public | score_dass.py; secondary modulator |
| w_MB | Model-based arbitration | work_n500 | XAI_Markov_TwoStep_Data | Academic (Zorowitz & Niv 2021) | fit_and_recover.py; rho=0.7824 N=500 |
| alpha_R | Learning rate (collapsed) | work_n500 | XAI_Markov_TwoStep_Data | Academic | fit_and_recover.py; [COLLAPSED-LR] |
| beta_choice | Inverse temperature | work_n500 | XAI_Markov_TwoStep_Data | Academic | fit_and_recover.py; [COLLAPSED-LR] |
| phi_per | Perseveration | work_n500 | XAI_Markov_TwoStep_Data | Academic | fit_and_recover.py |
| k_DD | Delay discounting | work_dd | XAI_Delay_Discounting_Data | Unknown - Qualtrics export, no license stated | score_dd.py; [ADAPTIVE-DESIGN] [NO-REFIT] |
| a_inhib | SSRT (inhibitory control) | work_ssrt | XAI_Risk_Impulsivity_Data | HCP restricted DUA | score_ssrt.py; integration method per Verbruggen 2019 |
| lambda_ineq | Inequity aversion (alpha only) | work_fs | XAI_Social_Games_Data | HCP restricted DUA | score_fs.py; [SIMPLIFIED MODEL] [ENVY-ONLY] |
| theta_T | Threat sensitivity | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[GROUNDED-PROXY: HCP-EMOTION-FACE-MEDIAN-RT]` `[DIRECTION-ASSUMED]` |
| u_U | Uncertainty aversion | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[GROUNDED-PROXY: HCP-GAMBLING-PUNISH-PROP-LARGER]` `[DIRECTION-ASSUMED]` |
| eta_vol | Volatility update | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived |
| omega_social | Social value weight | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[GROUNDED-PROXY: TRUST-GAME-INVESTMENT]` `[SAME-SOURCE-AS-lambda_ineq]` |
| lambda_norm | Moral norm sensitivity | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[INERT-IN-CALIBRATED-ENGINE]` |
| s_spite | Spite sensitivity | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[INERT-IN-CALIBRATED-ENGINE]` |
| tau_sens | Sensory precision | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[INERT-IN-CALIBRATED-ENGINE]` |
| gamma_sal | Salience gain | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived |
| kappa_ctx | Context integration | - | - | NO SOURCE | [SYNTHETIC PRIOR] literature-derived `[INERT-IN-CALIBRATED-ENGINE]` |
| m_t | Mood/state | - | - | NO SOURCE | [SYNTHETIC PRIOR] AR(1) state model `[GROUNDED-DIRECT: PANAS-PRE]` `[STATE-TO-TRAIT-MAPPING]` |
| beta_FS | Fehr-Schmidt guilt | - | XAI_Social_Games_Data | NOT IDENTIFIABLE | [NOT-IDENTIFIABLE-IN-HCP-UGR] no advantageous-inequity offers |
