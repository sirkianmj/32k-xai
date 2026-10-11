# xai_trace_v2.py — XAI trace for the calibrated engine (p6c_v5).
# Schema v3: 56 fields. Added is_mutual, partner_a1; renamed partner_reject
# to partner_a_acc for causal honesty (see notes below).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

from pathlib import Path
import importlib.util, sys, json
import numpy as np
import jax.numpy as jnp

_P6C_PATH = Path(__file__).parent / "p6c_v5.py"
_spec = importlib.util.spec_from_file_location("p6c", _P6C_PATH)
p6c = importlib.util.module_from_spec(_spec)
sys.modules["p6c"] = p6c
_spec.loader.exec_module(p6c)

TRACE_V2_KEYS = [
    # stage-1 self (19)
    "Q_mf_a0", "Q_mf_a1",
    "Q_MB_a0", "Q_MB_a1",
    "Q_net_a0", "Q_net_a1",
    "w_mb_eff", "tau",
    "engage_adv", "avoid_adv", "persev",
    "L1_a0", "L1_a1",
    "a1", "s2",
    "P_hat_a1_0", "P_hat_a1_1",
    "P_hat_alt_0", "P_hat_alt_1",
    # offer head (3)
    "a_off", "my_off", "Q_off_chosen",
    # send head (3)
    "a_send", "my_send", "Q_send_chosen",
    # accept head (6)
    "a_acc", "Q_acc_accept", "Q_acc_reject",
    "U_accept", "inequity_neg", "inequity_pos",
    # return head (3)
    "a_ret", "my_rm", "Q_ret_chosen",
    # reward decomposition (5)
    "r_off", "r_acc", "r_send", "r_ret", "reward_total",
    # partner view (11): order MUST match p6c_v5.step emission order
    "partner_id", "partner_alpha_e", "partner_beta_g",
    "partner_a_off", "partner_a_send", "partner_a_ret",
    "rec_off", "rec_send", "rec_rm",
    "partner_a_acc", "partner_a1",
    # events (3)
    "interacting", "rejected", "is_mutual",
    # context (3)
    "distress_pre", "baseline_dist", "wealth_pre",
]
TRACE_V2_DIM = len(TRACE_V2_KEYS)
assert TRACE_V2_DIM == 56, f"expected 56 trace fields, got {TRACE_V2_DIM}"

# ---- Active flags propagated with every trace (Spec 2.4 + Sec 7.2) ----
FLAGS_ACTIVE = [
    # Trace infrastructure (Phase 1)
    "[XAI-TRACE-CALIBRATED-ENGINE]",
    "[XAI-TRACE-56-FIELDS]",
    "[XAI-TRACE-DETERMINISTIC]",
    # Calibration results (P6c, verified across seeds 42-46)
    "[HUMAN-BENCHMARK-PASS-5-OF-5]",
    "[FEHR-SCHMIDT-CANONICAL]",
    "[PER-HEAD-Q]",
    "[KOOL-VALIDATION-PASSED]",
    # Documented limitations (Spec 7.2 vocabulary)
    "[REWARD-SCALE-DRIFT]",
    "[REWARD-SCALE-ASYMMETRY]",
    "[SIMPLIFIED MODEL]",
    "[ENVY-ONLY]",
    "[NOT-IDENTIFIABLE-IN-HCP-UGR]",
    "[SYNTHETIC-PRIOR-PARAMS]",
    "[COPULA-COLUMN-IS-LOG-K-DD]",
    "[KDE-TAIL-EXTRAPOLATION]",
    "[COLLAPSED-LR]",
    "[MF-GENERALIZED]",
    "[SYNTH-PRIOR-COEFF]",
    "[INERT-DEFAULT]",
    "[P6D-RANK-TRANSFORM-REJECTED]",
]

SYNTHETIC_PRIOR_NAMES = [
    "theta_T", "u_U", "eta_vol", "omega_social", "lambda_norm",
    "s_spite", "tau_sens", "gamma_sal", "kappa_ctx", "m_t",
]


DEFAULT_THETA_PATH = Path(__file__).parent / "p6c_results.json"

def load_calibrated_theta(path=None):
    p = Path(path) if path else DEFAULT_THETA_PATH
    d = json.load(open(p))["calib"]
    return {k: float(d[k]) for k in p6c.CALIB_NAMES}

def trace_agent(agent_i, N=32000, T=500, seed=42, theta=None, theta_path=None):
    if theta is None:
        theta = load_calibrated_theta(theta_path)
    theta_vec = jnp.array([theta[k] for k in p6c.CALIB_NAMES], jnp.float32)
    _, SIM_TRACED = p6c.build_simulator(N, T, trace_agent_i=int(agent_i))
    metrics, traces = SIM_TRACED(theta_vec, seed_val=int(seed))
    return {
        "agent_i": int(agent_i),
        "N": int(N), "T": int(T), "seed": int(seed),
        "keys": TRACE_V2_KEYS,
        "values": np.asarray(traces),
        "metrics": np.asarray(metrics),
        "theta": theta,
        "flags": list(FLAGS_ACTIVE),
        "synthetic_prior_params": list(SYNTHETIC_PRIOR_NAMES),
    }

def format_tick(tr, tick_idx):
    v = tr["values"][tick_idx]
    idx = {k: i for i, k in enumerate(TRACE_V2_KEYS)}
    lines = [f"  tick {tick_idx}  (agent {tr['agent_i']})"]
    for group, keys in [
        ("stage-1", ["Q_mf_a0","Q_mf_a1","Q_MB_a0","Q_MB_a1","Q_net_a0","Q_net_a1",
                     "w_mb_eff","a1","s2"]),
        ("offer",   ["a_off","my_off","Q_off_chosen"]),
        ("send",    ["a_send","my_send","Q_send_chosen"]),
        ("accept",  ["a_acc","U_accept","inequity_neg","inequity_pos",
                     "Q_acc_accept","Q_acc_reject"]),
        ("return",  ["a_ret","my_rm","Q_ret_chosen"]),
        ("rewards", ["r_off","r_acc","r_send","r_ret","reward_total"]),
        ("partner", ["partner_id","partner_alpha_e","partner_beta_g",
                     "partner_a_off","partner_a_send","partner_a_ret",
                     "rec_off","rec_send","rec_rm",
                     "partner_a_acc","partner_a1"]),
        ("events",  ["interacting","is_mutual","rejected"]),
        ("context", ["distress_pre","baseline_dist","wealth_pre"]),
    ]:
        lines.append(f"    {group:8s} " + " ".join(
            f"{k}={v[idx[k]]:.3f}" for k in keys))
    return "\n".join(lines)
