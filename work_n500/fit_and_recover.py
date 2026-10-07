import pandas as pd, numpy as np, time, os, sys
from scipy.optimize import minimize
from scipy.stats import spearmanr
from pathlib import Path
from joblib import Parallel, delayed

DATA = Path("XAI_Markov_TwoStep_Data/data.csv")
P_COMMON = 0.7
N_SIM    = 500
T_TRIALS = 201
N_JOBS   = -1

def neg_ll(p, a1, s2, a2, r):
    alpha, beta, persev, w = p
    if not (1e-6 < alpha < 0.999): return 1e10
    if not (0.01 < beta < 30.0):   return 1e10
    if not (-3.0 < persev < 3.0):  return 1e10
    if not (0.0 <= w <= 1.0):      return 1e10
    Q_mf, Q_td, last, ll = np.zeros(2), np.zeros((2,2)), -1, 0.0
    for t in range(len(a1)):
        m0, m1 = Q_td[0,:].max(), Q_td[1,:].max()
        Qnet0 = w*(P_COMMON*m0 + (1-P_COMMON)*m1) + (1-w)*Q_mf[0]
        Qnet1 = w*(P_COMMON*m1 + (1-P_COMMON)*m0) + (1-w)*Q_mf[1]
        L0 = beta*(Qnet0 + (persev if last==0 else 0.0))
        L1 = beta*(Qnet1 + (persev if last==1 else 0.0))
        m = max(L0, L1); e0, e1 = np.exp(L0-m), np.exp(L1-m)
        ll += np.log((e0 if a1[t]==0 else e1) / (e0+e1))
        L2a, L2b = beta*Q_td[s2[t],0], beta*Q_td[s2[t],1]
        m2 = max(L2a, L2b); e2a, e2b = np.exp(L2a-m2), np.exp(L2b-m2)
        ll += np.log((e2a if a2[t]==0 else e2b) / (e2a+e2b))
        Q_td[s2[t],a2[t]] += alpha*(r[t] - Q_td[s2[t],a2[t]])
        Q_mf[a1[t]]       += alpha*(r[t] - Q_mf[a1[t]])
        last = a1[t]
    return -ll

def fit(a1, s2, a2, r, seed):
    rng = np.random.default_rng(seed); best = None
    for _ in range(3):
        x0 = [rng.uniform(0.1,0.7), rng.uniform(1,10), rng.uniform(-1,1), rng.uniform(0,1)]
        res = minimize(neg_ll, x0, args=(a1,s2,a2,r), method="L-BFGS-B",
                       options={"maxiter":150})
        if best is None or res.fun < best.fun: best = res
    return best.x

def simulate(alpha, beta, persev, w, seed):
    rng = np.random.default_rng(seed)
    rp = np.array([[0.7,0.3],[0.3,0.7]])
    Q_mf, Q_td, last = np.zeros(2), np.zeros((2,2)), -1
    A1, S2, A2, R = [], [], [], []
    for _ in range(T_TRIALS):
        m0, m1 = Q_td[0,:].max(), Q_td[1,:].max()
        Qnet0 = w*(P_COMMON*m0 + (1-P_COMMON)*m1) + (1-w)*Q_mf[0]
        Qnet1 = w*(P_COMMON*m1 + (1-P_COMMON)*m0) + (1-w)*Q_mf[1]
        L0 = beta*(Qnet0 + (persev if last==0 else 0.0))
        L1 = beta*(Qnet1 + (persev if last==1 else 0.0))
        m = max(L0,L1); e0,e1 = np.exp(L0-m), np.exp(L1-m); Z = e0+e1
        a1 = 0 if rng.random() < e0/Z else 1
        s2 = a1 if rng.random() < P_COMMON else 1-a1
        L2a, L2b = beta*Q_td[s2,0], beta*Q_td[s2,1]
        m2 = max(L2a,L2b); e2a,e2b = np.exp(L2a-m2), np.exp(L2b-m2); Z2 = e2a+e2b
        a2 = 0 if rng.random() < e2a/Z2 else 1
        r  = int(rng.binomial(1, rp[s2,a2]))
        Q_td[s2,a2] += alpha*(r - Q_td[s2,a2])
        Q_mf[a1]     += alpha*(r - Q_mf[a1])
        last = a1
        A1.append(a1); S2.append(s2); A2.append(a2); R.append(r)
    return np.array(A1), np.array(S2), np.array(A2), np.array(R)

print(f"Data file: {DATA.resolve()}", flush=True)
print(f"Exists: {DATA.exists()}", flush=True)
df = pd.read_csv(DATA).sort_values(["subject","trial"]).reset_index(drop=True)
subs = sorted(df.subject.unique())
print(f"Subjects: {len(subs)}  Trials: {len(df)}", flush=True)

print("\n=== REAL DATA FIT ===", flush=True)
t0 = time.time()
trials = {sid: {
    "a1": df[df.subject==sid].state_1_choice.to_numpy(int),
    "s2": df[df.subject==sid].state_2.to_numpy(int),
    "a2": df[df.subject==sid].state_2_choice.to_numpy(int) - 2*df[df.subject==sid].state_2.to_numpy(int),
    "r":  df[df.subject==sid].outcome.to_numpy(int),
} for sid in subs}

results = Parallel(n_jobs=N_JOBS, verbose=0)(
    delayed(fit)(trials[sid]["a1"], trials[sid]["s2"], trials[sid]["a2"], trials[sid]["r"], i)
    for i, sid in enumerate(subs)
)
real = pd.DataFrame(results, columns=["alpha","beta","persev","w"])
real.insert(0, "subject", subs)
real.to_csv("w_mb_real.csv", index=False)
print(f"Done in {time.time()-t0:.1f}s", flush=True)
print(real[["alpha","beta","persev","w"]].describe().to_string(), flush=True)

print(f"\n=== RECOVERY ({N_SIM} simulated agents) ===", flush=True)
rng = np.random.default_rng(42)
true_w = rng.uniform(0.0, 1.0, N_SIM)
sim_trials = []
for i in range(N_SIM):
    a1, s2, a2, r = simulate(
        rng.uniform(0.15, 0.6), rng.uniform(1.5, 8.0), rng.uniform(-0.5, 0.5),
        true_w[i], seed=1000+i)
    sim_trials.append({"a1": a1, "s2": s2, "a2": a2, "r": r})

t0 = time.time()
rec = Parallel(n_jobs=N_JOBS)(
    delayed(fit)(tr["a1"], tr["s2"], tr["a2"], tr["r"], 2000+i)
    for i, tr in enumerate(sim_trials)
)
rec_w = np.array(rec)[:, 3]

rho, pval = spearmanr(true_w, rec_w)
bias = float(np.mean(rec_w - true_w))
rmse = float(np.sqrt(np.mean((rec_w - true_w)**2)))
with open("recovery.txt","w") as f:
    f.write(f"N_SIM={N_SIM}\nT={T_TRIALS}\nMODEL=ts_par4_MLE\nPARALLEL=joblib\n")
    f.write(f"rho={rho:.4f}\np={pval:.3e}\nbias={bias:.4f}\nrmse={rmse:.4f}\n")
    f.write(f"pass={'YES' if rho>0.6 else 'NO'}\n")
pd.DataFrame({"true_w": true_w, "rec_w": rec_w}).to_csv("recovery_scatter.csv", index=False)

print(f"\nRecovery done in {time.time()-t0:.1f}s", flush=True)
print("="*60, flush=True)
print(f"  Spearman rho = {rho:.4f}", flush=True)
print(f"  p-value      = {pval:.3e}", flush=True)
print(f"  bias         = {bias:.4f}", flush=True)
print(f"  rmse         = {rmse:.4f}", flush=True)
print(f"  PASS (rho>0.6): {'YES' if rho>0.6 else 'NO'}", flush=True)
print("="*60, flush=True)
