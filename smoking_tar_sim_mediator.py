"""
MEDIATOR ANALYSIS SIMULATION
"""


import numpy as np

N = 1000000

G = np.random.binomial(1, 0.5, N) # Genetic factor
S = np.random.binomial(1, np.where(G == 1, 0.8, 0.3), N) # Smoking factor
T = np.random.binomial(1, np.where(S == 1, 0.999, 0.001), N) # Tar factor

pC = 0.1 + 0.5*T + 0.4*G # Probability of Cancer

C = np.random.binomial(1, pC, N) # Cancer factor

# Naive Estimate
PC_S1 = np.mean(C[S == 1])
PC_S0 = np.mean(C[S == 0])
est_effect_naive = PC_S1 - PC_S0
print(f"Estimated effect (naive): {est_effect_naive}")

pc_t1 = np.mean(C[T == 1])
pc_t0 = np.mean(C[T == 0])
print(f"Estimated effect (naive) of tar on cancer: {pc_t1 - pc_t0}")

print(f"\n")

# Omniscient Estimate (just controlling for the unobserved confounder)
pt_s1 = np.mean(T[S == 1])
pt_s0 = np.mean(T[S == 0])
est_eff_s_on_t = pt_s1 - pt_s0
print(f"Effect of Smoking on Tar: {est_eff_s_on_t}")

pc_t1_g1 = np.mean(C[(T == 1) & (G == 1)])
pc_t0_g1 = np.mean(C[(T == 0) & (G == 1)])
pc_t1_g0 = np.mean(C[(T == 1) & (G == 0)])
pc_t0_g0 = np.mean(C[(T == 0) & (G == 0)])
est_eff_g0 = pc_t1_g0 - pc_t0_g0
est_eff_g1 = pc_t1_g1 - pc_t0_g1
pg1 = np.mean(G)
pg0 = 1 - pg1
est_eff_t_on_c = est_eff_g1 * pg1 + est_eff_g0 * pg0
print(f"Effect of tar on cancer {est_eff_t_on_c}")

pc_s = est_eff_s_on_t * est_eff_t_on_c
print(f"Effect of smoking on cancer: {pc_s}")
print(f"\n")

# Front-Door Estimate
pt1_s1 = np.mean(T[S == 1])
pt0_s1 = 1 - pt1_s1
pt1_s0 = np.mean(T[S == 0])
pt0_s0 = 1 - pt1_s0

pt1 = np.mean(T)
pt0 = 1 - pt1
ps1 = np.mean(S)
ps0 = 1 - ps1

py_t1_s1 = np.mean(C[(T == 1) & (S == 1)])
py_t1_s0 = np.mean(C[(T == 1) & (S == 0)])
py_t0_s1 = np.mean(C[(T == 0) & (S == 1)])
py_t0_s0 = np.mean(C[(T == 0) & (S == 0)])

sum_s_t1 = py_t1_s1 * ps1 + py_t1_s0 * ps0
sum_s_t0 = py_t0_s1 * ps1 + py_t0_s0 * ps0

est_eff_t1 = (pt1_s1 - pt1_s0) * sum_s_t1
est_eff_t0 = (pt0_s1 - pt0_s0) * sum_s_t0

est_eff_s_on_c = est_eff_t1 + est_eff_t0

print(f"Estimated effect of Smoking on Cancer (front-door): {est_eff_s_on_c}")