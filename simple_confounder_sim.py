# %%
"""
SIMPLE CONFOUNDER SIMULATION
"""


import numpy as np


# %%

N = 10000000

C = np.random.binomial(1, 0.5, N)
X = np.random.binomial(1, np.where(C == 1, 0.8, 0.3), N)

pY = 0.1 + 0.5*X + 0.3*C

Y = np.random.binomial(1, pY, N)

# %%

PY_X1 = np.mean(Y[X == 1])
PY_X0 = np.mean(Y[X == 0])
est_effect_naive = PY_X1 - PY_X0
print(f"Estimated effect (naive): {est_effect_naive}")

py_x1_c1 = np.mean(Y[(X == 1) & (C == 1)])
py_x1_c0 = np.mean(Y[(X == 1) & (C == 0)])
py_x0_c1 = np.mean(Y[(X == 0) & (C == 1)])
py_x0_c0 = np.mean(Y[(X == 0) & (C == 0)])

est_eff_c0 = py_x1_c0 - py_x0_c0
est_eff_c1 = py_x1_c1 - py_x0_c1

num_C = np.sum(C)

est_effect_controlled = (est_eff_c1*num_C + est_eff_c0*(N - num_C))/N

print(f"Estimated effect (C controlled): {est_effect_controlled}")