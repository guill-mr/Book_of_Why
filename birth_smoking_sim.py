# %%
"""
EFFECTS OF SMOKING ON BIRTH WEIGHT AND MORTALITY (SIMULATION)
"""

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.special import expit

np.random.seed(168)

# %% First simulation

""" FIRST SIMULATION """

N = 100000
Ws, pMs, Ms = np.zeros(N), np.zeros(N), np.zeros(N)

for i in range(N):
    S = np.random.binomial(1, 0.5)
    D = np.random.binomial(1, 0.01)

    W = np.random.normal(3000, 200) - 200*S - 600*D
    while W < 2387:
        W = np.random.normal(3000, 200) - 200*S - 600*D
    while W > 3700:
        W = np.random.normal(3000, 200) - 200*S - 600*D

    pM = 0.01 + 1 / ((-88200/114) + (37/114) * W)

    M = np.random.binomial(1, pM) or (np.random.binomial(1, 0.05) * S) or (np.random.binomial(1, 0.7) * D)

    Ws[i] = W
    pMs[i] = pM
    Ms[i] = M

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(Ws, bins=30)
plt.title("Histogram of Ws")

plt.subplot(1, 3, 2)
plt.hist(pMs, bins=30)
plt.title("Histogram of pMs")

plt.subplot(1, 3, 3)
plt.hist(Ms, bins=2)
plt.title("Histogram of Ms")

plt.tight_layout()
plt.show()

plt.close


# %% Second simulation

""" SECOND SIMULATION """

N = 100000
Ss, Ds, Ws, pMs, Ms = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

for i in range(N):
    S = np.random.binomial(1, 0.5)
    
    if S == 1:
        D = np.random.binomial(1, 0.01)
    else:
        D = np.random.binomial(1, 0.1)

    W = np.random.normal(3000, 200) - 200*S - 600*D
    
    pM = 0.85 - 0.00025*W + 0.2*S + 0.8*D
    
    if pM > 1:
        pM = 1
    if pM < 0:
        pM = 0

    M = np.random.binomial(1, pM)

    Ss[i] = S
    Ds[i] = D
    Ws[i] = W
    pMs[i] = pM
    Ms[i] = M

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(Ws, bins=30)
plt.title("Histogram of Ws")

plt.subplot(1, 3, 2)
plt.hist(pMs, bins=30)
plt.title("Histogram of pMs")

plt.subplot(1, 3, 3)
plt.hist(Ms, bins=2)
plt.title("Histogram of Ms")

plt.tight_layout()
plt.show()

plt.close

# NAIVE ANALYSIS:

# -- Effect of Smoking on Weight -- 
# We can estimate the effect of smoking on weight by comparing the average weight of the two groups.

mean_weight_smoking = np.mean(Ws[Ss == 1])
mean_weight_nonsmoking = np.mean(Ws[Ss == 0])
estimated_effect_smoking = mean_weight_smoking - mean_weight_nonsmoking
print(f"Estimated effect of smoking on weight: {estimated_effect_smoking:.2f} grams")

# In this case we do have an accurate estimate of the effect of smoking on weight due to the fact that there are
# no confounders in the specific part of the graph.

# -- Effect of Birth Defect on Weight --
# We can estimate the effect of birth defects on weight in a similar way.

mean_weight_defect = np.mean(Ws[Ds == 1])
mean_weight_nodefect = np.mean(Ws[Ds == 0])
estimated_effect_defect = mean_weight_defect - mean_weight_nodefect
print(f"Estimated effect of birth defect on weight: {estimated_effect_defect:.2f} grams")

# And in this case we also get an accurate estimate of the effect of birth defects on weight for the same reason.

# -- Effect of Birth Weight on Mortality --

# For a naive analysis, we would simply use a logistic regression model to estimate the effects of Birth Weight on Mortality.

# We can use the statsmodels library to fit a logistic regression model to the data.
# We will include the standardized weight as a predictor for the mortality outcome.

# Standardize the weight
print(np.mean(Ws), np.std(Ws))
W_std = (Ws - np.mean(Ws)) / np.std(Ws)

# Fit the model only to W_std to predict Ms
X = sm.add_constant(W_std)
logit_model = sm.Logit(Ms, X)
result = logit_model.fit(disp=0)

print(result.summary())

# -- Effect of Smoking on Mortality -- 

# We can estimate the effect of smoking on mortality by comparing the mortality rates of the two groups.

mortality_smoking = np.mean(Ms[Ss == 1])
mortality_nonsmoking = np.mean(Ms[Ss == 0])
estimated_effect_smoking_mortality = mortality_smoking - mortality_nonsmoking
print(f"Estimated effect of smoking on mortality (naive): {estimated_effect_smoking_mortality:.4f}")


# CORRECT ANALYSIS:

# -- Effect of Smoking on Mortality -- 

# We can estimate the effect of smoking on mortality by comparing the mortality rates of the two groups
# while controlling for the effect of birth defects.

M_smoking_defect = np.mean(Ms[(Ss == 1) & (Ds == 1)])
M_nonS_nonD = np.mean(Ms[(Ss == 0) & (Ds == 0)])
M_smoking_nonD = np.mean(Ms[(Ss == 1) & (Ds == 0)])
M_nonS_defect = np.mean(Ms[(Ss == 0) & (Ds == 1)])

est_effect_defect = M_smoking_defect - M_nonS_defect
est_effect_nonD = M_smoking_nonD - M_nonS_nonD

num_defect = np.sum(Ds)

avg_effect_smoking = est_effect_defect * num_defect / N + est_effect_nonD * (1 - num_defect / N)

print(f"Estimated effect of smoking on mortality (controlling for D): {avg_effect_smoking:.4f}")


# %% Third simulation

""" THIRD SIMULATION """

# In this case we treat weight as a binary variable, just considering whether
# the child is underweight or not.

N = 1000000
Ss, Ds, Ws, pMs, Ms = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

for i in range(N):
    S = np.random.binomial(1, 0.5)
    D = np.random.binomial(1, 0.1)
    
    # if S == 1:
    #     D = np.random.binomial(1, 0.01)
    # else:
    #     D = np.random.binomial(1, 0.1)
    
    if S == 1 or D == 1:
        W = np.random.binomial(1, 0.6)
    else:
        W = np.random.binomial(1, 0.01)
    
    pM = 0.01 + 0.09*W + 0.1*S + 0.8*D
    
    if pM > 1:
        pM = 1
    elif pM < 0:
        pM = 0

    M = np.random.binomial(1, pM)

    Ss[i] = S
    Ds[i] = D
    Ws[i] = W
    pMs[i] = pM
    Ms[i] = M

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(Ws, bins=30)
plt.title("Histogram of Ws")

plt.subplot(1, 3, 2)
plt.hist(pMs, bins=30)
plt.title("Histogram of pMs")

plt.subplot(1, 3, 3)
plt.hist(Ms, bins=2)
plt.title("Histogram of Ms")

plt.tight_layout()
plt.show()

plt.close


# NAIVE ANALYSIS:

# -- Effect of Birth Weight on Mortality --

mean_mort_weight = np.mean(Ms[Ws == 1])
mean_mort_noweight = np.mean(Ms[Ws == 0])
estimated_effect_weight_mortality = mean_mort_weight - mean_mort_noweight
print(f"Estimated effect of birth weight on mortality(naive): {estimated_effect_weight_mortality:.4f}")

# -- Effect of Smoking on Mortality -- 

# We can estimate the effect of smoking on mortality by comparing the mortality rates of the two groups.

mortality_smoking = np.mean(Ms[Ss == 1])
mortality_nonsmoking = np.mean(Ms[Ss == 0])
estimated_effect_smoking_mortality = mortality_smoking - mortality_nonsmoking
print(f"Estimated effect of smoking on mortality (naive): {estimated_effect_smoking_mortality:.4f}")

# -- Effect of Smoking on Mortality when analyzing low birth weight cases --

mortality_smoking_low_weight = np.mean(Ms[(Ss == 1) & (Ws == 1)])
mortality_nonsmoking_low_weight = np.mean(Ms[(Ss == 0) & (Ws == 1)])
estimated_effect_smoking_mortality_low_weight = mortality_smoking_low_weight - mortality_nonsmoking_low_weight
print(f"Estimated effect of smoking on mortality (low weight cases): {estimated_effect_smoking_mortality_low_weight:.4f}")



# CORRECT ANALYSIS:

# -- Effect of Birth Weight on Mortality --

# We use the regression model to estimate it:
#   y = α + β1 * W + β2 * S + β3 * D

X = np.column_stack((Ws, Ss, Ds))
X = sm.add_constant(X)

linear_model = sm.OLS(Ms, X)
result = linear_model.fit()

print(f"Estimated effect of birth weight on mortality: {result._results.params[1]:.4f}")


# -- Effect of Smoking on Mortality -- 

# We can estimate the effect of smoking on mortality by comparing the mortality rates of the two groups
# while controlling for the effect of birth defects.

M_smoking_defect = np.mean(Ms[(Ss == 1) & (Ds == 1)])
M_nonS_nonD = np.mean(Ms[(Ss == 0) & (Ds == 0)])
M_smoking_nonD = np.mean(Ms[(Ss == 1) & (Ds == 0)])
M_nonS_defect = np.mean(Ms[(Ss == 0) & (Ds == 1)])

est_effect_defect = M_smoking_defect - M_nonS_defect
est_effect_nonD = M_smoking_nonD - M_nonS_nonD

num_defect = np.sum(Ds)

avg_effect_smoking = est_effect_defect * num_defect / N + est_effect_nonD * (1 - num_defect / N)

print(f"Estimated effect of smoking on mortality (controlling for D): {avg_effect_smoking:.4f}")


# We also estimate it by using the regression model:
#   y = α + β2 * S + β3 * D

X = np.column_stack((Ss, Ds))
X = sm.add_constant(X)

linear_model = sm.OLS(Ms, X)
result = linear_model.fit()

print(f"Estimated effect of smoking on mortality: {result._results.params[1]:.4f}")
