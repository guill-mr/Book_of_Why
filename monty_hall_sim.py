# %%
"""
MONTY HALL SIMULATIONS
"""

import numpy as np



# %% Monty Hall simulation

""" LET'S MAKE A DEAL """

# Base case, where switching the chosen door is the correct answer.

N = 10000

SWITCH_STRAT = np.zeros(N)
MAINTAIN_STRAT = np.zeros(N)

for i in range(N):
    list_of_options = ["A", "B", "C"]

    CAR = np.random.choice(list_of_options)

    CHOICE_1 = np.random.choice(list_of_options)

    if CAR == CHOICE_1:
        list_of_options.remove(CAR)
    else:
        list_of_options.remove(CAR)
        list_of_options.remove(CHOICE_1)

    REVEAL = np.random.choice(list_of_options)
    
    list_of_options = ["A", "B", "C"]
    list_of_options.remove(REVEAL)
    list_of_options.remove(CHOICE_1)
    
    SWITCH = list_of_options[0]
    MAINTAIN = CHOICE_1

    SWITCH_STRAT[i] =  int(SWITCH == CAR)
    MAINTAIN_STRAT[i] = int(MAINTAIN == CAR)

print("Switching strategy win rate: ", np.mean(SWITCH_STRAT))
print("Maintaining strategy win rate: ", np.mean(MAINTAIN_STRAT))




# %% Monty Hall simulation with a twist

""" LET'S FAKE A DEAL """

# Pearl's proposed tweaking to understand when it's indifferent to switch or maintain the chosen door.

N = 10000

SWITCH_STRAT = np.zeros(N)
MAINTAIN_STRAT = np.zeros(N)
CAR_REVEALED = np.zeros(N)

for i in range(N):
    list_of_options = ["A", "B", "C"]

    CAR = np.random.choice(list_of_options)

    CHOICE_1 = np.random.choice(list_of_options)

    list_of_options.remove(CHOICE_1)

    REVEAL = np.random.choice(list_of_options)
    
    if REVEAL == CAR:
        SWITCH_STRAT[i] = 0
        MAINTAIN_STRAT[i] = 0
        CAR_REVEALED[i] = 1
    
    else:
        list_of_options.remove(REVEAL)
        
        SWITCH = list_of_options[0]
        MAINTAIN = CHOICE_1
        
        SWITCH_STRAT[i] =  int(SWITCH == CAR)
        MAINTAIN_STRAT[i] = int(MAINTAIN == CAR)
        CAR_REVEALED[i] = 0

print("Switching strategy win rate: ", np.mean(SWITCH_STRAT))
print("Maintaining strategy win rate: ", np.mean(MAINTAIN_STRAT))
print("Car revealed rate: ", np.mean(CAR_REVEALED))