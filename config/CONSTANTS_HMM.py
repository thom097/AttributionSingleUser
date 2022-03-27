# ENVIRONMENT PARAMETERS
time = execution_duration = 30  # Days
lw = 2  # line width, for plots
N_users = 10000  # Set users' pool dimension. Notice that the total number of users is dynamical

########################################################################################################################
########################################################################################################################

# For HMM
HMM_TEST_EXECUTION = True
LR_EXPONENTIAL_DECAY = False
# This variable helps understand if the fit works or not. If TRUE the HMM is not hidden!
STATES_ARE_OBSERVABLE = True

# PARAMETERS TO FIT THE HIDDEN MARKOV MODEL
N_states = 3  # 4

# Parameters to fit the HMM
LEARNING_RATE = 1e-3
initial_learning_rate = 1e-2  # Only if LR_EXPONENTIAL_DECAY = True in config/execution_parameters
decay_steps = 10000  # Only if LR_ExponentialDecay = 1 in config/execution_parameters
decay_rate = 0.9  # Only if LR_ExponentialDecay = 1 in config/execution_parameters
EPOCHS = 100
BATCH_SIZE = 250
basis = 1e-4

# Parameters to run HMM in test mode
# TODO: check this to be consistent with the N_camp from simulator
N_camp = 1  #2

discount_factor = 1
p_exp = [0.6]  # , 0.5] # Probability of exposition to each campaign
N_exp_1 = 2000  # 800
# N_exp_2 = 500

# Parameters to define Users behaviour. See Abishek paper
MU = [ -4., -6.5, -1., -5.5 ]
BETA = [ 0.9, 0.55, 0.1, 0.65 ]
