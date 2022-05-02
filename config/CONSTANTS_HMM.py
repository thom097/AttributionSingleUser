# ENVIRONMENT PARAMETERS
time = execution_duration = 30  # Days
lw = 2  # line width, for plots
N_users = 20000  # Set users' pool dimension. Notice that the total number of users is dynamical

########################################################################################################################
########################################################################################################################

# For HMM
HMM_TEST_EXECUTION = True
LR_EXPONENTIAL_DECAY = False
# This variable helps understand if the fit works or not. If TRUE the HMM is not hidden!
STATES_ARE_OBSERVABLE = False

# PARAMETERS TO FIT THE HIDDEN MARKOV MODEL
N_states = 3  # 4

# Parameters to fit the HMM
LEARNING_RATE = 1e-3
initial_learning_rate = 1e-2  # Only if LR_EXPONENTIAL_DECAY = True in config/execution_parameters
decay_steps = 10000  # Only if LR_ExponentialDecay = 1 in config/execution_parameters
decay_rate = 0.9  # Only if LR_ExponentialDecay = 1 in config/execution_parameters
EPOCHS = 1000
BATCH_SIZE = 500
basis = 1e-4

# Parameters to run HMM in test mode
# TODO: check this to be consistent with the N_camp from simulator
N_camp = 2

discount_factor = 0.8
p_exp = [0.5, 0.5, 0.4] # Probability of exposition to each campaign
N_exp_1 = 8000
N_exp_2 = 6000
#N_exp_3 = 1200

# Parameters to define Users behaviour. See Abishek paper
MU = [ -2.7419195, -1.1010361, -5.92728  ]
BETA = [0.7, -0.3, 0.15, 0.4, -0.4, 0.8]
INITIAL_STATE_PROB = [0.8] # Probability of entering the HMM in the first N_states-2 states. Must be same length of N_states
