# ENVIRONMENT PARAMETERS
time = 30  # Days
lw = 2  # line width, for plots
N_users = 20000  # Set users' pool dimension. Notice that the total number of users is dynamical

########################################################################################################################
########################################################################################################################

# Parameters to run the simulator
ANALYSIS_MODE = True
# TODO: check no debug_mode or analysis_mode
execution_duration = time_failure = 30  # Number of days after which we suppose a user is not interested
# Campaign properties
N_awn_camp = 1  # number of awareness campaigns
N_traff_camp = 1  # number of traffic campaigns
N_cnv_camp = 1  # number of conversion campaigns
N_camp = N_awn_camp + N_traff_camp + N_cnv_camp  # number of campaigns
alpha = 0.5  # Saturation level for awareness campaigns
# Set number of expositions for each campaign
campaigns = {'n_awareness_1': 3500, 'n_traffic_1': 3500, 'n_conversion_1': 3500}

# User Properties

# Define all the features considered and the Target Groups available
features = [["only_tg"]]
TG = [["only_tg"]]
# Define the average probability of exposition for each feature. The joint probability will be computed automatically
mean_prob_awn = [[0.6]]
mean_prob_traff = [[0.45]]
mean_prob_cnv = [[0.3]]
# Define the variance for the computation of each campaign actual exposition probability
probability_variance = 0.05
# Define conversion thresholds for each Target Group
thresholds = [0.8]

# Set the probabilities and result of clicking an awareness or a click campaign
p_click_awn = 0.2
reward_click_awn = 1.3
reward_noclick_awn = 0.95
p_click_traff = 0.2
reward_click_traff = 1.2
reward_noclick_traff = 0.85

users_actions_flag = 0  # Flag to activate possibility of users taking an external action

# Parameters to fit HMM
discount_factor = 0.8

# For HMM
LR_EXPONENTIAL_DECAY = False
# This variable helps understand if the fit works or not. If TRUE the HMM is not hidden!
STATES_ARE_OBSERVABLE = False

# PARAMETERS TO FIT THE HIDDEN MARKOV MODEL
N_states = 3

# Parameters to fit the HMM
LEARNING_RATE = 1e-3
initial_learning_rate = 1e-2  # Only if LR_EXPONENTIAL_DECAY = True in config/execution_parameters
decay_steps = 10000  # Only if LR_ExponentialDecay = 1 in config/execution_parameters
decay_rate = 0.9  # Only if LR_ExponentialDecay = 1 in config/execution_parameters
EPOCHS = 100
BATCH_SIZE = 500
basis = 1e-4