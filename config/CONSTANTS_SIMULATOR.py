# ENVIRONMENT PARAMETERS
time = 60  # Days
lw = 2  # line width, for plots
N_users = 50  # Set users' pool dimension. Notice that the total number of users is dynamical

########################################################################################################################
########################################################################################################################

# Parameters to run the simulator
ANALYSIS_MODE = True
# TODO: check no debug_mode or analysis_mode
time_failure = 30  # Number of days after which we suppose a user is not interested
# Campaign properties
N_awn_camp = 3  # number of awareness campaigns
N_traff_camp = 2  # number of traffic campaigns
N_cnv_camp = 1  # number of conversion campaigns
N_camp = 6
#N_camp = N_awn_camp + N_traff_camp + N_cnv_camp  # number of campaigns
alpha = 0.7 # Saturation level for awareness campaigns
# Set number of expositions for each campaign
n_awareness_1 = 8
n_awareness_2 = 4
n_awareness_3 = 6
n_traffic_1 = 4
n_traffic_2 = 3
n_conversion_1 = 5
campaigns = {'n_awareness_1': 8, 'n_awareness_2': 4, 'n_awareness_3': 6,
             'n_traffic_1': 4, 'n_traffic_2': 3, 'n_conversion_1': 5}

# User Properties

# Define all the features considered and the Target Groups available
features = [["only_tg"]]
        #features = [["male", "female"],
        #            ["gamer", "no_gamer"]]
TG = [["only_tg"]]
        #TG = [["male", "gamer"], #NB every row of TG MUST be as long as feature vector. #### CHECK IT WORKS WITH EMPTY VALS
        #      ["male", "no_gamer"],
        #      ["female", "gamer"],
        #      ["female", "no_gamer"]]
# Define the average probability of exposition for each feature. The joint probability will be computed automatically
mean_prob_awn = [[0.7]]
        #np.array([[0.7, 0.4],   # mean prob of exposition of an awareness campaigns, feature 1
            #      [0.85, 0.3]])  # """""", feature 2
mean_prob_traff = [[0.5]]
        #np.array([(0.5, 0.25),
            #      (0.6, 0.15)])
mean_prob_cnv = [[0.35]]
        #np.array([(0.4, 0.2),
            #      (0.5, 0.1)])
# Define the variance for the computation of each campaign actual exposition probability
probability_variance = 0.05
# Define conversion thresholds for each Target Group
thresholds = [0.8] #[0.6, 0.8, 0.7, 0.9] # The higher is Prob_exp, the lower we choose the threshold

# Set the probabilities and result of clicking an awareness or a click campaign
p_click_awn = 0.05
reward_click_awn = 1.3
reward_noclick_awn = 0.95
p_click_traff = 0.4
reward_click_traff = 1.2
reward_noclick_traff = 0.85
users_actions_flag = 0  # Flag to activate possibility of users taking an external action
action_influence = 0.5
p_action = 0.1