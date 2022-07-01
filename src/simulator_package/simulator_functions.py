import numpy as np
import copy
from src.plot_and_print_info.plots_and_print_info import *


def joint_exposition_p(marginal_prob):
    '''
    This function computes the joint probability of exposition to a campaign, given the marginal ones
    :param marginal_prob: vector with all the marginal probabilities
    :return p_exp: a single joint probability computed through weight function w
    '''
    w = marginal_prob * marginal_prob - marginal_prob + 1  # weight function. Adjust if needed
    p_exp = (w * marginal_prob).sum() / w.sum()
    return p_exp


def influence_generator(exposition_probability):
    '''
    This function should compute the basis value of the influence of a campaign on the user
    :param exposition_probability: Probability for a user of being exposed to a campaign
    :return influence: Influence of the campaign on the user
    '''
    influence = exposition_probability  # TODO: COSA A CASO, PENSARE A COME
    return influence


def growth_function(old_influence):
    '''
    The discount function is denoted in the paper as Function A.
    The function returns the coefficient of the influence given by previous campaigns.
    :param old_influence: Influence given by previous exposition of the user to other campaigns
    :return coefficient: the multiplicative coefficient to take into account previous expositions
    '''
    coefficient = np.tanh(old_influence)+1 # TODO:  aggiusta con senso
    return coefficient


def discount_function(original_vector):
    '''
    The discount function is denoted in the paper as Function G.
    We take into account that a campaign has its maximum effect a few after the exposition. This function gives the
    temporal coefficient of this effect. Coeff = [ln(0.1t+1)+2.8]/[(0.8t-1)^2+1.8]
    :param original_vector: Vector with previous influences
    :return discounted_funnel_position: float with the influence of previous expositions adjusted in time
    '''
    t = np.linspace(len(original_vector) - 1, 0, len(original_vector))
    #discount_factors = (np.log(0.1 * t + 1) + 5) / (np.power(0.8 * t - 1, 2) + 4)
    discount_factors = np.power(0.9, t)
    discounted_funnel_position = original_vector * discount_factors
    discounted_funnel_position = np.sum(discounted_funnel_position)
    return discounted_funnel_position


def rescale_function(discounted_funnel_position):
    '''
    The rescale function is denoted in the paper as Function S.
    The function has the goal of rescaling the funnel position into the interval (0,1)
    :param discounted_funnel_position:
    :return:
    '''
    # TODO: set the coefficient in constant functions?
    rescaled_funnel_position = np.tanh(0.75 * discounted_funnel_position)
    return rescaled_funnel_position


# Function to generate campaign dictionaries
def generate_campaign(cm, name, target):
    '''
    Initialize the dictionary with all the information needed to define a campaign
    :param cm: constants manager
    :param name: Name of the campaign
    :param target: Target of the campaign, which can be 'awareness', 'traffic' or 'conversion'
    :return campaign: Dictionary with all the information
    '''
    campaign = {
        "name": name,
        "target": target,
    }

    if target == "awareness":
        prob_exposition = np.random.normal(cm['mean_prob_awn'], cm['probability_variance'])
        campaign["who_can_affect_me"] = []
    elif target == "traffic":
        prob_exposition = np.random.normal(cm['mean_prob_traff'], cm['probability_variance'])
        campaign["who_can_affect_me"] = range(cm['N_awn_camp'])
    elif target == "conversion":
        prob_exposition = np.random.normal(cm['mean_prob_cnv'], cm['probability_variance'])
        campaign["who_can_affect_me"] = range(cm['N_awn_camp'] + cm['N_traff_camp'])

    campaign["prob_exposition"] = []
    campaign["influence"] = []
    for tg in cm['TG']:  # TODO: NB THIS LOOP MUST BE CHANGED TO FACE MULTIPLE FEATURES AND SUBGROUPS OF FEATURES.. NOW PARTICULAR CASE

        idx = ([i for i in range(len(cm['features']))],  # take all the features
               [cm['features'][ii].index(tg[ii]) for ii in range(len(cm['features']))])
        # Set probability of exposition to each target group
        campaign["prob_exposition"].append(joint_exposition_p(prob_exposition[idx]))
        # Set campaign basic influence on target groups
        campaign["influence"].append(influence_generator(campaign["prob_exposition"][-1]))

    return campaign


def generate_empty_dictionaries(cm):
    '''
    This method generates initializes the dictionary for users and campaigns for a simulation.
    :param cm: constants manager
    :return user_list: a dictionary containing N_users initialized users for the simulation
    :return campaigns_list: a dictionary containing the initialized campaigns
    '''
    user_list = {}

    for ii in range(cm['N_users']):
        user_list[ii] = generate_user(cm, "user_" + str(ii + 1))

    campaigns_list = {}
    counter = 0
    for ii in range(cm['N_awn_camp']):
        campaigns_list[counter] = generate_campaign(cm,
            "camp_awn_" + str(ii + 1), "awareness")
        counter += 1

    for ii in range(cm['N_traff_camp']):
        campaigns_list[counter] = generate_campaign(cm,
            "camp_traff_" + str(ii + 1), "traffic")
        counter += 1

    for ii in range(cm['N_cnv_camp']):
        campaigns_list[counter] = generate_campaign(cm,
            "camp_cnv_" + str(ii + 1), "conversion")
        counter += 1

    return [user_list, campaigns_list]


def generate_user(cm, name):
    """
    The method creates an initialized user with all zeros values and the correct shapes.
    :param cm: constants manager
    :param name: Id of the user
    :return user: dictionary with information initialized for one user
    """
    user = {"name": name, "funnel_position": 0, "entering_time": 0, "history": np.zeros([cm['N_camp'], cm['time']]),
            "actions": np.zeros(cm['time']), "conversion": 0, "feat": []}

    for ii in range(len(cm['features'])):
        idx = np.round(np.random.uniform(0, len(cm['features'][ii]) - 1)).astype(int)
        user["feat"].append(cm['features'][ii][idx])
    user["target_group"] = cm['TG'].index(user["feat"])

    # Assign Conversion Threshold
    user["conv_threshold"] = cm['thresholds'][user["target_group"]]

    return user


def build_exposition_vector(cm):
    """
    The method is in charge of creating the vector with the expositions to assign
    :return exposition_vector: Vector with len=number of daily expositions
    """
    cont = 0
    exposition_vector = []
    for camp in cm['campaigns']:
        exposition_vector = np.append(exposition_vector, np.ones(cm['campaigns'][camp]) * cont)
        cont = cont + 1

    return exposition_vector.astype(int)


def simulate_response(cm, user, campaign):
    """
    For a user state and a campaign, the method decides whether the user had an interaction or not
    :param user: dictionary with the information on the user
    :param campaign: dictionary with the information of the campaign
    :return:
    """
    if campaign['target'] == "awareness":
        clk = np.random.binomial(1, cm['p_click_awn'])*(user['funnel_position']>0.4)
        #reward = (clk == 0) * reward_noclick_awn + (clk == 1) * reward_click_awn
        return 1+0.01*clk #reward
    elif campaign['target'] == "traffic":
        clk = np.random.binomial(1, cm['p_click_traff'])*(user['funnel_position']>0.4)
        #reward = (clk == 0) * reward_noclick_traff + (clk == 1) * reward_click_traff
        return 1+0.01*clk #reward
    elif campaign['target'] == "conversion":
        clk = np.random.binomial(1, cm['p_click_traff'])*(user['funnel_position']>0.4)
        #reward = (clk == 0) * reward_noclick_traff + (clk == 1) * reward_click_traff
        return 1+0.01*clk #reward
    else:
        return 1


class SimulationClass:
    """
    The SimulationClass is in charge of packing all the necessary methods to run a simulation and store its results.
    The attributes of the class are:
        - analysis_mode: type=bool, indicates whether we want to store all the information of the execution or just the
                                results;
        - user_list: type=dict, this dict has a fixed length, corresponding to the number of users we update at the same
                                time. Note that every time there is a conversion or more than 'time_failure' days has
                                passed, the corresponding user in user_list is resetted with a new one.
        - campaign_list: type=dict, this dict will store the information on the campaigns;
        - results: type=dict, in this attribute, the results of the simulation necessary to fit the model are stored.
                                This includes two main keys: 'user_expositions', with the data on which campaigns has
                                been shown to a user in a certain day, and 'user_outcome' with the response of the user.
        - simulation: type=dict, is an enhanced version of the results, where we keep track of ALL the users simulated
                                in the execution, and not only the active ones in user_list.
        - state: type=dict, stores the number of total users generated, and the number of conversions reached.
    """

    def __init__(self, cm, analysis_mode):
        self.analysis_mode = analysis_mode
        self.user_list, self.campaigns_list = generate_empty_dictionaries(cm)
        self.exposition_vector = build_exposition_vector(cm)
        self.results = {"user_expositions": np.empty([cm['N_camp'], cm['time_failure'], 0]), "user_outcome": np.empty([0, cm['time_failure']])}

        if analysis_mode:
            self.simulation = {"funnel_position_history": np.zeros((cm['N_users'], cm['time'])),
                               "inf_growth_history": np.zeros((cm['time'], cm['N_users'], cm['N_camp'] - cm['N_awn_camp'])),
                               "expositions": np.zeros((cm['time'], cm['N_users'], cm['N_camp'])), "conversions_users": {}}
        else:
            self.simulation = None

        self.state = {"tot_users": cm['N_users'], "conversions": 0}

    # TODO: remove simulation from functions when not needed
    # Simulate the campaign evolution
    def simulate(self, cm):
        """
        This is the core method of the class Simulation, in charge of simulating the evolution of the whole campaign.
        For every day, it assigns the expositions randomly to the users, and compute their evolution in the funnel.
        """

        # Reset history before simulation
        if 'flag' in globals():
            raise RuntimeError('You must reset the environment before running a new simulation!')

        # For each day of the campaign, simulate the environment
        for day in range(cm['time']):

            # Firstly, check whether some users must be reset or took an action
            for ii in range(cm['N_users']):
                elapsed_time = day - self.user_list[ii]["entering_time"]
                if elapsed_time >= cm['time_failure']:
                    self.reset_user(cm, ii, day, has_converted=False)

                if cm['users_actions_flag'] == 1:
                    self.action_check(cm, ii, day)

            # Shuffle the expositions to assign them randomly
            np.random.shuffle(self.exposition_vector)

            # Assign each exposition to a user, randomly
            for exposition in self.exposition_vector:
                flag = 0

                while flag != 1:
                    usr = round(np.random.uniform(0, cm['N_users'] - 1))

                    # Avoid bombing the same user
                    if sum(self.simulation["expositions"][day, usr, :]) > cm['max_daily_expositions']:
                        continue

                    if self.campaigns_list[exposition]["prob_exposition"][self.user_list[usr]["target_group"]] >= \
                            np.random.uniform(0, 1):
                        # TODO: metti un if campagna != awareness
                        # Compute the increase in awareness generated by the impression
                        delta_funnel_position = self.compute_delta_funnel_position(cm, day, usr, exposition)
                        self.user_list[usr]["history"][(exposition, day)] += delta_funnel_position
                        self.simulation["expositions"][day, usr, exposition] += 1

                        # If the campaign is a conversion campaign, check whether the user has converted or not
                        if self.campaigns_list[exposition]["target"] == "conversion":
                            rescaled_funnel_position = self.compute_rescaled_funnel_position(cm, day, usr)
                            self.user_list[usr]["funnel_position"] = rescaled_funnel_position
                            if rescaled_funnel_position >= self.user_list[usr]["conv_threshold"]:
                                self.save_conversion(cm, usr, day)

                        flag = 1

            for temp_usr in range(cm['N_users']):
                funnel_position = self.compute_rescaled_funnel_position(cm, day, temp_usr)
                self.user_list[temp_usr]['funnel_position'] = funnel_position
                if funnel_position > 0.4:
                    self.user_list[temp_usr]['actions'][day] = 1

            self.simulation["funnel_position_history"][:, day] = [usr['funnel_position'] for usr in self.user_list.values()]

        # Store all the information for the users that did not convert
        for ii in range(cm['N_users']):
            elapsed_time = day+1 - self.user_list[ii]["entering_time"]
            if elapsed_time == cm['execution_duration']:
                self.reset_user(cm, ii, day+1, has_converted=False)

    # Reset a user and create a new one
    def reset_user(self, cm, usr, day, has_converted):
        self.save_result(cm, usr, day, has_converted)
        self.state["tot_users"] += 1
        self.user_list[usr] = generate_user(cm, "user_" + str(self.state["tot_users"]))
        self.user_list[usr]["entering_time"] = day

    def save_result(self, cm, usr, day, has_converted):
        initial_time = self.user_list[usr]["entering_time"]
        outcome = np.zeros([1, cm['time_failure']])
        actions = self.user_list[usr]['actions'][initial_time:cm['time_failure']+initial_time].copy()
        outcome[0,:len(actions)] += actions
        # Since we reset the users that converted before checking action, we apply it here
        if has_converted:
            outcome[0, day] = 1
        # TODO: check if you start on initial_time or (-1/+1) and the same for day
        expositions = np.transpose(self.simulation["expositions"][initial_time:day+1, usr:1+usr, :], [2, 0, 1])

        if has_converted:
            # The conversion index is 2, as the value 1 is used to store expositions
            outcome[0,day+1-initial_time:] = 2
            expositions = np.append(expositions, np.zeros([cm['N_camp'], cm['time_failure']+initial_time-day-1, 1]), axis=1)

        self.results["user_outcome"] = np.append(self.results["user_outcome"], outcome, axis=0)
        self.results["user_expositions"] = np.append(self.results["user_expositions"], expositions, axis=2)

    # This function simulates possible actions of a user, in case this option is enabled (See users_actions_flag)
    # TODO: Check this function, now it doesn't work...
    def action_check(self, cm, usr, day):
        # Compute the current position in the funnel. Probability of taking an action is proportional
        funnel_position = self.compute_rescaled_funnel_position(cm, day, usr)
        prob_action = cm['p_action'] * funnel_position

        # Sample the user behaviour.
        action = np.random.binomial(1, prob_action)

        if action == 1:
            self.user_list[usr]["actions"][day] = cm['action_influence']
            if self.compute_rescaled_funnel_position(cm, day, usr) > self.user_list[usr]["conv_threshold"]:
                self.save_conversion(cm, usr, day)

        # TODO: ricalcola con fattore moltiplicativo la posizione nel funnel
        # controlla threshold
        # se superiore, conversione esterna

    # Delta funnel_position function
    def compute_delta_funnel_position(self, cm, t, user, campaign):
        """
        This method is in charge of computing the effect of an exposition on the user. It considers both the single
        effect of the campaign itself, and the increase due to previous expositions to other campaigns
        :param cm: constants manager
        :param t: time of the simulation
        :param user: user exposed
        :param campaign: advertising campaign showed to the user
        :return delta_funnel_position: increase in the funnel position
        """
        # Basic influence of the campaign
        influence = self.campaigns_list[campaign]["influence"][self.user_list[user]["target_group"]]

        # Coefficient for response
        R_coefficient = simulate_response(cm, self.user_list[user], self.campaigns_list[campaign])

        # If there was a click, save it
        #if R_coefficient>1:
        #    self.user_list[user]['actions'][t] = 1

        # Coefficient due to old expositions
        hist = self.user_list[user]["history"][self.campaigns_list[campaign]["who_can_affect_me"], 0:t]
        daily_hist = np.sum(hist, axis=0)
        discounted_hist = discount_function(daily_hist)
        inf_growth = growth_function(discounted_hist) if self.user_list[user]['funnel_position']>0.4 else 1.

        if self.analysis_mode & (self.campaigns_list[campaign]["target"] != "awareness"):
            self.simulation["inf_growth_history"][t, user, campaign - cm['N_awn_camp']] = inf_growth

        # The delta in the funnel position is given by the base influence of the campaign times increasing factors
        delta_funnel_position = influence * R_coefficient * inf_growth

        return delta_funnel_position

    # Rescaled funnel_position function
    def compute_rescaled_funnel_position(self, cm, t, user):
        """
        This method rescale the current position in the funnel of the user in an interval from zero to one
        :param cm: constants manager
        :param t: day of the simulation
        :param user: user for which it is necessary to compute the update
        :return rescaled_funnel_position: position of the funnel rescaled in [0,1]
        """

        # Compute increase due to awareness and conversion campaigns
        daily_funnel_position_awn = np.sum(self.user_list[user]["history"][:cm['N_awn_camp'], :t + 1], axis=0)
        daily_funnel_position_traff_cnv = np.sum(self.user_list[user]["history"][cm['N_awn_camp']:, :t + 1], axis=0)

        # Apply the discount function
        discounted_funnel_position_awn = discount_function(daily_funnel_position_awn)
        discounted_funnel_position_traff_cnv = discount_function(daily_funnel_position_traff_cnv)

        # Include actions in the computation
        discounted_action_increase = 0 #discount_function(self.user_list[user]["actions"][:t + 1])

        # Compute rescaled funnel position
        rescaled_funnel_position = cm['alpha'] * rescale_function(
            discounted_funnel_position_awn + cm['alpha'] * discounted_action_increase) + \
                                   (1 - cm['alpha']) * rescale_function(
            discounted_funnel_position_traff_cnv + (1 - cm['alpha']) * discounted_action_increase)

        return rescaled_funnel_position

    # Update the environment once a user reaches conversion
    def save_conversion(self, cm, usr, day):
        if self.analysis_mode:
            self.simulation[self.user_list[usr]["name"]] = day
            self.simulation["conversions_users"][self.state["conversions"]] = copy.deepcopy(self.user_list[usr])  # Save this user
            self.simulation["conversions_users"][self.state["conversions"]]["exit_time"] = day  # Store also exit time
            # Compute user's history for partial plot
            self.simulation["conversions_users"][self.state["conversions"]]["funnel_history"] = \
                [self.compute_rescaled_funnel_position(cm, temp_day, usr) for temp_day in
                 range(self.user_list[usr]["entering_time"], day + 1)]

            self.simulation["conversions_users"][self.state["conversions"]]["expositions"] = \
                self.simulation["expositions"][self.user_list[usr]["entering_time"]:day + 1, usr, :].copy()

        # user_list[usr]["conversion"]=1 # TODO: da sistemare viene resettato subito
        self.state["conversions"] += 1
        self.reset_user(cm, usr, day, has_converted=True)

    # Plot results of the simulation
    def plot_results(self, cm):
        if self.analysis_mode:
            set_matplotlib_properties()
            plot_all_funnel_positions(cm, self)
            plot_conversion_paths(cm, simulation=self)



