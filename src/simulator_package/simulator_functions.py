from src.plot_and_print_info.plots_and_print_info import *
import numpy as np
from config.CONSTANTS_HMM import *


# Joint Exposition Function
def joint_exposition_p(marginal_prob):
    # INPUT: marginal_prob: vector with all the marginal probabilities
    # OUTPUT: p_exp: a single joint probability computed through weight function w

    w = marginal_prob * marginal_prob - marginal_prob + 1  # weight function. Adjust if needed
    p_exp = (w * marginal_prob).sum() / w.sum()

    return p_exp


# Influence generator function from exposition probability
def influence_generator(exposition_probability):
    influence = exposition_probability / 1.5  # TODO: COSA A CASO, PENSARE A COME

    return influence


# Function A: growth function for old influence...TODO:  aggiusta con senso
def growth_function( old_influence ):
    coefficient = np.tanh(old_influence)+1

    return coefficient


# Function G: discount in time... TODO: aggiustare un po, ha picco troppo alto
def discount_function(original_vector):
    t = np.linspace(len(original_vector) - 1, 0, len(original_vector))
    discount_factors = (np.log(0.1 * t + 1) + 2.8) / (np.power(0.8 * t - 1, 2) + 1.8)

    discounted_funnel_position = original_vector * discount_factors
    discounted_funnel_position = np.sum(discounted_funnel_position)

    return discounted_funnel_position


# Function S: rescale discounted funnel_position in [0,1]
# TODO: set the coefficient in constant functions?
def rescale_function(discounted_funnel_position):
    rescaled_funnel_position = np.tanh(0.75 * discounted_funnel_position)

    return rescaled_funnel_position


# Function to generate campaign dictionaries
def generate_campaign(name, target):
    campaign = {
        "name": name,
        "target": target,
    }

    if target == "awareness":
        prob_exposition = np.random.normal(mean_prob_awn, probability_variance)
        campaign["who_can_affect_me"] = []
    elif target == "traffic":
        prob_exposition = np.random.normal(mean_prob_traff, probability_variance)
        campaign["who_can_affect_me"] = range(N_awn_camp)
    elif target == "conversion":
        prob_exposition = np.random.normal(mean_prob_cnv, probability_variance)
        campaign["who_can_affect_me"] = range(N_awn_camp + N_traff_camp)

    campaign["prob_exposition"] = []
    campaign["influence"] = []
    for tg in TG:  # TODO: NB THIS LOOP MUST BE CHANGED TO FACE MULTIPLE FEATURES AND SUBGROUPS OF FEATURES.. NOW PARTICULAR CASE

        idx = ([i for i in range(len(features))],  # take all the features
               [features[ii].index(tg[ii]) for ii in range(len(features))])
        # Set probability of exposition to each target group
        campaign["prob_exposition"].append(joint_exposition_p(prob_exposition[idx]))
        # Set campaign basic influence on target groups
        campaign["influence"].append(influence_generator(campaign["prob_exposition"][-1]))

    return campaign


def generate_empty_dictionaries():
    user_list = {}

    for ii in range(N_users):
        user_list[ii] = generate_user("user_" + str(ii + 1))

    campaigns_list = {}
    counter = 0
    for ii in range(N_awn_camp):
        campaigns_list[counter] = generate_campaign(
            "camp_awn_" + str(ii + 1), "awareness")
        counter += 1

    for ii in range(N_traff_camp):
        campaigns_list[counter] = generate_campaign(
            "camp_traff_" + str(ii + 1), "traffic")
        counter += 1

    for ii in range(N_cnv_camp):
        campaigns_list[counter] = generate_campaign(
            "camp_cnv_" + str(ii + 1), "conversion")
        counter += 1

    return [user_list, campaigns_list]


# Function to define User dictionaries
def generate_user(name):
    user = {"name": name, "funnel_position": 0, "entering_time": 0, "history": np.zeros([N_camp, time]),
            "actions": np.zeros(time), "conversion": 0, "feat": []}

    for ii in range(len(features)):
        idx = np.round(np.random.uniform(0, len(features[ii]) - 1)).astype(int)
        user["feat"].append(features[ii][idx])
    user["target_group"] = TG.index(user["feat"])

    # Assign Conversion Threshold
    user["conv_threshold"] = thresholds[user["target_group"]]

    return user


# Given the campaigns and number of expositions, create a list
def build_exposition_vector():

    cont = 0
    exposition_vector = []
    for camp in campaigns:
        exposition_vector = np.append(exposition_vector, np.ones(campaigns[camp]) * cont)
        cont = cont + 1

    return exposition_vector.astype(int)


# If the user is exposed to an awareness or traffic campaign, then this function computes the contribute of a click/no-click action
def simulate_response(target):
    if target == "awareness":
        clk = np.random.binomial(1, p_click_awn)
        reward = (clk == 0) * reward_noclick_awn + (clk == 1) * reward_click_awn
        return reward
    elif target == "traffic":
        clk = np.random.binomial(1, p_click_traff)
        reward = (clk == 0) * reward_noclick_traff + (clk == 1) * reward_click_traff
        return reward
    else:
        return 1


class SimulationClass:

    def __init__(self, analysis_mode):

        self.analysis_mode = analysis_mode
        self.user_list, self.campaigns_list = generate_empty_dictionaries()
        self.exposition_vector = build_exposition_vector()

        if analysis_mode:
            self.simulation = {"funnel_position_history": np.zeros((N_users, time)),
                               "inf_growth_history": np.zeros((time, N_users, N_camp - N_awn_camp)),
                               "expositions": np.zeros((time, N_users, N_camp)), "conversions_users": {}}
        else:
            self.simulation = None

        self.state = {"tot_users": N_users, "conversions": 0}

    # TODO: remove simulation from functions when not needed
    # Simulate the campaign evolution
    def simulate(self):

        # Reset history before simulation TODO: check that this actually works
        if 'flag' in globals():
            raise RuntimeError('You must reset the environment before running a new simulation!')

        for day in range(time):

            for ii in range(N_users):
                elapsed_time = day - self.user_list[ii]["entering_time"]
                if elapsed_time > time_failure:
                    self.reset_user(ii, day)

                if users_actions_flag == 1:
                    self.action_check(ii, day)

            np.random.shuffle(self.exposition_vector)  # Get a random order of the expositions to allocate

            for exposition in self.exposition_vector:
                flag = 0

                while flag != 1:
                    usr = round(np.random.uniform(0, N_users - 1))
                    if self.campaigns_list[exposition]["prob_exposition"][self.user_list[usr]["target_group"]] >= \
                            np.random.uniform(0, 1):
                        # metti un if campagna != awareness
                        delta_funnel_position = self.compute_delta_funnel_position(day, usr, exposition)
                        self.user_list[usr]["history"][(exposition, day)] += delta_funnel_position

                        if self.analysis_mode:
                            self.simulation["expositions"][day, usr, exposition] += 1

                        if self.campaigns_list[exposition]["target"] == "conversion":
                            rescaled_funnel_position = self.compute_rescaled_funnel_position(day, usr)
                            self.user_list[usr]["funnel_position"] = rescaled_funnel_position
                            if rescaled_funnel_position >= self.user_list[usr]["conv_threshold"]:
                                self.save_conversion(usr, day)

                        flag = 1

            if self.analysis_mode:
                self.simulation["funnel_position_history"][:, day] = \
                    [self.compute_rescaled_funnel_position(day, temp_usr) for temp_usr in range(N_users)]

    # Reset a user and create a new one
    def reset_user(self, usr, day):
        self.state["tot_users"] += 1
        self.user_list[usr] = generate_user("user_" + str(self.state["tot_users"]))
        self.user_list[usr]["entering_time"] = day

    # This function simulates possible actions of a user, in case this option is enabled (See users_actions_flag)
    # TODO: Check this function, now it doesn't work...
    def action_check(self, usr, day):
        # Compute the current position in the funnel. Probability of taking an action is proportional
        funnel_position = self.compute_rescaled_funnel_position(day, usr)
        prob_action = p_action * funnel_position

        # Sample the user behaviour.
        action = np.random.binomial(1, prob_action)

        if action == 1:
            self.user_list[usr]["actions"][day] = action_influence
            if self.compute_rescaled_funnel_position(day, usr) > self.user_list[usr]["conv_threshold"]:
                self.save_conversion(usr, day)

        # TODO: ricalcola con fattore moltiplicativo la posizione nel funnel
        # controlla threshold
        # se superiore, conversione esterna

    # Delta funnel_position function
    def compute_delta_funnel_position(self, t, user, campaign):
        # Basic influence of the campaign
        influence = self.campaigns_list[campaign]["influence"][self.user_list[user]["target_group"]]

        # Coefficient for response
        R_coefficient = simulate_response(user)

        # Coefficient due to old expositions
        hist = self.user_list[user]["history"][self.campaigns_list[campaign]["who_can_affect_me"], 0:t]
        daily_hist = np.sum(hist, axis=0)
        discounted_hist = discount_function(daily_hist)
        inf_growth = growth_function(discounted_hist)

        if self.analysis_mode & (self.campaigns_list[campaign]["target"] != "awareness"):
            self.simulation["inf_growth_history"][t, user, campaign - N_awn_camp] = inf_growth

        delta_funnel_position = influence * R_coefficient * inf_growth

        return delta_funnel_position

    # Rescaled funnel_position function
    def compute_rescaled_funnel_position(self, t, user):

        daily_funnel_position_awn = np.sum(self.user_list[user]["history"][:N_awn_camp, :t + 1], axis=0)
        daily_funnel_position_traff_cnv = np.sum(self.user_list[user]["history"][N_awn_camp:, :t + 1], axis=0)

        discounted_funnel_position_awn = discount_function(daily_funnel_position_awn)
        discounted_funnel_position_traff_cnv = discount_function(daily_funnel_position_traff_cnv)

        discounted_action_increase = discount_function(self.user_list[user]["actions"][:t + 1])

        rescaled_funnel_position = alpha * rescale_function(
            discounted_funnel_position_awn + alpha * discounted_action_increase) + \
                                   (1 - alpha) * rescale_function(
            discounted_funnel_position_traff_cnv + (1 - alpha) * discounted_action_increase)

        return rescaled_funnel_position

    # Update the environment once a user reaches conversion
    def save_conversion(self, usr, day):
        if self.analysis_mode:
            self.simulation[self.user_list[usr]["name"]] = day
            self.simulation["conversions_users"][self.state["conversions"]] = self.user_list[usr]  # Save this user
            self.simulation["conversions_users"][self.state["conversions"]]["exit_time"] = day  # Store also exit time
            # Compute user's history for partial plot
            self.simulation["conversions_users"][self.state["conversions"]]["funnel_history"] = \
                [self.compute_rescaled_funnel_position(temp_day, usr) for temp_day in
                 range(self.user_list[usr]["entering_time"], day + 1)]

            self.simulation["conversions_users"][self.state["conversions"]]["expositions"] = \
                self.simulation["expositions"][self.user_list[usr]["entering_time"]:day + 1, usr, :]

        # user_list[usr]["conversion"]=1 # TODO: da sistemare viene resettato subito
        self.state["conversions"] += 1
        self.reset_user(usr, day)

    # Plot results of the simulation
    def plot_results(self):
        if self.analysis_mode:
            set_matplotlib_properties()
            plot_all_funnel_positions(self)
            plot_conversion_paths(self)



