import numpy as np
import csv
import os
from scipy.special import digamma
from src.hmm_package.generate_hmm import *


class OptimizerPGPE:

    def __init__(self, cm, hmm_trials, num_sample, N_users_to_simulate, init_values = None, uniform_initialization = None):
        """
        The class OptimizerPGPE contains the methods necessary to optimize the budget allocation for the Hidden Markov
        Model fitted previously.
        :param cm: constants manager
        :param hmm_trials: Number of times we sample from the HMM to reduce variance in computing the conversion per
        :param num_sample: Number of samples we extract from the Dirichlet at each step
        :param N_users_to_simulate: Dimension of the pool of users
        :param init_values: Initial budget distribution
        :param uniform_initialization: If initial distribution is not specified, take rules from this dict
        """

        if init_values is None:
            init_values = np.ones(uniform_initialization['num_campaigns'])/uniform_initialization['num_campaigns']
            init_values *= uniform_initialization['tot_impressions']

        self.cm = cm
        self.epsilon = 1e-16
        self.hmm_trials = hmm_trials
        self.num_sample = num_sample
        self.number_of_users = N_users_to_simulate
        self.tot_budget = sum(init_values)
        self.rho = np.array(np.log(init_values / self.tot_budget), copy=True)
        self.theta = np.zeros(init_values.shape[0])
        self.simulation_number = 0
        self.resample()
        self.variance = np.inf

    def resample(self):
        self.theta = np.random.dirichlet(np.exp(self.rho), self.num_sample)
        return np.copy(self.theta)

    def act(self, budget_distribution):
        complete_budget_distribution = budget_distribution*self.tot_budget
        impressions = [round(impression) for impression in complete_budget_distribution]
        avg_conversion = 0

        # Generate Test observation
        observation = simulate_observations(self.cm, impressions=impressions, number_of_users=self.number_of_users)
        # Compute Adstock
        adstock = compute_adstock(self.cm, observation=observation)
        hmm_distributions = generate_hmm_distributions(self.cm, initial_state_prob_vector=self.cm['INITIAL_STATE_PROB'],
                                                       click_prob=tf.constant(self.cm['CLICK_PROB']),
                                                       adstock=adstock)

        # Build Real HMM to simulate
        real_hmm = tfd.HiddenMarkovModel(
            initial_distribution=hmm_distributions['initial_distribution'],
            transition_distribution=hmm_distributions['transition_distribution'],
            observation_distribution=hmm_distributions['observation_distribution'],
            time_varying_transition_distribution=True,
            num_steps=self.cm['time'] + 1
        )

        # To reduce noise, we compute the average conversion on this particular campaign
        for ii in range(self.hmm_trials):
            # Sample emissions
            emission_real = real_hmm.sample().numpy()
            idx_conversion = max(emission_real[:, -1])
            avg_conversion += sum(1 for el in emission_real if el[-1] == idx_conversion)

        return avg_conversion/(self.hmm_trials*self.number_of_users)

    def get_theta(self):
        return np.copy(self.theta)

    def get_params(self):
        return np.copy(self.rho)

    def set_params(self, rho):
        self.rho = rho
        self.resample()

    def set_theta(self, theta):
        self.theta = theta

    def eval_gradient(self, thetas, returns, use_baseline=False):

        alpha = self.alpha
        sum_alpha = sum(alpha)
        b = 0
        gradients, gradient_norms = [], []

        self.simulation_number += 1

        min_vec = np.ones(len(alpha)) * 1e-4
        for theta in thetas:
            d_alpha = alpha*(digamma(sum_alpha) - digamma(alpha) + np.log(theta))/np.max([min_vec, 1-np.power(np.tanh(alpha), 2)], axis=0)
            gradients.append(d_alpha)
            gradient_norms.append(np.linalg.norm(d_alpha))

        if use_baseline:
            if len(returns) > 1:
                gradient_norms = np.array(gradient_norms)
                num = (returns * gradient_norms ** 2).mean()
                den = (gradient_norms ** 2).mean()
                b = num / den

        self.variance = (sum_alpha-alpha)*alpha/(alpha**2*(alpha+1))

        return np.array(1) + np.tanh(gradients * (np.array(returns) - b)[:, np.newaxis]).mean(axis=0)

    def show_theta(self):
        print(self.theta)

    def get_budget_distribution(self):
        return np.exp(self.rho)/sum(np.exp(self.rho))

    @property
    def alpha(self):
        return np.exp(self.rho)
