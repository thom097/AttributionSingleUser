from config.CONSTANTS_HMM import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


# Observation generator
def simulate_observations():
    # This method builds a scenario to define which campaigns to attribute to each user and when

    exposition_vector = np.ones(N_exp_1) * 1
    # exposition_vector = np.append(exposition_vector, np.ones(N_exp_2)*2)
    exposition_vector = exposition_vector.astype(int)
    observation = np.zeros([N_camp, time, N_users])

    for day in range(time):
        np.random.shuffle(exposition_vector)
        for exposition in exposition_vector:
            flag = 0
            while flag != 1:
                usr = round(np.random.uniform(0, N_users - 1))

                if np.random.binomial(1, p_exp[exposition - 1]) == 1:
                    observation[exposition - 1, day, usr] += 1
                    flag = 1

    return observation


# Compute Adstock function given the observation
def compute_adstock(observation):

    adstock = np.zeros(observation.shape)
    adstock[:, 0, :] = observation[:, 0, :]

    for ii in range(1, time):
        updates = observation[:, ii, :]
        adstock[:, ii, :] = discount_factor * adstock[:, ii - 1, :] + observation[:, ii, :]

    adstock = tf.constant(adstock, dtype=tf.float32)

    # NB MODIFY THIS CODE TO PUT THIS AT THE START
    adstock = tf.concat([tf.zeros([N_camp, 1, N_users]), adstock], axis=1)

    adstock = tf.transpose(adstock, perm=[2, 0, 1])
    return adstock


# Compile the transition matrix
# TODO: This works with an iterator, which may result slow. Improve through straight computation of matrix when there is time
# Compile the transition matrix
# This works with an iterator, which may result slow. Improve through straight computation of matrix when there is time
def make_transition_matrix(mu, beta, adstock, basis=1e-10):
    batch_shape = tf.shape(adstock)[0]
    Q_final = tf.zeros([batch_shape, time, N_states, N_states])
    for iterator in tf.range(batch_shape):

        # Reshape as square matrices
        mu_nosame_states = tf.reshape(mu, [1, N_states - 1, N_states - 1])
        beta_no_same_states = tf.reshape(beta, [N_camp, N_states - 1, N_states - 1])

        # Compute mu+O'*beta
        # Take time elements. First of adstock is zeros.
        # Adstock = [users, campaigns, time]
        O_beta = tf.tensordot(adstock[iterator, :, 1:], beta_no_same_states,
                              [0, 0])  # ATTENZIONE ORA PRENDO SOLO UN ADSTOCK!!
        mu_nosame_states = tf.repeat(mu_nosame_states, time, axis=0)

        # Solve Matrix computation
        num = tf.exp(O_beta + mu_nosame_states)
        den_vec = 1 + tf.reduce_sum(num, 2)
        den = tf.reshape(den_vec, [time, N_states - 1, 1])
        Q_div = num / den

        Q_same = 1 - tf.math.reduce_sum(Q_div, 2)
        Q_temp = tf.zeros([time, N_states - 1, N_states], dtype=tf.float32)

        Q_temp = tf.linalg.set_diag(Q_temp, Q_same)
        Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=0), k=1)
        for ii in range(1, N_states - 1):
            Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=ii), k=ii + 1)
            Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=-ii), k=-ii)

        # Attach the slice for last observable state
        last_piece = np.ones([time, 1, N_states]) * basis
        last_piece[:, 0, -1] = 1 - (N_states - 1) * basis

        Q_iter = tf.concat([Q_temp, last_piece], axis=1)
        Q_iter = tf.expand_dims(Q_iter, axis=0)
        if iterator == 0:
            Q_final = Q_iter

        else:
            Q_final = tf.concat([Q_final, Q_iter], axis=0)

    return Q_final


# HMM Parameters
def generate_hmm_distributions(states_observable, adstock=None, transition_matrix=None):

    if transition_matrix is not None: times_to_rep = tf.shape(transition_matrix)[0]
    else: times_to_rep = N_users
    # Set the user in the initial state
    initial_state_probs = np.zeros(N_states, dtype=np.float32)
    initial_state_probs[0] = 1
    initial_distribution = tfd.Categorical(probs=tf.repeat(initial_state_probs.reshape(1, -1), times_to_rep, axis=0))

    # TODO: change the observation into a function depending on N_states
    if states_observable:
        # The HMM is not hidden.
        obs_mat = np.eye(N_states, dtype=np.float32)
        observation_distribution = tfd.Categorical(probs=tf.repeat(
            obs_mat.reshape(1, N_states, N_states), times_to_rep, axis=0))
    else:
        obs_mat = np.zeros([N_states, 2], dtype=np.float32)
        obs_mat[-1, -1] = 1
        obs_mat[:-1, 0] = 1
        observation_distribution = tfd.Categorical(probs=tf.repeat(
            obs_mat.reshape(1, N_states, 2), times_to_rep, axis=0))

    if transition_matrix is not None:
        return {'initial_distribution': initial_distribution,
                'observation_distribution': observation_distribution}
    else:
        mu = tf.Variable(MU, dtype=np.float32)
        beta = tf.Variable(BETA, dtype=np.float32)
        transition_state_probabilities = make_transition_matrix(mu, beta, adstock)
        transition_distribution = tfd.Categorical(probs=transition_state_probabilities)

        return {'initial_distribution': initial_distribution,
                'observation_distribution': observation_distribution,
                'transition_distribution': transition_distribution}


def learning_rate(mode):
    # Generate the learning decay. Input is the mode desired (0=constant, 1=exponential decay)
    if ~mode:
        lr = LEARNING_RATE
    elif mode:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
    else:
        raise ValueError("Mode can only be 0 for constant Learning Rate or 1 for exponential decay.")
    return lr

# Layer for functional API
class TransitionProbLayer(tf.keras.layers.Layer):

    def __init__(self, N_states):  # N_states should be passed as parameter to the layer to determine matrix dimension
        super(TransitionProbLayer, self).__init__()
        self.N_states = N_states

    def build(self, input_shape):
        N_camp = input_shape[1]
        days = input_shape[2]
        N_states = self.N_states

        beta_dim = (N_states - 1) * (N_states - 1) * (N_camp)
        mu_dim = (N_states - 1) * (N_states - 1)
        # TODO: set a variable to run with real parameters
        self.mu = self.add_weight("mu", shape=[mu_dim],
                                  dtype='float32',
                                  constraint=tf.keras.constraints.non_neg(),
                                  # initializer=tf.keras.initializers.Constant(-mu),
                                  trainable=True)
        self.beta = self.add_weight("beta", shape=[beta_dim],
                                    dtype='float32',
                                    constraint=tf.keras.constraints.non_neg(),
                                    # initializer=tf.keras.initializers.Constant(beta),
                                    trainable=True)

    def call(self, adstock):
        # We suppose that the weights mu are all non-positive.
        # TODO: remove basis from make_transition_matrix
        Q = make_transition_matrix(-self.mu, self.beta, adstock, basis)

        return tf.math.maximum(Q, basis)


def build_hmm_to_fit(states_observable):
    # Generate functional model

    adstock_input = tf.keras.layers.Input(shape=(N_camp, time + 1,))
    Q = TransitionProbLayer(N_states)(adstock_input)
    out = tfp.layers.DistributionLambda(
        lambda t: tfd.HiddenMarkovModel(
            initial_distribution=generate_hmm_distributions(transition_matrix=t, states_observable=states_observable)['initial_distribution'],
            transition_distribution=tfd.Categorical(probs=t),
            observation_distribution=generate_hmm_distributions(transition_matrix=t, states_observable=states_observable)['observation_distribution'],
            time_varying_transition_distribution=True,
            num_steps=time + 1))(Q)

    return tf.keras.Model(inputs=adstock_input, outputs=out)


# Define Loss Function for our model
def loss_function(y, rv_y):
    """Negative log likelihood"""
    return -tf.reduce_sum(rv_y.log_prob(y))


# Define optimizer
def optimizer_function(lr_type):
    lr = learning_rate(mode=lr_type)
    return tf.keras.optimizers.Adam(learning_rate=lr)


# Define a handler to return the compiler options
class CompilerInfo():
    def __init__(self, lr_type):
        self.loss = loss_function
        self.optimizer = optimizer_function(lr_type)


def fit_model(model, adstock, emission_real):
    return model.fit(adstock,
                         emission_real,
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         verbose=True)
