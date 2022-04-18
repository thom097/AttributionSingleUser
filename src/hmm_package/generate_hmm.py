from config.CONSTANTS_HMM import *
#from config.execution_parameters import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


# Observation generator
def simulate_observations():
    # This method builds a scenario to define which campaigns to attribute to each user and when

    exposition_vector = np.ones(N_exp_1) * 1
    exposition_vector = np.append(exposition_vector, np.ones(N_exp_2)*2)
    #exposition_vector = np.append(exposition_vector, np.ones(N_exp_3) * 3)
    exposition_vector = exposition_vector.astype(int)
    observation = np.zeros([N_camp, execution_duration, N_users])

    for day in range(execution_duration):
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
    # shape is [N_camp, execution_duration, N_users]
    tot_users = observation.shape[2]
    adstock = np.zeros(observation.shape)
    adstock[:, 0, :] = observation[:, 0, :]

    for ii in range(1, execution_duration):
        updates = observation[:, ii, :]
        adstock[:, ii, :] = discount_factor * adstock[:, ii - 1, :] + observation[:, ii, :]

    adstock = tf.constant(adstock, dtype=tf.float32)

    # TODO: NB MODIFY THIS CODE TO PUT THIS AT THE START
    adstock = tf.concat([tf.zeros([N_camp, 1, tot_users]), adstock], axis=1)

    adstock = tf.transpose(adstock, perm=[2, 0, 1])
    return adstock

# Compile the transition matrix
# TODO: This works with an iterator, which may result slow. Improve through straight computation of matrix when there is time
# Compile the transition matrix
# This works with an iterator, which may result slow. Improve through straight computation of matrix when there is time
def make_transition_matrix(mu, beta, adstock, basis=1e-10):
    batch_shape = tf.shape(adstock)[0]
    Q_final = tf.zeros([batch_shape, execution_duration, N_states, N_states])

    # Premodify structure
    # Reshape as square matrices
    mu_nosame_states = tf.reshape(mu, [1, N_states - 1, N_states - 1])
    beta_no_same_states = tf.reshape(beta, [N_camp, N_states - 1, N_states - 1])
    mu_nosame_states = tf.repeat(mu_nosame_states, execution_duration, axis=0)

    for iterator in tf.range(batch_shape):
        # Compute mu+O'*beta
        # Take time elements. First of adstock is zeros.
        # Adstock = [users, campaigns, time]
        O_beta = tf.tensordot(adstock[iterator, :, 1:], beta_no_same_states, [0, 0])

        # Solve Matrix computation
        num = tf.exp(O_beta + mu_nosame_states)
        den_vec = 1 + tf.reduce_sum(num, 2)
        den = tf.reshape(den_vec, [execution_duration, N_states - 1, 1])
        Q_div = num / den
        Q_same = 1 - tf.math.reduce_sum(Q_div, 2)

        Q_temp = tf.zeros([execution_duration, N_states - 1, N_states], dtype=tf.float32)
        Q_temp = tf.linalg.set_diag(Q_temp, Q_same)
        Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=0), k=1)
        for ii in range(1, N_states - 1):
            Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=ii), k=ii + 1)
            Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=-ii), k=-ii)

        # Attach the slice for last observable state
        last_piece = np.ones([execution_duration, 1, N_states]) * basis
        last_piece[:, 0, -1] = 1 - (N_states - 1) * basis

        Q_iter = tf.concat([Q_temp, last_piece], axis=1)
        Q_iter = tf.expand_dims(Q_iter, axis=0)
        if iterator == 0:
            Q_final = Q_iter

        else:
            Q_final = tf.concat([Q_final, Q_iter], axis=0)

    return Q_final


# Simplified version of make_transition_matrix without beta computation
def make_non_exposed_user_transtion_matrix(mu, adstock, basis = 1e-6):
    batch_shape = tf.shape(adstock)[0]
    mu_nosame_states = tf.reshape(mu, [1, N_states - 1, N_states - 1])
    mu_nosame_states = tf.repeat(mu_nosame_states, execution_duration, axis=0)
    num = tf.exp(mu_nosame_states)
    den_vec = 1 + tf.reduce_sum(num, 2)
    den = tf.reshape(den_vec, [execution_duration, N_states - 1, 1])
    Q_div = num / den
    Q_same = 1 - tf.math.reduce_sum(Q_div, 2)

    Q_temp = tf.zeros([execution_duration, N_states - 1, N_states], dtype=tf.float32)
    Q_temp = tf.linalg.set_diag(Q_temp, Q_same)
    Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=0), k=1)
    for ii in range(1, N_states - 1):
        Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=ii), k=ii + 1)
        Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=-ii), k=-ii)

    # Attach the slice for last observable state
    last_piece = np.ones([execution_duration, 1, N_states]) * basis
    last_piece[:, 0, -1] = 1 - (N_states - 1) * basis

    Q = tf.concat([Q_temp, last_piece], axis=1)
    Q = tf.expand_dims(Q, axis=0)
    return tf.repeat(Q, batch_shape, axis=0)


# HMM Parameters
def generate_hmm_distributions(states_observable, adstock=None, transition_matrix=None):

    if transition_matrix is not None: times_to_rep = tf.shape(transition_matrix)[0]
    else: times_to_rep = adstock.shape[0]
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
        obs_mat = np.zeros([N_states, 2], dtype=np.float32)+basis
        obs_mat[-1, -1] = 1-basis
        obs_mat[:-1, 0] = 1-basis
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


class set_beta_sign(tf.keras.constraints.Constraint):
    # TODO: this should depend on size.
    def __call__(self, w):
        #weights_correct_sign = []
        #for idx, weight in enumerate(w):
        #    weights_correct_sign.append(weight*tf.cast(tf.math.greater_equal(weight, 0.), w.dtype) if (idx+1)%N_states!=0 else
        #                                weight*tf.cast(tf.math.greater_equal(-weight, 0.), w.dtype))
        #final_weights = tf.concat(
        #    weights_correct_sign, axis=0
        #)
        final_weights = tf.concat(
            [w[0] * tf.cast(tf.math.greater_equal(w[0], 0.), w.dtype),
            w[1] * tf.cast(tf.math.greater_equal(w[1], 0.), w.dtype),
            w[2] * tf.cast(tf.math.greater_equal(-w[2], 0.), w.dtype),
            w[3] * tf.cast(tf.math.greater_equal(w[3], 0.), w.dtype),
            w[4] * tf.cast(tf.math.greater_equal(w[4], 0.), w.dtype),
            w[5] * tf.cast(tf.math.greater_equal(w[5], 0.), w.dtype),
            w[6] * tf.cast(tf.math.greater_equal(-w[6], 0.), w.dtype),
            w[7] * tf.cast(tf.math.greater_equal(w[7], 0.), w.dtype)],
            axis=0
        )
        return final_weights


class TransitionProbLayerMu(tf.keras.layers.Layer):

    def __init__(self, N_states):  # N_states should be passed as parameter to the layer to determine matrix dimension
        super(TransitionProbLayerMu, self).__init__()
        self.N_states = N_states

    def build(self, input_shape):
        N_states = self.N_states

        mu_dim = (N_states - 1) * (N_states - 1)
        # TODO: set a variable to run with real parameters
        self.mu = self.add_weight("mu", shape=[mu_dim],
                                  dtype='float32',
                                  #constraint=set_mu_sign(),
                                  # initializer=tf.keras.initializers.Constant(-mu),
                                  trainable=True)

    def call(self, adstock):
        # We suppose that the weights mu are all non-positive.
        # TODO: remove basis from make_transition_matrix
        Q = make_non_exposed_user_transtion_matrix(self.mu, adstock)

        return tf.math.maximum(Q, basis)


# Layer for functional API
class TransitionProbLayerBeta(tf.keras.layers.Layer):

    def __init__(self, N_states, mu, initializer = None):  # N_states should be passed as parameter to the layer to determine matrix dimension
        super(TransitionProbLayerBeta, self).__init__()
        self.N_states = N_states
        self.mu = mu
        self.initializer = initializer[0]

    def build(self, input_shape):
        N_camp = input_shape[1]
        days = input_shape[2]
        N_states = self.N_states

        beta_dim = (N_states - 1) * (N_states - 1) * (N_camp)
        mu_dim = (N_states - 1) * (N_states - 1)
        # TODO: set a variable to run with real parameters
        self.beta = self.add_weight("beta", shape=[beta_dim],
                                    dtype='float32',
                                    constraint = set_beta_sign(),
                                    initializer=self.initializer,
                                    trainable=True)

    def call(self, adstock):
        # We suppose that the weights mu are all non-positive.
        # TODO: remove basis from make_transition_matrix
        Q = make_transition_matrix(self.mu, self.beta, adstock, basis)

        return tf.math.maximum(Q, basis)


def build_hmm_to_fit_beta(states_observable, mu, initializer):
    # Generate functional model

    adstock_input = tf.keras.layers.Input(shape=(N_camp, execution_duration + 1,))
    Q = TransitionProbLayerBeta(N_states, mu, initializer)(adstock_input)
    out = tfp.layers.DistributionLambda(
        lambda t: tfd.HiddenMarkovModel(
            initial_distribution=generate_hmm_distributions(transition_matrix=t, states_observable=states_observable)['initial_distribution'],
            transition_distribution=tfd.Categorical(probs=t),
            observation_distribution=generate_hmm_distributions(transition_matrix=t, states_observable=states_observable)['observation_distribution'],
            time_varying_transition_distribution=True,
            num_steps=execution_duration + 1))(Q)

    return tf.keras.Model(inputs=adstock_input, outputs=out)


def build_hmm_to_fit_mu(states_observable):
    # Generate functional model

    adstock_input = tf.keras.layers.Input(shape=(N_camp, execution_duration + 1,))
    Q = TransitionProbLayerMu(N_states)(adstock_input)
    out = tfp.layers.DistributionLambda(
        lambda t: tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(probs=[1,0,0]),
            transition_distribution=tfd.Categorical(probs=t),
            observation_distribution=tfd.Categorical(
                probs = np.array([[1.0,0.0],[1.0,0.0],[0.0,1.0]], dtype=np.float32).reshape(1,3,2) ),
            time_varying_transition_distribution=True,
            num_steps=execution_duration + 1))(Q)

    return tf.keras.Model(inputs=adstock_input, outputs=out)


# Define Loss Function for our model
def loss_function(y, rv_y):
    """Negative log likelihood"""
    #posterior = tf.exp(rv_y.posterior_marginals(y).logits)[:,:,-1]
    #prior = tf.exp(rv_y.prior_marginals().logits[:,:,-1])
    #loss = tf.reduce_sum(tf.math.multiply(posterior, 1-prior)) +\
    #       tf.reduce_sum(tf.math.multiply(1-posterior, prior))

    return -tf.reduce_sum(rv_y.log_prob(y))

# Define Loss Function for our model
def loss_function_mu_matrix(y, rv_y):
    """Negative log likelihood"""
    loss_final_state_too_low = max([0, 0.67*(0.75 - rv_y.transition_distribution.probs[0,0,-2,-2])])
        #max([0, 16.7*(0.03-np.linalg.matrix_power( rv_y.transition_distribution.probs[0,0,:].numpy(), 31 )[0,2])]) +\
        #max([0, 0.67*(0.75 - rv_y.transition_distribution.probs[0,0,-2,-2])])
    loss = tf.reduce_sum(1-rv_y.tensor_distribution.prob(y))/BATCH_SIZE + loss_final_state_too_low
    return loss#-tf.reduce_sum(rv_y.log_prob(y))

# Define optimizer
def optimizer_function(lr_type):
    lr = learning_rate(mode=lr_type)
    return tf.keras.optimizers.Adam(learning_rate=lr)


# Define a handler to return the compiler options
class CompilerInfoBeta():
    def __init__(self, lr_type):
        self.loss = loss_function
        self.optimizer = optimizer_function(lr_type)


# Define a handler to return the compiler options
class CompilerInfoMu():
    def __init__(self, lr_type):
        self.loss = loss_function_mu_matrix
        self.optimizer = optimizer_function(lr_type)


def fit_model(model, adstock, emission_real):
    print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda batch, logs: print(f"Beta: {list(model.get_weights())}"))
    return model.fit(adstock,
                     emission_real,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     callbacks=[print_weights],
                     verbose=True)
