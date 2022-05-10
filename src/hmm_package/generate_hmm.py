#from config.CONSTANTS_HMM import *
from config.execution_parameters import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import pickle
import os

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


def make_transition_matrix(mu, beta, adstock, basis=1e-10):

    len_mu = tf.shape(mu)[0]
    batch_shape = tf.shape(adstock)[0]

    beta_reshaped = tf.concat([tf.zeros([N_camp,1]), tf.reshape(beta, [N_camp, len_mu])], axis=1)
    O_beta = tf.tensordot(adstock[:,:,1:], beta_reshaped, axes=[1, 0])

    mu_reshaped = tf.expand_dims(tf.expand_dims( tf.concat([[-np.inf],mu],axis=0) , axis=0), axis=0)
    mu_reshaped = tf.repeat(tf.repeat(mu_reshaped, batch_shape, axis=0), execution_duration, axis=1)

    mu_plus_Obeta = tf.reshape( tf.expand_dims(mu_reshaped+O_beta, axis=-1), [batch_shape, execution_duration, N_states-1, -1])
    num = tf.exp(mu_plus_Obeta)
    den = 1 + tf.reduce_sum(num, axis=3)
    main_diag = tf.concat([1/den, tf.ones([batch_shape, execution_duration, 1])], axis=-1)
    lower_diag = tf.concat([num[:, :, 1:, 0] / den[:, :, 1:], tf.zeros([batch_shape, execution_duration, 1])], axis=-1)
    upper_diag = num[:, :, :, 1] / den

    Q = tf.zeros([batch_shape, execution_duration, N_states, N_states])
    Q = tf.linalg.set_diag(Q, main_diag)
    Q = tf.linalg.set_diag(Q, lower_diag, k=-1)
    Q = tf.linalg.set_diag(Q, upper_diag, k=1)

    return tf.math.maximum(Q, basis)


# # Compile the transition matrix
# # TODO: This works with an iterator, which may result slow. Improve through straight computation of matrix when there is time
# # Compile the transition matrix
# # This works with an iterator, which may result slow. Improve through straight computation of matrix when there is time
# def make_transition_matrix(mu, beta, adstock, basis=1e-10):
#     batch_shape = tf.shape(adstock)[0]
#     Q_final = tf.zeros([batch_shape, execution_duration, N_states, N_states])
#
#     # Premodify structure
#     # Reshape as square matrices
#     mu_nosame_states = tf.reshape(mu, [1, N_states - 1, N_states - 1])
#     beta_no_same_states = tf.reshape(beta, [N_camp, N_states - 1, N_states - 1])
#     mu_nosame_states = tf.repeat(mu_nosame_states, execution_duration, axis=0)
#
#     for iterator in tf.range(batch_shape):
#         # Compute mu+O'*beta
#         # Take time elements. First of adstock is zeros.
#         # Adstock = [users, campaigns, time]
#         O_beta = tf.tensordot(adstock[iterator, :, 1:], beta_no_same_states, [0, 0])
#
#         # Solve Matrix computation
#         num = tf.exp(O_beta + mu_nosame_states)
#         den_vec = 1 + tf.reduce_sum(num, 2)
#         den = tf.reshape(den_vec, [execution_duration, N_states - 1, 1])
#         Q_div = num / den
#         Q_same = 1 - tf.math.reduce_sum(Q_div, 2)
#
#         Q_temp = tf.zeros([execution_duration, N_states - 1, N_states], dtype=tf.float32)
#         Q_temp = tf.linalg.set_diag(Q_temp, Q_same)
#         Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=0), k=1)
#         for ii in range(1, N_states - 1):
#             Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=ii), k=ii + 1)
#             Q_temp = tf.linalg.set_diag(Q_temp, tf.linalg.diag_part(Q_div, k=-ii), k=-ii)
#
#         # Attach the slice for last observable state
#         last_piece = np.ones([execution_duration, 1, N_states]) * basis
#         last_piece[:, 0, -1] = 1 - (N_states - 1) * basis
#
#         Q_iter = tf.concat([Q_temp, last_piece], axis=1)
#         Q_iter = tf.expand_dims(Q_iter, axis=0)
#         if iterator == 0:
#             Q_final = Q_iter
#
#         else:
#             Q_final = tf.concat([Q_final, Q_iter], axis=0)
#
#     return Q_final


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
def generate_hmm_distributions(initial_state_prob_vector, click_prob,
                               adstock=None, transition_matrix=None):

    if transition_matrix is not None: times_to_rep = tf.shape(transition_matrix)[0]
    else: times_to_rep = adstock.shape[0]
    # Set the user in the initial state
    initial_state_probs = tf.concat([initial_state_prob_vector, [1-tf.reduce_sum(initial_state_prob_vector), 0]], axis=0)

    #if tf.shape(initial_state_probs)[0] != N_states or abs(sum(initial_state_probs) - 1) > 1e-7:
    #    raise ValueError(f"The initial_state_prob_vector must have {N_states-2} values!")

    initial_distribution = tfd.Categorical(probs=tf.repeat(tf.expand_dims(initial_state_probs, axis=0), times_to_rep, axis=0))
    # TODO: change the observation into a function depending on N_states
    if STATES_ARE_OBSERVABLE:
        # The HMM is not hidden.
        obs_mat = np.eye(N_states, dtype=np.float32)
        observation_distribution = tfd.Categorical(probs=tf.repeat(
            obs_mat.reshape(1, N_states, N_states), times_to_rep, axis=0))
    else:
        obs_mat = tf.reshape([1., 0., 0.], [1,3])
        intermediate_click_probs = tf.repeat(tf.reshape([1 - click_prob, click_prob, [0.]], [1,3]), N_states-2, axis=0)
        obs_mat = tf.concat([obs_mat, intermediate_click_probs], axis=0)
        obs_mat = tf.concat([obs_mat, tf.reshape([0., 0., 1.], [1,3])], axis=0)
        observation_distribution = tfd.Categorical(probs=tf.repeat(tf.reshape(obs_mat, [1,3,3]), times_to_rep, axis=0))

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
            w[1] * tf.cast(tf.math.greater_equal(-w[1], 0.), w.dtype),
            w[2] * tf.cast(tf.math.greater_equal(w[2], 0.), w.dtype),
            w[3] * tf.cast(tf.math.greater_equal(w[3], 0.), w.dtype),
            w[4] * tf.cast(tf.math.greater_equal(-w[4], 0.), w.dtype),
            w[5] * tf.cast(tf.math.greater_equal(w[5], 0.), w.dtype),
            w[6] * tf.cast(tf.math.greater_equal(w[6], 0.), w.dtype),
            w[7] * tf.cast(tf.math.greater_equal(-w[7], 0.), w.dtype),
            w[8] * tf.cast(tf.math.greater_equal(w[8], 0.), w.dtype)
             ], axis=0
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
        self.click_prob = None
        self.init_prob = None
        self.beta = None
        self.mu = None
        self.N_states = N_states
        #self.mu = mu
        self.initializer = initializer[0]

    def build(self, input_shape):
        N_camp = input_shape[1]
        days = input_shape[2]
        N_states = self.N_states

        mu_dim = 1 + 2 * (N_states - 2)
        beta_dim = mu_dim * N_camp
        init_prob_dim = N_states - 2

        # TODO: set a variable to run with real parameters
        self.mu = self.add_weight(shape=[3],
                                    dtype='float32',
                                    initializer=tf.keras.initializers.Constant([-2.,-1.,-4.]),
                                    trainable=True,
                                    name='Beta')
        self.beta = self.add_weight(shape=[beta_dim],
                                    dtype='float32',
                                    constraint = set_beta_sign(),
                                    initializer=self.initializer['BETA'],
                                    trainable=True,
                                    name='Beta')
        self.init_prob = self.add_weight(shape=[init_prob_dim],
                                         dtype='float32',
                                         initializer=self.initializer['INIT_PROB'],
                                         trainable=True,
                                         name='Entry Probability')
        self.click_prob = self.add_weight(shape=[1],
                                          dtype='float32',
                                          initializer=self.initializer['CLICK_PROB'],
                                          trainable=True,
                                          name='Click Probability in intermediate state')

    def call(self, adstock):
        # We suppose that the weights mu are all non-positive.
        # TODO: remove basis from make_transition_matrix
        Q = make_transition_matrix(self.mu, self.beta, adstock, basis)
        init_prob = tf.keras.activations.sigmoid(self.init_prob)
        click_prob = tf.keras.activations.sigmoid(self.click_prob)
        return {"Q": Q, "init_prob": init_prob, "click_prob": click_prob}


def build_hmm_to_fit_beta(mu, initializer):
    # Generate functional model

    adstock_input = tf.keras.layers.Input(shape=(N_camp, execution_duration + 1,))
    parameters = TransitionProbLayerBeta(N_states, mu, initializer)(adstock_input)
    out = tfp.layers.DistributionLambda(
        lambda t: tfd.HiddenMarkovModel(
            initial_distribution=generate_hmm_distributions(initial_state_prob_vector=t['init_prob'], click_prob=t['click_prob'],
                                                            transition_matrix=t['Q'])['initial_distribution'],
            transition_distribution=tfd.Categorical(probs=t['Q']),
            observation_distribution=generate_hmm_distributions(initial_state_prob_vector=t['init_prob'], click_prob=t['click_prob'],
                                                                transition_matrix=t['Q'])['observation_distribution'],
            time_varying_transition_distribution=True,
            num_steps=execution_duration + 1))(parameters)

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
    print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda batch, logs: print(f"Mu: {list(model.get_weights()[0])}"
                                                                                               f" Beta: {list(model.get_weights()[1])}"
                                                                                               f" Init Prob: {list(tf.keras.activations.sigmoid(model.get_weights()[2]))}"
                                                                                               f" Click Prob: {list(tf.keras.activations.sigmoid(model.get_weights()[3]))}"))
    return model.fit(adstock,
                     emission_real,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     callbacks=[print_weights],
                     verbose=True)

def save_result(model, adstock, emission_real, initializer):

    if LR_EXPONENTIAL_DECAY:
        lr_title = f"LR: Exp Decay, init={initial_learning_rate}, steps={decay_steps}, decay rate={decay_rate}"
    else:
        lr_title = f"LR: Constant, value={LEARNING_RATE}"

    title = f"N Users:{N_users}; Epochs: {EPOCHS}; Learning rate: {lr_title}"

    dict_to_store = {
        'Initializer': initializer,
        'Model Weights': {el.name: el.numpy() for el in model.weights},
        'Adstock': adstock,
        'Emissions observer': emission_real,
        'Beta Real': BETA,
        'Entry Prob Real': INITIAL_STATE_PROB,
        'Click Prob': CLICK_PROB,
        'History': model.history.history['loss']
    }
    #TODO: check that it works in main
    filepath = os.path.join(os.path.dirname(os.path.abspath('')), 'data', 'hmm_test_fitting.data')
    try:
        with open(filepath, "rb") as file:
            data = pickle.load(file)
        if title in data:
            raise ValueError("This test already exists in hmm_test_fitting.data!")
        data[title] = dict_to_store
        with open(filepath, "wb") as file:
            pickle.dump(data, file)
    except (OSError, IOError) as e:
        with open(filepath, "wb") as file:
            pickle.dump({title: dict_to_store}, file)

def load_results():
    filepath = os.path.join(os.path.dirname(os.path.abspath('')), 'data', 'hmm_test_fitting.data')
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    return result

