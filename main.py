# Constants
from config.CONSTANTS_HMM import *
from config.execution_parameters import *

# Project libraries
import src.simulator_package.simulator_functions
from src.simulator_package.simulator_functions import *
import src.hmm_package.generate_hmm
from src.hmm_package.generate_hmm import *
from src.plot_and_print_info.plots_and_print_info import *

# Built in directories
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import importlib

ANALYSIS_MODE = False

advertising_campaign = SimulationClass(ANALYSIS_MODE)
advertising_campaign.simulate()

