# Constants
from config.execution_parameters import *

# Project libraries
from src.simulator_package.simulator_functions import *
from src.hmm_package.generate_hmm import *
from src.plot_and_print_info.plots_and_print_info import *

advertising_campaign = SimulationClass(ANALYSIS_MODE)
advertising_campaign.simulate()

adstock = compute_adstock(observation=advertising_campaign.results["user_expositions"])

model = build_hmm_to_fit( states_observable=STATES_ARE_OBSERVABLE )

compiler = CompilerInfo(LR_EXPONENTIAL_DECAY)
model.compile(
    loss = compiler.loss,
    optimizer = compiler.optimizer,
    run_eagerly = True
)

fit_model(model, adstock, advertising_campaign.results["user_outcome"])