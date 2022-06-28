import numpy as np
import random as rnd
import tikzplotlib as tkz
from matplotlib import pylab as plt
from src.simulator_package.simulator_functions import *
from matplotlib.ticker import MaxNLocator

def set_matplotlib_properties():
    plt.rcParams.update(
        {
            "font.size": 16,
            "figure.figsize": (16, 10),
            "legend.fontsize": 12,
            "legend.frameon": True,
            "legend.loc": "upper right"
        }
    )


# Count conversions from REAL_HMM
def count_conversions(emission_real):
    idx_conversion = max(el[-1] for el in emission_real)
    counter = sum(1 for el in emission_real if el[-1] == idx_conversion)
    conversion_percentage = 100*counter/emission_real.shape[0]
    print(f"Percentage of conversion is: {conversion_percentage}%.")
    return conversion_percentage


# Plot a batch of emissions sampled
def plot_sample_emissions(hmm):
    plt.figure(figsize=(10, 4))
    plt.title('Example of observations from the real model')
    for _ in range(6):
        plt.plot(hmm.sample().numpy()[0, :], 'o-')
    plt.xlabel('Steps')
    plt.ylabel('Emission')
    plt.yticks([0, 1, 2], ['No action', 'Click', 'Conversion'])
    tkz.save('/Users/Thom/OneDrive - Politecnico di Milano/Università/TESI/images/emissions_hmm.tex')


# Plot auxiliary functions
def plot_auxiliary_functions():

    plt.figure(figsize=(10,10))

    # Joint Exposition Probability weight function
    plt.subplot(2,2,1)
    plt.title("Probability Weight Function")
    x_plot = np.linspace(0, 1, 100)
    plt.plot(x_plot, np.power(x_plot, 2) - x_plot + 1)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Probability')
    plt.ylabel('Weight')

    # Growth function
    plt.subplot(2,2,2)
    plt.title("Growth function due to old expositions")
    x_plot = np.linspace(0, 20, 100)
    plt.plot(x_plot, growth_function(x_plot))
    plt.axis([0, 5, 1, 2.1])
    plt.axhline(2, color='r', linestyle='dotted')
    plt.xlabel('Previous expositions influence')
    plt.ylabel('Influence Coefficient')

    # Discount function
    plt.subplot(2,2,3)
    plt.title("Discount function")
    x_plot = np.linspace(0, 10, 1000)
    discount_factors = (np.log(0.1 * x_plot + 1) + 5) / (np.power(0.8 * x_plot - 1, 2) + 4)
    plt.plot(x_plot, discount_factors)
    plt.xlabel('Days from exposition')
    plt.ylabel('Campaign effect')

    # Rescale Function
    plt.subplot(2,2,4)
    plt.title("Rescale Function")
    x_plot = np.linspace(0,20,1000)
    plt.plot(x_plot, rescale_function(x_plot))
    plt.axis([0,20,0,1])


def plot_all_funnel_positions(cm, simulation, n_to_plot):
    # Users funnel_position
    plt.figure(figsize=(10,10))
    plt.title("Funnel Position")
    users_list = [usr for usr in range(cm['N_users'])]
    rnd.shuffle(users_list)
    for usr in users_list[:n_to_plot]:
        plt.plot(range(0, cm['time']), simulation.simulation["funnel_position_history"][usr], \
                 label=simulation.user_list[usr]["name"], lw=cm['lw'])
    plt.legend()
    plt.xticks(range(0, cm['time']))
    plt.xlabel("Days")

    # Average growth
    simulation.simulation["inf_growth_history"][
        simulation.simulation["inf_growth_history"] == 0] = np.nan
    plt.figure(figsize=(10,10))
    plt.title("Average influence growth for traffic and conversion campaigns")
    for camp in range(cm['N_awn_camp'], cm['N_camp']):
        plt.plot(range(0, cm['time']),
                 np.nanmean(simulation.simulation["inf_growth_history"][:, :, camp - cm['N_awn_camp']], axis=1), \
                 label=simulation.campaigns_list[camp]["name"], lw=cm['lw'])
    plt.xticks(range(0, cm['time']))
    plt.xlabel('Days')
    plt.ylabel('Influence growth coefficient')
    plt.legend()
    tkz.save('/Users/Thom/OneDrive - Politecnico di Milano/Università/TESI/images/influence_growth.tex')
    return


def plot_conversion_paths(cm, simulation):
    # funnel_position of users who had a conversion
    plt.figure()
    plt.title("Funnel Position of users who had a conversion")
    ax = plt.gca()
    for usr in simulation.simulation["conversions_users"]:
        tmp_usr = simulation.simulation["conversions_users"][usr]
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(range(tmp_usr["entering_time"], tmp_usr["exit_time"] + 1),
                 tmp_usr["funnel_history"],
                 label=tmp_usr["name"], lw=cm['lw'], color=color)
        plt.hlines(tmp_usr["conv_threshold"], 0, cm['time'], color=color, linestyles="dashed")
        plt.vlines(tmp_usr["exit_time"], 0, 1, color=color, linestyles="dotted")
    plt.legend()

    # Plot the history which brought to conversion for each user
    for usr in simulation.simulation["conversions_users"]:

        tmp_usr = simulation.simulation["conversions_users"][usr]

        plt.figure()
        plt.title("Funnel Position evolution of " + tmp_usr["name"])
        ax = plt.gca()

        #   usr_id = int(usr[5:])-1
        color = next(ax._get_lines.prop_cycler)['color']
        x_plot = np.linspace(tmp_usr["entering_time"], tmp_usr["exit_time"],
                             tmp_usr["exit_time"] - tmp_usr["entering_time"] + 1, dtype='int')
        plt.plot(x_plot,
                 tmp_usr["funnel_history"],
                 label=tmp_usr["name"], lw=cm['lw'], color=color)
        plt.hlines(tmp_usr["conv_threshold"], tmp_usr["entering_time"], tmp_usr["exit_time"], color='r',
                   linestyles="dashed")
        plt.vlines(tmp_usr["exit_time"], 0, 1, color=color, linestyles="dotted")

        plt.xlabel('Days')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('Position in the funnel')

        for campaign_id in range(cm['N_camp']):
            x_plot_camp = x_plot[tmp_usr["history"][campaign_id, x_plot] > 0]
            y_plot = np.array(tmp_usr["funnel_history"])[tmp_usr["history"][campaign_id, x_plot] > 0]
            plt.plot(x_plot_camp, y_plot, 'o')

        dx = 0  # -0.5
        dy = 0  # +0.02
        for x_tmp in x_plot:
            positions = np.where(tmp_usr["history"][:, x_tmp] > 0)[0]
            txt = ''
            for jj in positions:
                txt = txt + simulation.campaigns_list[jj]["name"][5:] + '\n'
            #  if (simulation["expositions"][x_plot[ii],usr_id,jj])==1:
            #      txt = txt+campaigns_list[jj]["name"][5:]+'\n'
            #  else:
            #      txt = txt+str(simulation["expositions"][x_plot[ii],usr_id,jj].astype(int))+'x'+campaigns_list[jj]["name"][5:]+'\n'
            y_tmp = tmp_usr["funnel_history"][x_tmp - tmp_usr["entering_time"]]
            ax.annotate(txt, (x_tmp, y_tmp), (x_tmp + dx, y_tmp + dy))
    plt.legend()

def plot_loss_decay(hmm_model):
    plt.figure(figsize=(10, 4))
    plt.title('Loss decay fitting HMM')
    plt.plot(hmm_model.history.epoch, hmm_model.history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    tkz.save('/Users/Thom/OneDrive - Politecnico di Milano/Università/TESI/images/loss_decay.tex')

