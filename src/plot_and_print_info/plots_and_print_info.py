import numpy as np
from matplotlib import pylab as plt
from src.simulator_package.simulator_functions import *


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
def count_conversions(emission_real, states_are_observable):
    counter = 0
    conversion_id = int(states_are_observable)+1
    for ii in range(emission_real.shape[0]):
        if emission_real[ii, -1] == conversion_id:
            counter += 1
    print(counter)
    return counter


# Plot a batch of emissions sampled
def plot_sample_emissions(hmm):
    plt.figure(figsize=(30, 1))
    plt.title('Example of observations from the real model')
    for _ in range(10):
        plt.plot(hmm.sample().numpy()[0, :], 'o-')
    plt.xlabel('Steps')
    plt.ylabel('Emission')


# Plot auxiliary functions
def plot_auxiliary_functions():

    plt.figure()

    # Joint Exposition Probability weight function
    plt.subplot(2,2,1)
    plt.title("Probability Weight Function")
    x_plot = np.linspace(0,1,100)
    plt.plot(x_plot, np.power(x_plot,2)-x_plot+1)
    plt.axis([0, 1, 0, 1])

    # Growth function
    plt.subplot(2,2,2)
    plt.title("Growth function due to old expositions")
    x_plot = np.linspace(0,20,1000)
    plt.plot(x_plot, growth_function(x_plot))

    # Discount function
    plt.subplot(2,2,3)
    plt.title("Discount function")
    x_plot = np.linspace(0,10,1000)
    discount_factors = (np.log(0.1*x_plot+1)+2.8) / ( np.power(0.8*x_plot-1, 2)+1.8 )
    plt.plot(x_plot, discount_factors)

    # Rescale Function
    plt.subplot(2,2,4)
    plt.title("Rescale Function")
    x_plot = np.linspace(0,20,1000)
    plt.plot(x_plot, rescale_function(x_plot))
    plt.axis([0,20,0,1])


def plot_all_funnel_positions(simulation):
    # Users funnel_position
    plt.figure()
    plt.title("Funnel Position")
    for usr in range(N_users):
        plt.plot(range(0, time), simulation.simulation["funnel_position_history"][usr], \
                 label=simulation.user_list[usr]["name"], lw=lw)
    plt.legend()
    plt.xticks(range(0, time))
    plt.xlabel("Day")

    # Average growth
    simulation.simulation["inf_growth_history"][
        simulation.simulation["inf_growth_history"] == 0] = np.nan
    plt.figure()
    plt.title("Average influence growth for traffic and conversion campaigns")
    for camp in range(N_awn_camp, N_camp):
        plt.plot(range(0, time),
                 np.nanmean(simulation.simulation["inf_growth_history"][:, :, camp - N_awn_camp], axis=1), \
                 label=simulation.campaigns_list[camp]["name"], lw=lw)
    plt.legend()


def plot_conversion_paths(simulation):
    # funnel_position of users who had a conversion
    plt.figure()
    plt.title("Funnel Position of users who had a conversion")
    ax = plt.gca()
    for usr in simulation.simulation["conversions_users"]:
        tmp_usr = simulation.simulation["conversions_users"][usr]
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(range(tmp_usr["entering_time"], tmp_usr["exit_time"] + 1),
                 tmp_usr["funnel_history"],
                 label=tmp_usr["name"], lw=lw, color=color)
        plt.hlines(tmp_usr["conv_threshold"], 0, time, color=color, linestyles="dashed")
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
                 label=tmp_usr["name"], lw=lw, color=color)
        plt.hlines(tmp_usr["conv_threshold"], tmp_usr["entering_time"], tmp_usr["exit_time"], color=color,
                   linestyles="dashed")
        plt.vlines(tmp_usr["exit_time"], 0, 1, color=color, linestyles="dotted")

        for campaign_id in range(N_camp):
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

