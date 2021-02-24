import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


class PLTLIB:
    def __init__(self):
        self._colors = ["#000000", '#2D328F', '#F15C19', "#BF3EFF", "#81b13c", "#00FFFF", "#ca49ac", "#006400",
                        "#800000", "#DAA520", "#0000EE", "#A52A2A"]
        self._marker = [" ", "o", "3", "^", ">", " ", "+", "v", "x", "2", "d", "p","3"]
        self._linestyle = ["-", "dashdot", "dashed", (0, (3, 5, 1, 5, 1, 5)),"dotted", "-", "-", "-", "-", "-", "-", "-", "-"]
        self._label_fontsize = 14
        self._legend_fontsize = 13
        self._tick_fontsize = 14
        self._linewidth = 3
        self._markersize = [10,10, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    def reward_it(self, rewards):
        n_it = np.arange(len(rewards))
        plt.figure()

        plt.plot(n_it, rewards,
                 linestyle=self._linestyle[0],
                 marker=self._marker[0], color=self._colors[0], linewidth=self._linewidth,
                 markersize=self._markersize[0])

        plt.axis([n_it[0], n_it[-1], 0, 1.05*np.max(rewards)])
        plt.xlabel('Number of iterations', fontsize=self._label_fontsize)
        plt.ylabel('Total reward', fontsize=self._label_fontsize)
        plt.xticks(fontsize=self._tick_fontsize)
        plt.yticks(fontsize=self._tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        plt.show()
