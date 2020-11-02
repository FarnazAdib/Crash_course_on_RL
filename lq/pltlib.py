import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


class PLTLIB:
    def __init__(self, j_inf_opt):
        self._colors = ["#000000", '#2D328F', '#F15C19', "#BF3EFF", "#81b13c", "#00FFFF", "#ca49ac", "#006400",
                        "#800000", "#DAA520", "#0000EE", "#A52A2A"]
        self._marker = [ ".", "o", "3", "^", ">", " ", "+", "v", "x", "2", "d", "p","3"]
        self._linestyle = [":", "dashdot", "dashed", (0, (3, 5, 1, 5, 1, 5)),"dotted", "-", "-", "-", "-", "-", "-", "-", "-"]
        self._label_fontsize = 14
        self._legend_fontsize = 13
        self._tick_fontsize = 14
        self._linewidth = 3
        self._markersize = [10,10, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self._j_inf_opt=j_inf_opt

    def relative_inf_cost(self, j, tot_samples, alg, xlab, uplim=6, zoomplt = True):
        '''
        Relative Infinite average cost error for for N_fix iterations for different trajectory length over MC stable samples
        :param j: Infinite average cost
        :param tot_samples:
        :param alg: Algorithm names
        :return:
        '''
        fig, ax = plt.subplots()
        for alg_i in range(len(alg)):
            ax.plot(tot_samples, np.median((j[alg_i]-self._j_inf_opt)/self._j_inf_opt, axis=0),
                    linestyle=self._linestyle[alg_i+1],
                    marker=self._marker[alg_i+1], color=self._colors[alg_i+1],
                    linewidth=self._linewidth, markersize=self._markersize[alg_i+1],
                    label=alg[alg_i])
            ax.fill_between(tot_samples, np.percentile((j[alg_i]-self._j_inf_opt)/self._j_inf_opt,25,axis=0),
                            np.percentile((j[alg_i]-self._j_inf_opt)/self._j_inf_opt,75,axis=0), alpha=0.25)

        ax.axis([tot_samples[0], tot_samples[-1], 0, uplim])

        ax.set_xlabel(xlab, fontsize=self._label_fontsize)
        ax.set_ylabel('Relative average cost error', fontsize=self._label_fontsize)
        # plt.legend(fontsize=self._legend_fontsize, loc='upper left',bbox_to_anchor=(0.4, 0.5))
        plt.legend(fontsize=self._legend_fontsize, loc='upper left')
        plt.xticks(fontsize=self._tick_fontsize)
        plt.yticks(fontsize=self._tick_fontsize)
        plt.grid(True)

        fig.set_size_inches(9, 6)
        plt.show()

        if(zoomplt):
            axins = inset_axes(ax, 2,1, loc=7)
            for alg_i in range(len(alg)):
                axins.plot(tot_samples, np.median((j[alg_i]-self._j_inf_opt)/self._j_inf_opt, axis=0),
                           linestyle=self._linestyle[alg_i+1],
                           marker=self._marker[alg_i+1], color=self._colors[alg_i+1],
                           linewidth=self._linewidth, markersize=self._markersize[alg_i+1],
                           label=alg[alg_i])
                axins.fill_between(tot_samples, np.percentile((j[alg_i]-self._j_inf_opt)/self._j_inf_opt,25,axis=0),
                                   np.percentile((j[alg_i]-self._j_inf_opt)/self._j_inf_opt,75,axis=0), alpha=0.25)

            axins.set_xlim(3900, 5000)  # apply the x-limits
            axins.set_ylim(-0.05, 0.55)  # apply the y-limits
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    def frac_stable(self, j, tot_samples, mc_trials, alg, xlab):
        '''
        Plot fraction stable vs tot_samples
        :param j:
        :param tot_samples:
        :param mc_trials:
        :param alg:
        :return:
        '''
        plt.figure()
        for alg_i in range(len(alg)):
            plt.plot(tot_samples,1-np.sum(np.isinf(j[alg_i]),axis=0)/mc_trials,
                     linestyle=self._linestyle[alg_i+1],
                     marker=self._marker[alg_i+1], color=self._colors[alg_i+1], linewidth=self._linewidth,
                     markersize=self._markersize[alg_i+1],
                     label=alg[alg_i])
        plt.axis([tot_samples[0],tot_samples[-1],0,1.2])
        plt.xlabel(xlab,fontsize=self._label_fontsize)
        plt.ylabel('Fraction stable',fontsize=self._label_fontsize)
        #plt.legend(fontsize=18, bbox_to_anchor=(0, 0))
        plt.legend(fontsize=self._legend_fontsize)
        plt.xticks(fontsize=self._tick_fontsize)
        plt.yticks(fontsize=self._tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        plt.show()

    def cost(self, j, tot_samples, alg, xlab, ylab, uplim):
        plt.figure()
        for alg_i in range(len(alg)):
            plt.plot(tot_samples, np.median(j[alg_i], axis=0),
                     linestyle=self._linestyle[alg_i],
                     marker=self._marker[alg_i], color=self._colors[alg_i], linewidth=self._linewidth,
                     markersize=self._markersize[alg_i],
                     label=alg[alg_i])

        plt.axis([tot_samples[0],tot_samples[-1], 0, uplim])

        plt.xlabel(xlab, fontsize=self._label_fontsize)
        plt.ylabel(ylab, fontsize=self._label_fontsize)
        plt.legend(fontsize=self._legend_fontsize, bbox_to_anchor=(0.4, 0.5))
        plt.xticks(fontsize=self._tick_fontsize)
        plt.yticks(fontsize=self._tick_fontsize)
        plt.grid(True)

        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        plt.show()

    def est_e(self, e, tot_samples, alg, xlab, ylab, uplim):
        plt.figure()
        for alg_i in range(len(alg)):
            plt.plot(tot_samples, np.median(e[alg_i], axis=0),
                     linestyle=self._linestyle[alg_i+1],
                     marker=self._marker[alg_i+1], color=self._colors[alg_i+1], linewidth=self._linewidth,
                     markersize=self._markersize[alg_i+1],
                     label=alg[alg_i])

        plt.axis([tot_samples[0], tot_samples[-1], 0, uplim])

        plt.xlabel(xlab, fontsize=self._label_fontsize)
        plt.ylabel(ylab, fontsize=self._label_fontsize)
        plt.legend(fontsize=self._legend_fontsize, bbox_to_anchor=(0.4, 0.5))
        plt.xticks(fontsize=self._tick_fontsize)
        plt.yticks(fontsize=self._tick_fontsize)
        plt.grid(True)

        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        plt.show()
