import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_bias(plot_data, labels, 
              cutoffs=[0.6, 0.7, 0.8, 0.9], 
              qvals=[0.05, 0.01, 0.001], 
              xlabel='Demographic Correction',
              ylabel=r'$\mathbf{R^2}$ Difference', 
              xlim=None,
              ylim=None,
              colors = ['#7f6d5f', 'olive', '#2d7f5e', 'steelblue'], 
              sigmarkers = ['*', '^', '.'],
              sigsizes = [200, 140, 300],
              sig_height = 0.015,
              figsize=(12, 7), 
              savepath=None,
              barlabels=True,
              show=True):
    # Plot data is a dataframe of the form output by test_bias
    # Labels is a list of the labels to use for the plot
    # Cutoffs is a list of the cutoffs used
    # Qvals is a list of the qvals used
    # figsize is the size of the figure
    
    ncut = len(cutoffs)
    nq = len(qvals)
    nlab = len(labels)
    nrow = len(plot_data)
    # print(f'ncut: {ncut}, nq: {nq}, nlab: {nlab}, nrow: {nrow}')
    assert nrow % ncut == 0
    assert nlab == nrow / ncut
    assert len(sigmarkers) == len(sigsizes)

    # setup bars
    barWidth = 0.9 / ncut
    locations = []
    for i in range(nlab):
        for j in range(ncut):
            locations.append(i + (j) * barWidth)
    plot_data['locations'] = locations
    bars = []
    for i in range(ncut):
        diff = np.abs(plot_data.iloc[i::ncut]['r2_1'].values - plot_data.iloc[i::ncut]['r2_0'].values)
        bars.append(diff)
    locs = [np.arange(len(bars[0]))]
    for i in range(1, ncut):
        locs.append(locs[i-1] + barWidth)

    # Make plot
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i in range(ncut):
        if i < len(colors):
            ax.bar(locs[i], bars[i], color=colors[i], width=barWidth, edgecolor='white', label=f'Threshold Percentile: {cutoffs[i]}')
        else:
            ax.bar(locs[i], bars[i], width=barWidth, edgecolor='white', label=f'Threshold Percentile: {cutoffs[i]}')
        # Add value labels
        if barlabels:
            for j in range(nlab):
                val = bars[i][j]
                ax.text(locs[i][j], val + 0.001, f'{val:.2f}', ha='center', va='bottom')
    # Add significance scatter points
    indlist = []
    qrev = qvals[::-1]
    for i in range(len(qvals)):
        q = qrev[i]
        inds = plot_data[f'Sig at {q}'].values
        for j in range(i):
            inds = inds & ~indlist[j]
        indlist.append(inds)

    for i in range(nq):
        inds = indlist[i]
        n = len(qvals)
        lab = r'Significant Bias at $\alpha='+repr(qvals[n-i-1])+r'$'
        if i < len(sigmarkers):
            ax.scatter(plot_data['locations'][inds], [sig_height for _ in range(sum(inds))], color='black', 
                       edgecolors='white', marker=sigmarkers[i], s=sigsizes[i], 
                       label=lab)
        else:
            ax.scatter(plot_data['locations'][inds], [sig_height for _ in range(sum(inds))], color='black', 
                       edgecolors='white', label=lab)
            
    # Add labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks([r + 1.5*barWidth for r in range(len(bars[0]))], labels)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([locs[0][0] - 1.5*barWidth, locs[-1][-1] + 1.5*barWidth])
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0, 0.9])
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=1)
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig