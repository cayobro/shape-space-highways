import numpy as np
import pickle
import scipy
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *

save = False
plt.style.use('/Users/cveil/git/paper_style.mplstyle')
columnwidth = 3.5
textwidth = 7

scenario = 'cylinder' # cylinder or box
custom_dir = f'/Users/cveil/Desktop/sim/shape_sequences/{scenario}_obstacle'
plot_dir = '/Users/cveil/Desktop/'

if scenario == 'box':
    waypoint_indices = {'basic': [0,27,58],
                        'sdf': [0, 60, 120],
                        'sdf_energy': [0, 74, 139]}
elif scenario == 'cylinder':
    waypoint_indices = {'sdf': [0, 0, 0, 0, 0]}


# sequences = ['basic', 'sdf', 'sdf_energy']
# labels = ['Geometric', "SDF only", "SDF + Energy"]
# sequences = ['sdf', 'sdf_energy']
# labels = ["SDF only", "SDF + Energy"]
sequences = ['sdf']
labels = ["SDF only"]
path_color = [get_color('plum'), get_color('palo'), get_color('poppy'), get_color('bay'), get_color('sky'),  get_color('spirited'), get_color('brick'), get_color('archway')]
fig_g, ax_g = plt.subplots(3,1 ,figsize=(columnwidth, 2*columnwidth), sharex=True)  
i = 0
for seq in sequences:
    gamma = pickle.load(open(custom_dir + f'/gamma_seq_{seq}.pickle', 'rb'))
    number_of_sections = len(waypoint_indices) - 1
    x_sections = np.linspace(0, 1, num=number_of_sections + 1) 
    for j_sec in range(len(waypoint_indices)-1):
        idx1 = waypoint_indices[seq][j_sec]
        idx2 = waypoint_indices[seq][j_sec+1]
        section_gammas = gamma[idx1:idx2+1]
        xvec = np.linspace(x_sections[j_sec], x_sections[j_sec+1], num=section_gammas.shape[0])
        for j_g in range(3):
            ax_g[j_g].plot(xvec, section_gammas[:,j_g], marker='.', markersize=3,color=path_color[i],
                           label=labels[i] if j_sec == 0 and j_g == 0 else None)
            ax_g[j_g].vlines(x=xvec[0], ymin=-1.6, ymax=0.0, color='gray')
            ax_g[j_g].vlines(x=xvec[-1], ymin=-1.6, ymax=0.0, color='gray')
    i = i + 1
for i, a in enumerate(ax_g):
    # a.legend()
    a.set_ylabel(f'$\gamma_{i+1}$')
    a.grid(True)
    a.set_xticklabels([])
fig_g.set_size_inches((columnwidth, 1.3*columnwidth)) # 1X 1.3X
fig_g.tight_layout()

fig_legend = plt.figure(figsize=(columnwidth, 0.5))
legend_handles, legend_labels = ax_g[0].get_legend_handles_labels()
fig_legend.legend(legend_handles, legend_labels, loc='center', ncol=len(labels))
fig_legend.tight_layout()

if save:
    fig_g.savefig(plot_dir + f'/gammas_{scenario}_obstacle.pdf', bbox_inches='tight')
    fig_legend.savefig(plot_dir + f'/gammas_legend_{scenario}_obstacle.pdf', bbox_inches='tight')




