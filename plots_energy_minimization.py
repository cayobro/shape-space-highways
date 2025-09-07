import numpy as np
import pickle
import pylab
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *

save = False
plotdir = '/Users/cveil/Desktop/'
plt.style.use('/Users/cveil/git/paper_style.mplstyle')
columnwidth = 3.5
textwidth = 7

r, R, gamma, N, P = get_data()

# load benchmark
custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
nbrs, idxs = pickle.load(open(custom_dir + '/knn.pickle', 'rb')), pickle.load(open(custom_dir + '/idxs.pickle', 'rb'))
adj_basic = pickle.load(open(custom_dir + '/adj_basic.pickle', 'rb'))
adj_a1_b1_d1 = pickle.load(open(custom_dir + '/adj_energy_a1_b1_d1.pickle', 'rb'))
adj_a500_b1_d1 = pickle.load(open(custom_dir + '/adj_energy_a500_b1_d1.pickle', 'rb'))

labels = ['$\\alpha=0$, $\\beta=\delta=0$', '$\\alpha=\\beta=\delta=1$', '$\\alpha=500$, $\\beta=\delta=1$']

r0 = r[-1,:,:]; r3 = r[4,:,:]; waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

full_path_indices = []
for temp_adj in [adj_basic, adj_a1_b1_d1, adj_a500_b1_d1]:
    path_indices = waypoint_planner(waypoint_indices, adj=temp_adj)
    full_path_indices.append(path_indices)

path_color = [get_color('plum'), get_color('palo'), get_color('poppy'), get_color('bay'), get_color('sky'),  get_color('spirited'), get_color('brick'), get_color('archway')]

## PLOT GAMMAS 
fig, ax = plt.subplots(1,3 ,figsize=(columnwidth, 3*columnwidth), sharex=True)  # Create subplots for gamma1, gamma2, gamma3
i = 0
for path_indices in full_path_indices:
    gammas = gamma[path_indices]
    xvec = np.arange(1, gammas.shape[0] + 1) 
    ax[0].plot(xvec, gammas[:, 0], marker='.', color=path_color[i])
    ax[1].plot(xvec, gammas[:, 1], marker='.', color=path_color[i], label = labels[i])
    ax[2].plot(xvec, gammas[:, 2], marker='.', color=path_color[i])
    i = i + 1
# ax[1].legend()
ax[0].set_ylabel('$\gamma_1$'); ax[1].set_ylabel('$\gamma_2$'); ax[2].set_ylabel('$\gamma_3$')
ax[0].set_xlabel('Index'); ax[1].set_xlabel('Index'); ax[2].set_xlabel('Index')
fig.set_size_inches((2*columnwidth, 0.5*columnwidth)) # 1X 1.3X
fig.tight_layout()  # Adjust layout to prevent overlap
plt.show(block=False)  # Display the plot

# legu = pylab.figure(figsize=(2*columnwidth, 0.2))
# legu.legend(linesleg, labels, ncol=3, frameon=False, loc='center')

if save:
    fig.savefig(plotdir + 'energy-focus-gammas.pdf', bbox_inches='tight')

## PLOT SHAPES
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

i = 0
for path_indices in full_path_indices:
    if i == 2: continue # skip last shapes
    shapes = r[path_indices]
    leg = False
    for t, curve in enumerate(shapes):
        if leg is False: 
            ax.plot(curve[:,0], curve[:,1], curve[:,2], color=path_color[i], label = labels[i], linewidth=1.0)
            leg = True
        else: ax.plot(curve[:,0], curve[:,1], curve[:,2], color=path_color[i], linewidth=1.0)
    i = i + 1

leg = False
for wp in waypoints:
    if leg is False:
        ax.plot(wp[:,0], wp[:,1], wp[:,2], color=get_color('cardinal'), label="Start and End", linewidth=2.5)
        leg = True
    else:
        ax.plot(wp[:,0], wp[:,1], wp[:,2], color=get_color('cardinal'), linewidth=2.5)

# --- equalize axes ---
x_limits = ax.get_xlim(); y_limits = ax.get_ylim(); z_limits = ax.get_zlim()
x_range = x_limits[1]-x_limits[0]; y_range = y_limits[1]-y_limits[0]; z_range = z_limits[1]-z_limits[0]
max_range = max(x_range, y_range, z_range)
x_mid = sum(x_limits)/2; y_mid = sum(y_limits)/2; z_mid = sum(z_limits)/2
ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

ax.legend(); ax.set_ylabel('$y$'); ax.set_xlabel('$x$'); ax.set_zlabel('$z$')

fig.set_size_inches((columnwidth, columnwidth)); fig.tight_layout()  # Adjust layout to prevent overlap
plt.show(block=False) 

if save:
    fig.savefig(plotdir + 'energy-focus-shapes-2.pdf', bbox_inches='tight')

input("Debug breakpoint. Press Enter to exit...")