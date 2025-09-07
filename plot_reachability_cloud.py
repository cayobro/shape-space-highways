import numpy as np
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

r0 = r[-1,:,:]  
r3 = r[4,:,:]
r4 = r[54576,:,:] 
r5 = r[1203,:,:]  
shapes = [r0, r3, r4, r5]
tips = r[:,-1,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tips[:,0], tips[:,1], tips[:,2], color=get_color('palo'), alpha=0.01)
for curve in shapes:
    ax.plot(curve[:,0], curve[:,1], curve[:,2], color=get_color('cardinal'), linewidth=2.5)
# --- equalize axes ---
x_limits = ax.get_xlim(); y_limits = ax.get_ylim(); z_limits = ax.get_zlim()
x_range = x_limits[1]-x_limits[0]; y_range = y_limits[1]-y_limits[0]; z_range = z_limits[1]-z_limits[0]
max_range = max(x_range, y_range, z_range)
x_mid = sum(x_limits)/2; y_mid = sum(y_limits)/2; z_mid = sum(z_limits)/2
ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

ax.set_ylabel('$y$'); ax.set_xlabel('$x$'); ax.set_zlabel('$z$')
fig.set_size_inches((columnwidth, columnwidth))
# orientation: 130 azimuth, roll 180, elevation 0
fig.tight_layout()

if save:
    fig.savefig(plotdir + 'reachability.pdf', bbox_inches='tight')
