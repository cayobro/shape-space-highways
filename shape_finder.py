import numpy as np
import pandas as pd
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *

save = False
save_dir = '/Users/cveil/Desktop/'
r, _, gamma, _, _ = get_data()

idx = 4950 # 1203 # 4

fig, ax = plt.subplots(1,1, subplot_kw={'projection': '3d'})
ax.plot(r[idx,:,0], r[idx,:,1], r[idx,:,2], color='C0', linewidth=1.0)
# --- equalize axes ---
x_limits = ax.get_xlim(); y_limits = ax.get_ylim(); z_limits = ax.get_zlim()
x_range = x_limits[1]-x_limits[0]; y_range = y_limits[1]-y_limits[0]; z_range = z_limits[1]-z_limits[0]
max_range = max(x_range, y_range, z_range)
x_mid = sum(x_limits)/2; y_mid = sum(y_limits)/2; z_mid = sum(z_limits)/2
ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
plt.show(block=False)


if save:
    r1 = r[idx,:,:]
    gamma1 = gamma[idx,:].reshape(1,3)
    df = pd.DataFrame (r1, columns = ['x', 'y', 'z']); df.to_csv(save_dir + 'r_1.csv')
    df = pd.DataFrame (gamma1, columns = ['act 1', 'act 2', 'act 3']); df.to_csv(save_dir + 'gamma_1.csv')


input("Debug breakpoint. Press Enter to exit...")