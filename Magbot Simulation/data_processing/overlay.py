import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

root_path = '/mnt/newstorage/summer_project'
folder1 = 'results_dx1'
folder2 = 'results_dy1'
folder3 = 'results_dz1'

csv_1 = os.path.join(root_path, folder1, 'result.csv')
csv_2 = os.path.join(root_path, folder2, 'result.csv')
csv_3 = os.path.join(root_path, folder3, 'result.csv')

data1 = pd.read_csv(csv_1)
data2 = pd.read_csv(csv_2)
data3 = pd.read_csv(csv_3)

cx = 0
cy = 6

x = data1.iloc[:, cx]
y1 = data1.iloc[:, cy]
y2 = data2.iloc[:, cy]
y3 = data3.iloc[:, cy]

plt.scatter(x, y1, color = 'blue', label = f'{data1.columns[cy]} vs {data1.columns[0]} [mm]')
plt.scatter(x, y2, color = 'red', label = f'{data2.columns[cy]} vs {data2.columns[1]} [mm]')
plt.scatter(x, y3, color = 'green', label = f'{data3.columns[cy]} vs {data3.columns[2]} [mm]')
plt.xlabel('Magnet1 Displacement [mm]')
plt.ylabel('delta_q1 [deg]')
plt.title('Delta_q1 vs. different magnet1 displacement')
plt.legend()
#plt.show()

save_path = '/mnt/newstorage/summer_project/Magbot Simulation/data_processing/overlay_d1.png'
plt.savefig(save_path)
