import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

mag1_folders = ['results_dx1', 'results_dy1', 'results_dz1']
mag2_folders = ['results_dx2', 'results_dy2', 'results_dz2']
png_names = ['d1_dq1_max', 'd1_dq2_max', 'd2_dq1_max', 'd2_dq2_max']
#png_names = ['d1_dq1_mean', 'd1_dq2_mean', 'd2_dq1_mean', 'd2_dq2_mean']
x_labels = ['Magnet1 Displacement [mm]', 'Magnet2 Displacement [mm]']
y_labels = ['Delta_q1 [deg]', 'Delta_q2 [deg]']
colors = ['blue', 'green', 'red']

root_path = '/mnt/newstorage/summer_project'

cy1 = 6 #column for dq1_max
cy2 = 8 #column for dq2_max

#process the mag_1 dq1
for i in range(len(mag1_folders)):
    csv_file = os.path.join(root_path, mag1_folders[i], 'result.csv')
    data = pd.read_csv(csv_file)

    x = data.iloc[:, i]
    y = data.iloc[:, cy1]

    plt.scatter(x, y, color = colors[i], label = f'{data.columns[cy1]} vs. {data.columns[i]} [mm]')

plt.xlabel(x_labels[0])
plt.ylabel(y_labels[0])
plt.title(f'{y_labels[0]} vs. different {x_labels[0]}')
plt.legend()
png_name = png_names[0]
png_path = os.path.join(root_path, 'images_present')
save_path = os.path.join(png_path, png_name)
plt.savefig(save_path)
plt.close()

#mag_1 dq2
for i in range(len(mag2_folders)):
    csv_file = os.path.join(root_path, mag1_folders[i], 'result.csv')
    data = pd.read_csv(csv_file)

    x = data.iloc[:, i]
    y = data.iloc[:, cy2]

    plt.scatter(x, y, color = colors[i], label = f'{data.columns[cy2]} vs. {data.columns[i]} [mm]')

plt.xlabel(x_labels[0])
plt.ylabel(y_labels[1])
plt.title(f'{y_labels[1]} vs. different {x_labels[0]}')
plt.legend()
png_name = png_names[1]
png_path = os.path.join(root_path, 'images_present')
save_path = os.path.join(png_path, png_name)
plt.savefig(save_path)
plt.close()

#mag_2 dq1
for i in range(len(mag1_folders), len(mag2_folders)+len(mag1_folders)):
    csv_file = os.path.join(root_path, mag2_folders[i-len(mag1_folders)], 'result.csv')
    data = pd.read_csv(csv_file)

    x = data.iloc[:, i]
    y = data.iloc[:, cy1]

    plt.scatter(x, y, color = colors[i-len(mag1_folders)], label = f'{data.columns[cy1]} vs. {data.columns[i]} [mm]')

plt.xlabel(x_labels[1])
plt.ylabel(y_labels[0])
plt.title(f'{y_labels[0]} vs. different {x_labels[1]}')
plt.legend()
png_name = png_names[2]
png_path = os.path.join(root_path, 'images_present')
save_path = os.path.join(png_path, png_name)
plt.savefig(save_path)
plt.close()

#mag_2 dq2
for i in range(len(mag1_folders), len(mag2_folders)+len(mag1_folders)):
    csv_file = os.path.join(root_path, mag2_folders[i-len(mag1_folders)], 'result.csv')
    data = pd.read_csv(csv_file)

    x = data.iloc[:, i]
    y = data.iloc[:, cy2]

    plt.scatter(x, y, color = colors[i-len(mag1_folders)], label = f'{data.columns[cy2]} vs. {data.columns[i]} [mm]')

plt.xlabel(x_labels[1])
plt.ylabel(y_labels[1])
plt.title(f'{y_labels[1]} vs. different {x_labels[1]}')
plt.legend()
png_name = png_names[3]
png_path = os.path.join(root_path, 'images_present')
save_path = os.path.join(png_path, png_name)
plt.savefig(save_path)
plt.close()