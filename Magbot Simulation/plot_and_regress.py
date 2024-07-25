import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_and_regress(csv_file, x, c_y1, c_y2):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Extract columns (assuming the headers are 'Column1', 'Column2', and 'Column3')
    x = x
    y1 = data.iloc[:, c_y1]
    y2 = data.iloc[:, c_y2]

    # Split data into two halves
    mid_point = len(x) // 2
    x1, x2 = x[:mid_point], x[mid_point:]
    y1_first_half, y1_second_half = y1[:mid_point], y1[mid_point:]
    y2_first_half, y2_second_half = y2[:mid_point], y2[mid_point:]

    # Plot Column 2 vs Column 1 (First and Second Half)
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.scatter(x1, y1_first_half, color='blue', label=f'{data.columns[c_y1]} (First Half) vs delta_theta')
    plt.xlabel('delta_theta')
    plt.ylabel(data.columns[c_y1])
    plt.title(f'{data.columns[c_y1]} (First Half) vs delta_theta')

    # Linear regression for Column 2 vs Column 1 (First Half)
    slope1_first, intercept1_first, r_value1_first, p_value1_first, std_err1_first = linregress(x1, y1_first_half)
    plt.plot(x1, intercept1_first + slope1_first * x1, 'r', label=f'Fit: y={intercept1_first:.2f}+{slope1_first:.2f}x')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(x2, y1_second_half, color='blue', label=f'{data.columns[c_y1]} (Second Half) vs delta_theta')
    plt.xlabel('delta_theta')
    plt.ylabel(data.columns[c_y1])
    plt.title(f'{data.columns[c_y1]} (Second Half) vs delta_theta')

    # Linear regression for Column 2 vs Column 1 (Second Half)
    slope1_second, intercept1_second, r_value1_second, p_value1_second, std_err1_second = linregress(x2, y1_second_half)
    plt.plot(x2, intercept1_second + slope1_second * x2, 'r', label=f'Fit: y={intercept1_second:.2f}+{slope1_second:.2f}x')
    plt.legend()

    # Plot Column 3 vs Column 1 (First and Second Half)
    plt.subplot(2, 2, 3)
    plt.scatter(x1, y2_first_half, color='green', label=f'{data.columns[c_y2]} (First Half) vs delta_theta')
    plt.xlabel('delta_theta')
    plt.ylabel(data.columns[c_y2])
    plt.title(f'{data.columns[c_y2]} (First Half) vs delta_theta')

    # Linear regression for Column 3 vs Column 1 (First Half)
    slope2_first, intercept2_first, r_value2_first, p_value2_first, std_err2_first = linregress(x1, y2_first_half)
    plt.plot(x1, intercept2_first + slope2_first * x1, 'r', label=f'Fit: y={intercept2_first:.2f}+{slope2_first:.2f}x')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(x2, y2_second_half, color='green', label=f'{data.columns[c_y2]} (Second Half) vs delta_theta')
    plt.xlabel('delta_theta')
    plt.ylabel(data.columns[c_y2])
    plt.title(f'{data.columns[c_y2]} (Second Half) vs delta_theta')

    # Linear regression for Column 3 vs Column 1 (Second Half)
    slope2_second, intercept2_second, r_value2_second, p_value2_second, std_err2_second = linregress(x2, y2_second_half)
    plt.plot(x2, intercept2_second + slope2_second * x2, 'r', label=f'Fit: y={intercept2_second:.2f}+{slope2_second:.2f}x')
    plt.legend()

    plt.tight_layout()
    plt.show()