import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_and_regress(csv_file, c_x, c_y1, c_y2):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Extract columns based on the specified indices
    x = data.iloc[:, c_x]
    y1 = data.iloc[:, c_y1]
    y2 = data.iloc[:, c_y2]

    # Split data into two halves
    mid_point = len(x) // 2
    x1, x2 = x[:mid_point], x[mid_point:]
    y1_first_half, y1_second_half = y1[:mid_point], y1[mid_point:]
    y2_first_half, y2_second_half = y2[:mid_point], y2[mid_point:]

    # Plot Column y1 vs Column x (First and Second Half)
    plt.figure(figsize=(12, 12))

    # First subplot for y1 first half
    plt.subplot(2, 2, 1)
    plt.scatter(x1, y1_first_half, color='blue', label=f'{data.columns[c_y1]} (First Half) vs {data.columns[c_x]}')
    plt.xlabel(data.columns[c_x])
    plt.ylabel(data.columns[c_y1])
    plt.title(f'{data.columns[c_y1]} (First Half) vs {data.columns[c_x]}')

    # Polynomial regression for y1 vs x (First Half)
    coeffs1_first = np.polyfit(x1, y1_first_half, 2)
    poly1_first = np.polyval(coeffs1_first, x1)
    plt.plot(x1, poly1_first, 'r', label=f'Fit: y={coeffs1_first[0]:.2f}x^2+{coeffs1_first[1]:.2f}x+{coeffs1_first[2]:.2f}')
    plt.legend()

    # Second subplot for y1 second half
    plt.subplot(2, 2, 2)
    plt.scatter(x2, y1_second_half, color='blue', label=f'{data.columns[c_y1]} (Second Half) vs {data.columns[c_x]}')
    plt.xlabel(data.columns[c_x])
    plt.ylabel(data.columns[c_y1])
    plt.title(f'{data.columns[c_y1]} (Second Half) vs {data.columns[c_x]}')

    # Polynomial regression for y1 vs x (Second Half)
    coeffs1_second = np.polyfit(x2, y1_second_half, 2)
    poly1_second = np.polyval(coeffs1_second, x2)
    plt.plot(x2, poly1_second, 'r', label=f'Fit: y={coeffs1_second[0]:.2f}x^2+{coeffs1_second[1]:.2f}x+{coeffs1_second[2]:.2f}')
    plt.legend()

    # Third subplot for y2 first half
    plt.subplot(2, 2, 3)
    plt.scatter(x1, y2_first_half, color='green', label=f'{data.columns[c_y2]} (First Half) vs {data.columns[c_x]}')
    plt.xlabel(data.columns[c_x])
    plt.ylabel(data.columns[c_y2])
    plt.title(f'{data.columns[c_y2]} (First Half) vs {data.columns[c_x]}')

    # Polynomial regression for y2 vs x (First Half)
    coeffs2_first = np.polyfit(x1, y2_first_half, 2)
    poly2_first = np.polyval(coeffs2_first, x1)
    plt.plot(x1, poly2_first, 'r', label=f'Fit: y={coeffs2_first[0]:.2f}x^2+{coeffs2_first[1]:.2f}x+{coeffs2_first[2]:.2f}')
    plt.legend()

    # Fourth subplot for y2 second half
    plt.subplot(2, 2, 4)
    plt.scatter(x2, y2_second_half, color='green', label=f'{data.columns[c_y2]} (Second Half) vs {data.columns[c_x]}')
    plt.xlabel(data.columns[c_x])
    plt.ylabel(data.columns[c_y2])
    plt.title(f'{data.columns[c_y2]} (Second Half) vs {data.columns[c_x]}')

    # Polynomial regression for y2 vs x (Second Half)
    coeffs2_second = np.polyfit(x2, y2_second_half, 2)
    poly2_second = np.polyval(coeffs2_second, x2)
    plt.plot(x2, poly2_second, 'r', label=f'Fit: y={coeffs2_second[0]:.2f}x^2+{coeffs2_second[1]:.2f}x+{coeffs2_second[2]:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.show()