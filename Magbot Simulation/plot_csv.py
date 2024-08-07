import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_and_regress(csv_file):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Extract columns based on the specified indices
    x = data.iloc[:, 0]
    y1 = data.iloc[:, 1]
    y2 = data.iloc[:, 2]
    y3 = data.iloc[:, 3]
    y4 = data.iloc[:, 4]


    # Plot Column y1 vs Column x (First and Second Half)
    plt.figure(figsize=(12, 12))

    # subplot for y1 
    plt.subplot(2, 2, 1)
    plt.scatter(x, y1, color='blue', label=f'{data.columns[1]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title(f'{data.columns[1]} vs {data.columns[0]}')

    # Polynomial regression for y1 vs x 
    coeffs1 = np.polyfit(x, y1, 2)
    poly1 = np.polyval(coeffs1, x)
    plt.plot(x, poly1, 'r', label=f'Fit: y={coeffs1[0]:.2f}x^2+{coeffs1[1]:.2f}x+{coeffs1[2]:.2f}')
    plt.legend()


    # subplot for y2 
    plt.subplot(2, 2, 2)
    plt.scatter(x, y2, color='blue', label=f'{data.columns[2]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[2])
    plt.title(f'{data.columns[2]} vs {data.columns[0]}')

    # Polynomial regression for y2 vs x 
    coeffs2 = np.polyfit(x, y2, 2)
    poly2 = np.polyval(coeffs2, x)
    plt.plot(x, poly2, 'r', label=f'Fit: y={coeffs2[0]:.2f}x^2+{coeffs2[1]:.2f}x+{coeffs2[2]:.2f}')
    plt.legend()

    # subplot for y3 
    plt.subplot(2, 2, 3)
    plt.scatter(x, y3, color='blue', label=f'{data.columns[3]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[3])
    plt.title(f'{data.columns[3]} vs {data.columns[0]}')

    # Polynomial regression for y3 vs x 
    coeffs3 = np.polyfit(x, y3, 2)
    poly3 = np.polyval(coeffs3, x)
    plt.plot(x, poly3, 'r', label=f'Fit: y={coeffs3[0]:.2f}x^2+{coeffs3[1]:.2f}x+{coeffs3[2]:.2f}')
    plt.legend()


    # subplot for y4
    plt.subplot(2, 2, 4)
    plt.scatter(x, y4, color='blue', label=f'{data.columns[4]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[4])
    plt.title(f'{data.columns[4]} vs {data.columns[0]}')

    # Polynomial regression for y4 vs x 
    coeffs4 = np.polyfit(x, y4, 2)
    poly4 = np.polyval(coeffs4, x)
    plt.plot(x, poly4, 'r', label=f'Fit: y={coeffs4[0]:.2f}x^2+{coeffs4[1]:.2f}x+{coeffs4[2]:.2f}')
    plt.legend()

    plt.tight_layout()
    
    directory = '/mnt/newstorage/summer_project/results'
    filename = 'results.png'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
  