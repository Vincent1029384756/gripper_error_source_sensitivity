import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_and_regress(csv_file, cx, cy1, cy2, cy3, cy4, save_path):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Extract columns based on the specified indices
    x = data.iloc[:, cx]
    y1 = data.iloc[:, cy1]
    y2 = data.iloc[:, cy2]
    y3 = data.iloc[:, cy3]
    y4 = data.iloc[:, cy4]


    # Plot Column y1 vs Column x (First and Second Half)
    plt.figure(figsize=(12, 12))

    # subplot for y1 
    plt.subplot(2, 2, 1)
    plt.scatter(x, y1, color='blue', label=f'{data.columns[1]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title(f'{data.columns[1]} vs {data.columns[0]}')

    # Split the data into first half and second half
    midpoint = len(x) // 2  # Calculate the midpoint to split the data
    x_first_half = x[:midpoint]
    y1_first_half = y1[:midpoint]
    x_second_half = x[midpoint:]
    y1_second_half = y1[midpoint:]

    # Polynomial regression for the first half
    coeffs1_first_half = np.polyfit(x_first_half, y1_first_half, 2)
    poly1_first_half = np.polyval(coeffs1_first_half, x_first_half)
    plt.plot(x_first_half, poly1_first_half, 'r', \
             label=f'First Half Fit: y={coeffs1_first_half[0]:.2f}x^2\
                +{coeffs1_first_half[1]:.2f}x+{coeffs1_first_half[2]:.2f}')

    # Polynomial regression for the second half
    coeffs1_second_half = np.polyfit(x_second_half, y1_second_half, 2)
    poly1_second_half = np.polyval(coeffs1_second_half, x_second_half)
    plt.plot(x_second_half, poly1_second_half, 'g', \
             label=f'Second Half Fit: y={coeffs1_second_half[0]:.2f}x^2\
                +{coeffs1_second_half[1]:.2f}x+{coeffs1_second_half[2]:.2f}')
    plt.legend()




    # subplot for y2 
    plt.subplot(2, 2, 2)
    plt.scatter(x, y2, color='blue', label=f'{data.columns[2]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[2])
    plt.title(f'{data.columns[2]} vs {data.columns[0]}')

    # Split the data into first half and second half
    midpoint = len(x) // 2  # Calculate the midpoint to split the data
    x_first_half = x[:midpoint]
    y2_first_half = y2[:midpoint]
    x_second_half = x[midpoint:]
    y2_second_half = y2[midpoint:]

    # Polynomial regression for the first half
    coeffs2_first_half = np.polyfit(x_first_half, y2_first_half, 2)
    poly2_first_half = np.polyval(coeffs2_first_half, x_first_half)
    plt.plot(x_first_half, poly2_first_half, 'r', \
             label=f'First Half Fit: y={coeffs2_first_half[0]:.2f}x^2\
                +{coeffs2_first_half[1]:.2f}x+{coeffs2_first_half[2]:.2f}')

    # Polynomial regression for the second half
    coeffs2_second_half = np.polyfit(x_second_half, y2_second_half, 2)
    poly2_second_half = np.polyval(coeffs2_second_half, x_second_half)
    plt.plot(x_second_half, poly2_second_half, 'g', \
             label=f'Second Half Fit: y={coeffs2_second_half[0]:.2f}x^2\
                +{coeffs2_second_half[1]:.2f}x+{coeffs2_second_half[2]:.2f}')
    plt.legend()



    # subplot for y3 
    plt.subplot(2, 2, 3)
    plt.scatter(x, y3, color='blue', label=f'{data.columns[3]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[3])
    plt.title(f'{data.columns[3]} vs {data.columns[0]}')

    # Split the data into first half and second half
    midpoint = len(x) // 2  # Calculate the midpoint to split the data
    x_first_half = x[:midpoint]
    y3_first_half = y3[:midpoint]
    x_second_half = x[midpoint:]
    y3_second_half = y3[midpoint:]

    # Polynomial regression for the first half
    coeffs3_first_half = np.polyfit(x_first_half, y3_first_half, 2)
    poly3_first_half = np.polyval(coeffs3_first_half, x_first_half)
    plt.plot(x_first_half, poly3_first_half, 'r', \
             label=f'First Half Fit: y={coeffs3_first_half[0]:.2f}x^2\
                +{coeffs3_first_half[1]:.2f}x+{coeffs3_first_half[2]:.2f}')

    # Polynomial regression for the second half
    coeffs3_second_half = np.polyfit(x_second_half, y3_second_half, 2)
    poly3_second_half = np.polyval(coeffs3_second_half, x_second_half)
    plt.plot(x_second_half, poly3_second_half, 'g', \
             label=f'Second Half Fit: y={coeffs3_second_half[0]:.2f}x^2\
                +{coeffs3_second_half[1]:.2f}x+{coeffs3_second_half[2]:.2f}')
    plt.legend()



    # subplot for y4
    plt.subplot(2, 2, 4)
    plt.scatter(x, y4, color='blue', label=f'{data.columns[4]} vs {data.columns[0]}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[4])
    plt.title(f'{data.columns[4]} vs {data.columns[0]}')

    # Split the data into first half and second half
    midpoint = len(x) // 2  # Calculate the midpoint to split the data
    x_first_half = x[:midpoint]
    y4_first_half = y4[:midpoint]
    x_second_half = x[midpoint:]
    y4_second_half = y4[midpoint:]

    # Polynomial regression for the first half
    coeffs4_first_half = np.polyfit(x_first_half, y4_first_half, 2)
    poly4_first_half = np.polyval(coeffs4_first_half, x_first_half)
    plt.plot(x_first_half, poly4_first_half, 'r', \
             label=f'First Half Fit: y={coeffs4_first_half[0]:.2f}x^2\
                +{coeffs4_first_half[1]:.2f}x+{coeffs4_first_half[2]:.2f}')

    # Polynomial regression for the second half
    coeffs4_second_half = np.polyfit(x_second_half, y4_second_half, 2)
    poly4_second_half = np.polyval(coeffs4_second_half, x_second_half)
    plt.plot(x_second_half, poly4_second_half, 'g', \
             label=f'Second Half Fit: y={coeffs4_second_half[0]:.2f}x^2\
                +{coeffs4_second_half[1]:.2f}x+{coeffs4_second_half[2]:.2f}')
    plt.legend()

    plt.tight_layout()
    
    #directory = '/mnt/newstorage/summer_project/results'
    plt.savefig(save_path)
