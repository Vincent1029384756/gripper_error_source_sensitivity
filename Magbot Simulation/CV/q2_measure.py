import cv2
import numpy as np
import os
import pandas as pd
import time
import re

def calculate_angle(line1, line2):
    def line_angle(vx, vy):
        return np.arctan2(vy, vx)
    
    angle1 = line_angle(line1[0], line1[1])
    angle2 = line_angle(line2[0], line2[1])
    angle = np.degrees(np.abs(angle1 - angle2))
    if angle > 90:
        angle = 180 - angle
    return angle

def process_image(image_path, roi):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x, y, w, h = roi
    roi_img = img[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    # Use morphological operations to enhance the edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Use the Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    
    img_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    if lines is not None and len(lines) >= 2:
        lines = [line[0] for line in lines]  # Extract the line endpoints

        # Fit lines to the detected segments
        fitted_lines = []
        for x1, y1, x2, y2 in lines:
            vx = x2 - x1
            vy = y2 - y1
            fitted_lines.append(([vx, vy], (x1, y1)))
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check if at least two lines are detected
        if len(fitted_lines) >= 2:
            line1 = fitted_lines[0]
            line2 = fitted_lines[1]
            angle_between_links = calculate_angle(line1[0], line2[0])
            return float(angle_between_links)  # Return image with drawn lines and angle
    else:
        print(f"Could not detect enough lines to calculate the angle in {image_path}.")
        return None
    
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
def process_folder_in_batches(folder_path, roi, output_csv_path, batch_size, delay_between_batches):
    angles = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        for image_file in batch_files:
            image_path = os.path.join(folder_path, image_file)
            angle = process_image(image_path, roi)
            if angle is not None:
                angles.append(angle)
            else:
                angles.append(np.nan)  # Append NaN if no angle was detected

        # Save the angles to a CSV file without a header after each batch
        df = pd.DataFrame(angles)
        df.to_csv(output_csv_path, header=False, index=False)
        print(f"Processed batch {i//batch_size + 1}, angles saved to {output_csv_path}")

        # Introduce a delay between batches
        time.sleep(delay_between_batches)

def process_folder(folder_path, roi, output_csv_path):
    angles = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))], key=natural_sort_key)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        angle = process_image(image_path, roi)
        if angle is not None:
            angles.append(angle)
        else:
            angles.append(np.nan)  # Append NaN if no angle was detected

        # Save the angles to a CSV file without a header after each batch
        df = pd.DataFrame(angles)
        df.to_csv(output_csv_path, header=False, index=False)
    print(f"Processed folder {folder_path}, angles saved to {output_csv_path}")


# Define the folder containing the images, the ROI, and the output CSV file path
folder_path = '/mnt/newstorage/summer_project/Vincent/binary5'  # Replace with your folder path
roi = (370,286,74,104)  # Define the Region of Interest (x, y, width, height)
output_csv_path = '/mnt/newstorage/summer_project/Vincent/angle5_fast.csv'  # Replace with your output CSV file path

# Process the folder in batches and save the angle values to the CSV file
#process_folder_in_batches(folder_path, roi, output_csv_path, batch_size=30, delay_between_batches=10)
process_folder(folder_path, roi, output_csv_path)
