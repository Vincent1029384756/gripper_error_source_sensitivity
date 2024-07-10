import cv2
import numpy as np
import os
import pandas as pd
import time

def calculate_angle_with_vertical(vx, vy):
    angle = np.arctan2(vy, vx)
    angle_with_vertical = np.degrees(angle)
    if angle_with_vertical < 0:
        angle_with_vertical += 180  # Ensure angle is in the range [0, 180)
    return 90 - angle_with_vertical

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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    
    #img_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    if lines is not None and len(lines) >= 1:
        lines = [line[0] for line in lines]  # Extract the line endpoints

        # Fit lines to the detected segments
        for x1, y1, x2, y2 in lines:
            vx = x2 - x1
            vy = y2 - y1
            angle_with_vertical = calculate_angle_with_vertical(vx, vy)
            return float(angle_with_vertical)  # Return image with drawn lines and angle
    else:
        print(f"Could not detect any lines to calculate the angle in {image_path}.")
        return None

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

# Define the folder containing the images, the ROI, and the output CSV file path
folder_path = '/mnt/newstorage/summer_project/Vincent/binary1'  # Replace with your folder path
roi = (263, 58, 209, 185)  # Define the Region of Interest (x, y, width, height)
output_csv_path = '/mnt/newstorage/summer_project/Vincent/angle1.csv'  # Replace with your output CSV file path

# Process the folder in batches and save the angle values to the CSV file
process_folder_in_batches(folder_path, roi, output_csv_path, batch_size=30, delay_between_batches=10)

