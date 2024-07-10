import cv2
import os

'''
This program transforms full colr caaptured frames into pure black and white.
This will facilitate template fitting for angle detection.
'''

def convert_images_to_black_and_white(input_folder, output_folder, threshold_value):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct the full file path
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image (optional: you can add more extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Load the color image
            color_image = cv2.imread(file_path)

            # Convert the color image to grayscale
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Apply binary thresholding
            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

            # Construct the output file path
            output_file_path = os.path.join(output_folder, filename)

            # Save the binary image to the output folder
            cv2.imwrite(output_file_path, binary_image)

            print(f"Processed {filename}")

# Define input and output folders
input_folder = '/mnt/newstorage/summer_project/Vincent/CapturedVideoFrames_5'  
output_folder = '/mnt/newstorage/summer_project/Vincent/binary5' 

# Call the function to process images
convert_images_to_black_and_white(input_folder, output_folder, 100)