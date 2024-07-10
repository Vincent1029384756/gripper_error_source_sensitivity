import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
    img_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    if lines is not None and len(lines) >= 1:
        lines = [line[0] for line in lines]  # Extract the line endpoints

        # Fit lines to the detected segments
        for x1, y1, x2, y2 in lines:
            vx = x2 - x1
            vy = y2 - y1
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            angle_with_vertical = calculate_angle_with_vertical(vx, vy)
            #cv2.putText(img_color, f'Angle: {angle_with_vertical:.2f} degrees', (10, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return img_color, float(angle_with_vertical)  # Return image with drawn lines and angle
    else:
        print(f"Could not detect any lines to calculate the angle in {image_path}.")
        return img_color, None

image_path = '/mnt/newstorage/summer_project/Vincent/binary1/Frame7000.JPG'
roi = (263, 58, 209, 185)  # Define the Region of Interest (x, y, width, height)

processed_img, angle = process_image(image_path, roi)

# Display the processed image with detected lines and angle
plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Angle: {angle:.2f} degrees' if angle else 'No angle detected')
plt.axis('off')
plt.show()

print(f'Detected Angle: {angle:.2f} degrees' if angle else 'No angle detected')
