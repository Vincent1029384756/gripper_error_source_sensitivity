import cv2
import numpy as np
import matplotlib.pyplot as plt

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
            #cv2.putText(img_color, f'Angle: {angle_between_links:.2f} degrees', (10, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return img_color, float(angle_between_links)  # Return image with drawn lines and angle
    else:
        print(f"Could not detect enough lines to calculate the angle in {image_path}.")
        return img_color, None

image_path = '/mnt/newstorage/summer_project/Vincent/binary5/Frame4685.JPG'
roi = (370,286,74,104)  # Define the Region of Interest (x, y, width, height)

processed_img, angle = process_image(image_path, roi)

# Display the processed image with detected lines and angle
plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Angle: {angle:.2f} degrees' if angle else 'No angle detected')
plt.axis('off')
plt.show()

print(f'Detected Angle: {angle:.2f} degrees' if angle else 'No angle detected')