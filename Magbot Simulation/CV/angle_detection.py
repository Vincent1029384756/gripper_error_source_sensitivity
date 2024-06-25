import cv2
import numpy as np

# Load the source image
source_image_path = '/mnt/newstorage/summer_project/Magbot Simulation/CV/gripper.jpg'
img = cv2.imread(source_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the template images
template1_path = '/mnt/newstorage/summer_project/Magbot Simulation/CV/joint1.jpg'
template2_path = '/mnt/newstorage/summer_project/Magbot Simulation/CV/base.jpg'
template1 = cv2.imread(template1_path, 0)
template2 = cv2.imread(template2_path, 0)

# Perform template matching for the first template (link 1)
result1 = cv2.matchTemplate(gray, template1, cv2.TM_CCOEFF_NORMED)
threshold = 0.8  # Lower the threshold to detect more matches
loc1 = np.where(result1 >= threshold)
centers1 = [(pt[0] + template1.shape[1] // 2, pt[1] + template1.shape[0] // 2) for pt in zip(*loc1[::-1])]
print(f"Detected centers for link 1: {centers1}")

# Perform template matching for the second template (link 2)
result2 = cv2.matchTemplate(gray, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(result2 >= threshold)
centers2 = [(pt[0] + template2.shape[1] // 2, pt[1] + template2.shape[0] // 2) for pt in zip(*loc2[::-1])]
print(f"Detected centers for link 2: {centers2}")

# Ensure that we have at least one center for each link
if len(centers1) >= 2 and len(centers2) > 0:
    pt2 = centers2[0]  # Use the first detected center for link 2

    # Draw the centers
    for center in centers1:
        cv2.circle(img, center, 5, (0, 255, 0), -1)  # Draw the center point for link 1
    cv2.circle(img, pt2, 5, (255, 0, 0), -1)  # Draw the center point for link 2

    # Fit a line to the centers of link 1
    def fit_line(points):
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        return np.polyfit(x_coords, y_coords, 1)

    line1 = fit_line(centers1)  # Fit a line using all points of link 1

    # Draw the fitted line for link 1
    x_coords = [point[0] for point in centers1]
    x0, y0 = int(min(x_coords)), int(line1[1] + line1[0] * min(x_coords))
    x1, y1 = int(max(x_coords)), int(line1[1] + line1[0] * max(x_coords))
    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Draw a line from the detected center of link 2 to one of the points on link 1
    pt1 = centers1[0]  # Use the first center of link 1 for drawing the line
    cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    # Calculate the angle between the two lines
    angle1 = np.arctan(line1[0])
    angle2 = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    angle_between_links = np.degrees(abs(angle1 - angle2))
    if angle_between_links > 90:
        angle_between_links = 180 - angle_between_links

    print(f'Angle between the two links: {angle_between_links:.2f} degrees')

    # Print the angle on the image
    cv2.putText(img, f'Angle: {angle_between_links:.2f} degrees', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
else:
    print("Could not detect enough points to fit lines and calculate the angle.")

# Save and display the processed image
cv2.imwrite('gripper_with_angle.jpg', img)

# Resize the image to fit the screen
screen_res = 1920, 1080  # Example screen resolution
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

# Resize the window
cv2.namedWindow('Gripper Angle', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gripper Angle', window_width, window_height)

# Show the resized image
cv2.imshow('Gripper Angle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
