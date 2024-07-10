import cv2

# Load the source image
source_image_path = '/mnt/newstorage/summer_project/Vincent/binary5/Frame100.JPG'
img = cv2.imread(source_image_path)

# Display the image and select the ROI
roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)

# Extract the coordinates of the ROI
roi_x, roi_y, roi_w, roi_h = roi

print(f"Selected ROI - x: {roi_x}, y: {roi_y}, width: {roi_w}, height: {roi_h}")

# Draw a rectangle around the selected ROI for visualization
cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

# Display the image with the ROI
cv2.imshow("Selected ROI", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# You can now use the coordinates roi_x, roi_y, roi_w, roi_h in your further processing
