import cv2
import numpy as np

# Create a background subtraction object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Open camera device
cap = cv2.VideoCapture(0)

# Initialize variables
count = 0
width = int(cap.get(3))
height = int(cap.get(4))

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # If frame was not captured properly, break out of loop
    if not ret:
        break
    
    # Apply background subtraction to obtain foreground mask
    fgmask = fgbg.apply(frame)
    
    # Apply morphological transformations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over all contours and count vehicles
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # If contour area is smaller than a threshold, ignore it
        if area < 500:
            continue
        
        # Draw bounding box around contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Increment vehicle count
        count += 1
    
    # Display frame and count
    cv2.putText(frame, "Count: {}".format(count), (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera device and close all windows
cap.release()
cv2.destroyAllWindows()
