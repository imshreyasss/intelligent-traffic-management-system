import cv2
import numpy as np 
import time

# Create a background subtraction object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Open video file or capture device
cap = cv2.VideoCapture("33.mp4")

# Initialize variables
count = 0
width = int(cap.get(3))
height = int(cap.get(4))
vehicles = []  # List to store detected vehicles

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If frame was not captured properly, break out of loop
    if not ret:
        break
    
    # Apply background subtraction to obtain foreground mask
    fgmask = fgbg.apply(frame)
    
    # Apply morphological transformations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # Find object in foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over all contours and count vehicles
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # contour area smaller than a threshold, ignore
        if area < 20000:
            continue
        
        # Draw bounding box around contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Check if the vehicle has already been counted
        new_vehicle = True
        for vehicle in vehicles:
            if abs(x - vehicle[0]) < 50 and abs(y - vehicle[1]) < 50:
                new_vehicle = False
                break
        
        # new vehicle add to count
        if new_vehicle:
            count += 1
            vehicles.append((x, y))
    
    # Display frame and count
    cv2.putText(frame, "Count: {}".format(count), (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    
    # stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video
cap.release()
cv2.destroyAllWindows()

# Print final count of all vehicles detected
print("vehicle_count:", count)

# Countdown timer based on vehicle count
if count == 0:
    countdown_seconds = 0
elif count < 5:
    countdown_seconds = 5
elif 5 <= count < 20:
    countdown_seconds = 10
else:
    countdown_seconds = 15

for i in range(countdown_seconds, 0, -1):
    print(i)
    time.sleep(1)

print("RED")
