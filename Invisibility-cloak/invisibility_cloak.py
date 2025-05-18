import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)

# Capture static background (no one in the frame)
for i in range(60):
    ret, background = cap.read()
    if not ret:
        continue
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2

    # Morphological transformations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # Invert red mask to get part not covered by cloak
    inverse_mask = cv2.bitwise_not(red_mask)

    # Cloak-covered area replaced by background
    background_part = cv2.bitwise_and(background, background, mask=red_mask)

    # Area not covered by cloak remains as-is
    current_part = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine both
    final_output = cv2.add(background_part, current_part)

    cv2.imshow("Invisibility Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
