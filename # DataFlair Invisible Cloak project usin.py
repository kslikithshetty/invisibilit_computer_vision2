import cv2
import time
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Allow camera to warm up and capture background
print("Capturing background in 3 seconds. Stay out of frame!")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)
    
ret, background = cap.read()
if background is None:
    print("Error: Could not capture background.")
    exit()

# Kernels for noise removal
open_kernel = np.ones((3, 3), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)

def filter_mask(mask):
    # Apply morphological operations to clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    # Apply Gaussian blur for smoother edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

print("Setup complete. Hold up a green cloth to see the magic!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    
    # Convert current frame to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Green color range (may need adjustment based on your lighting and cloth)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Create initial mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the mask
    mask = filter_mask(mask)
    
    # Invert mask to get the non-green areas
    inv_mask = cv2.bitwise_not(mask)
    
    # Combine background with current frame using the masks
    # Where mask=1 (green cloth), show background
    background_part = cv2.bitwise_and(background, background, mask=mask)
    
    # Where mask=0 (not green cloth), show current frame
    frame_part = cv2.bitwise_and(frame, frame, mask=inv_mask)
    
    # Combine both parts
    result = cv2.add(background_part, frame_part)
    
    # Show mask for debugging (this displays the live mask you requested)
    cv2.imshow("Mask", mask)
    
    # Show result
    cv2.imshow("Invisible Cloak", result)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()