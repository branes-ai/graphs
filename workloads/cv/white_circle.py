import numpy as np
import cv2

# Create a 32x32 black image
image = np.zeros((32, 32), dtype=np.uint8)

# Draw a white circle in the center
cv2.circle(image, (16, 16), 10, (255,), -1)

print(image)

# Write the image to a file
cv2.imwrite('white_circle.png', image) 