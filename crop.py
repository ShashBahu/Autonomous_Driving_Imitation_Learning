import cv2
import numpy as np

path = "/home/krg/Capstone/PilotNet_Train/pilotnet/recordings/2025-06-17@15:26:03/[29084, -0.0002485241857357323, 0.4403201937675476, 0.0].png"
image = cv2.imread(path)
h, w, _ = image.shape
top = int(0.5 * h)
bottom = int(0.8 * h)
left = int(0.3*w)
right = int(0.7*w)
cropped = image[top:bottom, left:right, :] 
h, w, _ = cropped.shape
print(h)
print(w)
resize = cv2.resize(cropped, (200, 66))
cv2.imshow("OG image",image)
cv2.imshow("Cropped image",cropped)
cv2.imshow("Resized",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()