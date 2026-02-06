import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("face3.png")
image = cv2.resize(image, (500, 500))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

acne_count = 0

for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        acne_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

if acne_count < 5:
    severity = "Mild Acne"
elif acne_count < 17:
    severity = "Moderate Acne"
else:
    severity = "Severe Acne"

cv2.putText(image, f"Acne Count: {acne_count}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

cv2.putText(image, severity, (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
