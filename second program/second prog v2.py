import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load Image

image = cv2.imread("face2.png")
image = cv2.resize(image, (600, 600))
output = image.copy()


# Face Detection

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No face detected")
    exit()

(x, y, w, h) = faces[0]
face = image[y:y+h, x:x+w]


# Preprocessing

blur = cv2.GaussianBlur(face, (7,7), 0)


# Skin Mask (YCrCb)

ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
lower_skin = np.array([0, 133, 77])
upper_skin = np.array([255, 173, 127])
skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)


# Acne Color Detection (HSV)

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 + mask2


# Combine Skin + Acne Mask

acne_mask = cv2.bitwise_and(red_mask, skin_mask)

kernel = np.ones((5,5), np.uint8)
acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_OPEN, kernel)
acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_DILATE, kernel)


# Contour Detection

contours, _ = cv2.findContours(
    acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

acne_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 80 or area > 1500:
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.4:
        continue

    x1, y1, w1, h1 = cv2.boundingRect(cnt)
    aspect_ratio = w1 / h1
    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
        continue

    acne_count += 1
    cv2.rectangle(face, (x1,y1), (x1+w1,y1+h1), (0,255,0), 2)


# Severity Classification

face_area = w * h
density = acne_count / face_area

if density < 0.00003:
    severity = "Mild Acne"
elif density < 0.00008:
    severity = "Moderate Acne"
else:
    severity = "Severe Acne"


# Display Text

cv2.putText(face, f"Acne Count: {acne_count}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

cv2.putText(face, severity, (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

# Put processed face back
output[y:y+h, x:x+w] = face


# Show Result

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
