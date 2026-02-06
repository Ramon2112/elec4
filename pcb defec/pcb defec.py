import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ===================== LOAD IMAGES =====================
ref_img = cv2.imread("reference2.png")
test_img = cv2.imread("test2.png")

# Resize test image if needed
test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

# ===================== IMAGE ALIGNMENT (ORB) =====================
def align_images(image, template):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(gray_temp, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:50]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(image, matrix,
                                  (template.shape[1], template.shape[0]))
    return aligned

aligned_test = align_images(test_img, ref_img)

# ===================== PREPROCESS =====================
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)

ref_blur = cv2.GaussianBlur(ref_gray, (5,5), 0)
test_blur = cv2.GaussianBlur(test_gray, (5,5), 0)

# ===================== SSIM COMPARISON =====================
score, diff = ssim(ref_blur, test_blur, full=True)
print("SSIM Similarity Score:", score)

diff = (diff * 255).astype("uint8")

# ===================== THRESHOLDING =====================
thresh = cv2.adaptiveThreshold(
    diff, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# Morphological cleanup
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# ===================== CONTOUR DETECTION =====================
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = aligned_test.copy()
fault_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 300 < area < 5000:   # filter noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(output, "PCB Fault", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        fault_count += 1

print("Detected Fault Regions:", fault_count)

# ===================== DISPLAY RESULTS =====================
cv2.imshow("Reference PCB", ref_img)
cv2.imshow("Aligned Test PCB", aligned_test)
cv2.imshow("Difference Map", diff)
cv2.imshow("Fault Mask", thresh)
cv2.imshow("Detected PCB Faults", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
