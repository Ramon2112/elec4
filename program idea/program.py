import cv2
import numpy as np

def detect_cracks(image_path):
    # Step 1: Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found! Check the file path.")
        return

    original = img.copy()

    # Step 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Contrast enhancement
    gray = cv2.equalizeHist(gray)

    # Step 4: Noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 5: Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 6: Morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 7: Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Filter contours
    crack_mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 5000:
            cv2.drawContours(crack_mask, [cnt], -1, 255, thickness=1)

    # Step 9: Highlight cracks in red
    result = original.copy()
    result[crack_mask == 255] = [0, 0, 255]

    # Save results
    cv2.imwrite("crack_edges.png", edges)
    cv2.imwrite("crack_mask.png", crack_mask)
    cv2.imwrite("crack_detected.png", result)

    # Show results
    cv2.imshow("Original", original)
    cv2.imshow("Edges", edges)
    cv2.imshow("Crack Mask", crack_mask)
    cv2.imshow("Detected Cracks", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===== RUN =====
image_path = input("Enter wall/building image path: ")
# Example input:
# C:/Users/romualdo/PycharmProjects/elec4/program idea/Crack_test.png
detect_cracks(image_path)
