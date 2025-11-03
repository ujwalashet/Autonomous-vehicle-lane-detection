import cv2
import numpy as np
import os

# Base directory (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "road1.jpg")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)


def region_of_interest(image):
    """Applies mask to keep only the lower road portion of the image."""
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    # Trapezoid coordinates (focuses tightly on the road area)
    polygon = np.array([[
        (int(0.15 * width), height),
        (int(0.45 * width), int(0.65 * height)),
        (int(0.55 * width), int(0.65 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def process_image(image):
    """Performs lane detection and saves all intermediate results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Stronger thresholds to ignore background
    edges = cv2.Canny(blur, 100, 200)

    # Apply region of interest
    masked_edges = region_of_interest(edges)

    # Hough Transform for lane lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=50
    )

    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Overlay detected lines on original image
    combo = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # Save results
    cv2.imwrite(os.path.join(RESULTS_DIR, "lane_edges.jpg"), masked_edges)
    cv2.imwrite(os.path.join(RESULTS_DIR, "lane_masked.jpg"), line_image)
    cv2.imwrite(os.path.join(RESULTS_DIR, "lane_detected.jpg"), combo)

    print("✅ All results saved in:", RESULTS_DIR)
    return combo


if __name__ == "__main__":
    image = cv2.imread(DATA_PATH)
    if image is None:
        raise FileNotFoundError(f"❌ Could not find input image at {DATA_PATH}")

    result = process_image(image)

    # Show result window
    cv2.imshow("Final Lane Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

