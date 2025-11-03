import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def color_filter(img_bgr):
    hls = to_hls(img_bgr)
    white_mask = cv2.inRange(hls, (0, 200, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(hls, (10, 80, 100), (40, 255, 255))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=combined_mask)
    return result, combined_mask

def sobel_gradients(gray, ksize=3):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    direction = np.arctan2(np.abs(gy), np.abs(gx))
    return gx, gy, mag, direction

def save_results(mag, direction):
    cv2.imwrite("results/gradient_outputs/magnitude.png", mag)
    direction_viz = np.uint8(255 * (direction / np.pi))
    cv2.imwrite("results/gradient_outputs/direction.png", direction_viz)
    print("✅ Saved gradient magnitude and direction images!")

def run_gradient_pipeline(image_path):
    img = load_image(image_path)
    gray = to_gray(img)
    filtered, mask = color_filter(img)
    gx, gy, mag, direction = sobel_gradients(gray)
    save_results(mag, direction)
    return mag, direction

if __name__ == "__main__":
    run_gradient_pipeline("data/road1.jpg")
