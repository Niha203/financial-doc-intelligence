"""
Image preprocessing for receipt OCR.
Applies targeted fixes based on common receipt image quality issues:
- Low contrast (BHPetrol-style light prints)
- Skew / rotation
- Noise and stamps
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew using Hough line detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=100,
                             minLineLength=100,
                             maxLineGap=10)
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image

    logger.debug(f"Deskewing by {median_angle:.2f} degrees")
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement — fixes faint/light prints."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise — fixes stamps, handwriting artifacts."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding — handles uneven lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )


def upscale(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale small/low-res images for better OCR."""
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)


def preprocess_receipt(image_path: str,
                       debug_dir: str = None) -> np.ndarray:
    """
    Full preprocessing pipeline for a receipt image.
    Returns preprocessed image array ready for EasyOCR.

    Pipeline:
    1. Load
    2. Upscale if small
    3. Denoise
    4. Deskew
    5. Contrast enhance (CLAHE)
    6. Binarize (adaptive threshold)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    logger.debug(f"Original size: {w}x{h}")

    if w < 600 or h < 800:
        image = upscale(image, scale=2.0)
        logger.debug(f"Upscaled to: {image.shape[1]}x{image.shape[0]}")

    image = denoise(image)
    image = deskew(image)
    image = enhance_contrast(image)
    image = binarize(image)

    if debug_dir:
        debug_path = Path(debug_dir) / \
            (Path(image_path).stem + "_preprocessed.png")
        cv2.imwrite(str(debug_path), image)
        logger.debug(f"Saved preprocessed image: {debug_path}")

    return image