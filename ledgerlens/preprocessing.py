"""
ledgerlens.preprocessing
========================
Deterministic, dependency-light image preprocessing utilities used ahead of
OCR / VLM extraction. Every function here is a *pure function*: given the
same input array it always returns the same output, with no network calls,
no hidden state, and no reliance on external services. That keeps this
stage fully unit-testable and auditable on its own.

Pipeline order used by ledgerlens.pipeline:
    1. tight_crop      - remove background/margins around the item
    2. super_resolve    - upscale small crops (2x-4x) before OCR
    3. to_grayscale     - drop color channels; OCR/VLM want luminance
    4. autocontrast     - stretch the histogram to use the full range
    5. otsu_binarize    - hand-rolled Otsu threshold -> black/white image

Only OpenCV, Pillow, and numpy are used here - no cloud vision calls
happen in this module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def _to_numpy(image) -> np.ndarray:
    """Normalize a PIL.Image or numpy array to a numpy uint8 array (H, W[, C])."""
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)


def tight_crop(
    image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 6,
) -> np.ndarray:
    """
    Crop tightly around the item of interest.

    If `bbox` (x0, y0, x1, y1) is given, crop to that box (plus padding).
    Otherwise, auto-detect content bounds by finding the bounding box of
    non-background pixels -- useful for a scanned page that is mostly
    blank/white margin around a small printed/handwritten region.
    """
    arr = _to_numpy(image)
    h, w = arr.shape[:2]

    if bbox is not None:
        x0, y0, x1, y1 = bbox
    else:
        if arr.ndim == 2:
            gray = arr
        else:
            channels = arr.shape[2]
            rgb = arr[:, :, :3] if channels == 4 else arr
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Anything noticeably darker than near-white is "content".
        mask = gray < 245
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            # Nothing detected -- return the original image untouched.
            return arr
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return arr[y0:y1, x0:x1]


def super_resolve(image, scale: float = 2.0) -> np.ndarray:
    """
    Upscale an image by `scale` (2x-4x recommended) using high-quality cubic
    interpolation. This is a lightweight, deterministic stand-in for a
    learned super-resolution model: cheap and dependency-free, and good
    enough to give the OCR/VLM stage more pixels-per-character to work with
    on small crops (e.g. the courtesy-amount box on a check).
    """
    if not (1.0 <= scale <= 4.0):
        raise ValueError("scale should be between 1x and 4x for this demo pipeline")
    arr = _to_numpy(image)
    h, w = arr.shape[:2]
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(arr, new_size, interpolation=cv2.INTER_CUBIC)


def to_grayscale(image) -> np.ndarray:
    """Convert an RGB(A) or already-gray image to single-channel grayscale."""
    arr = _to_numpy(image)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def autocontrast(image, cutoff: float = 0.5) -> np.ndarray:
    """
    Stretch the intensity histogram so the darkest/lightest `cutoff` percent
    of pixels are clipped and the remaining range is stretched to [0, 255].
    Thin wrapper around Pillow's ImageOps.autocontrast so behaviour matches a
    well-known, well-tested reference implementation.
    """
    arr = _to_numpy(image)
    pil_img = _to_pil(arr)
    stretched = ImageOps.autocontrast(pil_img, cutoff=cutoff)
    return np.array(stretched)


def otsu_threshold(gray: np.ndarray) -> int:
    """
    Hand-implemented Otsu's method: choose the threshold `t` in [0, 255]
    that maximizes the between-class variance of the pixel population split
    into {<= t} and {> t} (equivalently, minimizes intra-class variance).

    This is a from-scratch implementation (no cv2.threshold(..., THRESH_OTSU)
    call) so the "determinism where it counts" principle extends to being
    able to explain *exactly* how the binarization threshold is chosen.
    """
    if gray.ndim != 2:
        raise ValueError("otsu_threshold expects a single-channel grayscale image")

    histogram, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total_pixels = int(gray.size)
    if total_pixels == 0:
        return 128

    sum_total = float(np.dot(np.arange(256), histogram))

    sum_background = 0.0
    weight_background = 0
    best_threshold = 0
    best_between_class_variance = -1.0

    for t in range(256):
        weight_background += int(histogram[t])
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = (
            weight_background * weight_foreground *
            (mean_background - mean_foreground) ** 2
        )

        if between_class_variance > best_between_class_variance:
            best_between_class_variance = between_class_variance
            best_threshold = t

    return best_threshold


def otsu_binarize(image) -> np.ndarray:
    """Apply the hand-rolled Otsu threshold above, returning a 0/255 binary image."""
    gray = to_grayscale(image)
    t = otsu_threshold(gray)
    binary = np.where(gray > t, 255, 0).astype(np.uint8)
    return binary


def preprocess_pipeline(
    image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    scale: float = 2.0,
    binarize: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper chaining the full preprocessing pipeline used ahead
    of Stage-1 OCR:

        crop -> upscale -> grayscale -> autocontrast -> (optional) Otsu binarize

    Stage-2 VLM extraction in ledgerlens.pipeline deliberately calls this
    with binarize=False: vision-language models benefit from the extra
    grayscale/contrast detail (faint ink, signature texture) that a hard
    black/white threshold would throw away, whereas Stage-1 OCR benefits
    from the cleaner binarized image.
    """
    step = tight_crop(image, bbox=bbox)
    step = super_resolve(step, scale=scale)
    step = to_grayscale(step)
    step = autocontrast(step)
    if binarize:
        step = otsu_binarize(step)
    return step
