"""Training-time augmentation for 320 px grayscale X-rays (numpy + OpenCV, never flips).

Order: photometric -> CLAHE -> low resolution -> blur / sharpen -> noise -> rotate / zoom / shift -> random black bars,
so added bars stay pure black like the padding written by preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugConfig:
    pad_p: float = 0.5                          # random black side bars (external TB sets are 97-100% padded)
    pad_fill: tuple[float, float] = (0.80, 0.97)  # share of the side the content keeps (Montgomery ~0.82)
    affine_p: float = 0.8
    rotate_deg: float = 7.0
    scale: tuple[float, float] = (0.9, 1.1)
    shift: float = 0.04                         # share of width / height
    photometric_p: float = 0.8                  # TBX11K classes differ in brightness / contrast
    brightness: float = 0.12                    # +- share of full range
    contrast: tuple[float, float] = (0.75, 1.3)
    gamma: tuple[float, float] = (0.75, 1.35)
    blur_p: float = 0.25                        # sharpness gap: TBX11K 99 vs Shenzhen 299 vs Montgomery 515
    blur_sigma: tuple[float, float] = (0.4, 1.5)
    sharpen_p: float = 0.25
    sharpen_amount: tuple[float, float] = (0.3, 1.5)
    noise_p: float = 0.2
    noise_std: float = 4.0
    clahe_p: float = 0.0                        # local contrast like other hospitals' processing (off by default)
    clahe_clip: tuple[float, float] = (1.0, 3.0)
    lowres_p: float = 0.0                       # downscale then upscale, like older or lower-resolution scanners
    lowres_scale: tuple[float, float] = (0.4, 0.8)


def photometric(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) * rng.uniform(*cfg.contrast) + 0.5 + rng.uniform(-cfg.brightness, cfg.brightness)
    x = np.clip(x, 0.0, 1.0) ** rng.uniform(*cfg.gamma)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def random_clahe(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=rng.uniform(*cfg.clahe_clip), tileGridSize=(8, 8)).apply(img)


def random_lowres(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    h, w = img.shape
    s = rng.uniform(*cfg.lowres_scale)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur_or_sharpen(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    u = rng.random()
    if u < cfg.blur_p:
        return cv2.GaussianBlur(img, (0, 0), rng.uniform(*cfg.blur_sigma))
    if u < cfg.blur_p + cfg.sharpen_p:
        amount = rng.uniform(*cfg.sharpen_amount)
        soft = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
        return np.clip(img.astype(np.float32) * (1 + amount) - soft * amount, 0, 255).astype(np.uint8)
    return img


def add_noise(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0.0, cfg.noise_std, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def affine(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-cfg.rotate_deg, cfg.rotate_deg), rng.uniform(*cfg.scale))
    m[:, 2] += rng.uniform(-cfg.shift, cfg.shift, 2) * (w, h)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def random_pad(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """Squeeze the image along one axis and add black bars, like a non-square X-ray padded to square."""
    h, w = img.shape
    fill = rng.uniform(*cfg.pad_fill)
    out = np.zeros_like(img)
    if rng.random() < 0.5:  # bars left / right
        nw = max(1, int(round(w * fill)))
        x0 = int(rng.integers(0, w - nw + 1))
        out[:, x0:x0 + nw] = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA)
    else:  # bars top / bottom
        nh = max(1, int(round(h * fill)))
        y0 = int(rng.integers(0, h - nh + 1))
        out[y0:y0 + nh, :] = cv2.resize(img, (w, nh), interpolation=cv2.INTER_AREA)
    return out


def augment(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """uint8 HxW in, uint8 HxW out (same size)."""
    if rng.random() < cfg.photometric_p:
        img = photometric(img, rng, cfg)
    if rng.random() < cfg.clahe_p:
        img = random_clahe(img, rng, cfg)
    if rng.random() < cfg.lowres_p:
        img = random_lowres(img, rng, cfg)
    img = blur_or_sharpen(img, rng, cfg)
    if rng.random() < cfg.noise_p:
        img = add_noise(img, rng, cfg)
    if rng.random() < cfg.affine_p:
        img = affine(img, rng, cfg)
    if rng.random() < cfg.pad_p:
        img = random_pad(img, rng, cfg)
    return img