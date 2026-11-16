from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
PREPROCESS_VERSION = "geometry-fixed-reference-percentile-letterbox-v2"


@dataclass
class SliceHeader:
    path: Path
    position: float | None
    instance: int
    orientation: tuple[float, ...] | None


def index_series(directory: Path, expected_series: str | None = None) -> tuple[list[SliceHeader], str]:
    headers = []
    reference_normal = None
    for path in sorted(directory.glob("*")):
        if not path.is_file():
            continue
        # Support extensionless DICOM, without trying sidecar files.
        if path.suffix.lower() not in ("", ".dcm", ".dicom"):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if expected_series and str(ds.get("SeriesInstanceUID", "")) != expected_series:
                raise ValueError("SeriesInstanceUID does not match metadata")
            if int(ds.get("NumberOfFrames", 1)) != 1:
                raise ValueError("Enhanced/multi-frame MRI requires a frame-level adapter")
            orientation = tuple(float(v) for v in ds.ImageOrientationPatient) if "ImageOrientationPatient" in ds else None
            position = None
            if orientation and "ImagePositionPatient" in ds:
                normal = np.cross(orientation[:3], orientation[3:])
                norm = np.linalg.norm(normal)
                if norm > .99:
                    normal /= norm
                    if reference_normal is None:
                        reference_normal = normal
                    if abs(np.dot(normal, reference_normal)) < .99:
                        raise ValueError("Inconsistent slice normals within one series")
                    position = float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), reference_normal))
            headers.append(SliceHeader(path, position, int(ds.get("InstanceNumber", 0)), orientation))
        except Exception as exc:
            # Do not silently turn unsupported frames or metadata mismatches into empty images.
            if isinstance(exc, ValueError):
                raise ValueError(f"Invalid series at {directory.name}: {exc}") from exc
            LOG.warning("Unreadable DICOM header %s (%s)", path.name, type(exc).__name__)
    if not headers:
        raise FileNotFoundError(f"No readable single-frame DICOMs in {directory}")
    if all(h.position is not None for h in headers):
        return sorted(headers, key=lambda h: (h.position, h.instance, h.path.name)), "geometry"
    if len({h.instance for h in headers}) == len(headers):
        LOG.warning("Geometry incomplete for series %s; using InstanceNumber", directory.name)
        return sorted(headers, key=lambda h: (h.instance, h.path.name)), "instance"
    raise ValueError(f"Series {directory.name} lacks unambiguous geometry or InstanceNumber ordering")


def sample_indices(n: int, count: int, strategy: str = "uniform", positions: list[float] | None = None,
                   central_fraction: float = .6) -> np.ndarray:
    if n < 1 or count < 1:
        raise ValueError("Slice count and available slices must be positive")
    if strategy not in ("uniform", "central"):
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if not 0 < central_fraction <= 1:
        raise ValueError("central_fraction must be in (0, 1]")
    margin = (1 - central_fraction) / 2 if strategy == "central" else 0
    # Mid-bin sampling avoids disproportionately selecting the volume endpoints.
    fractions = margin + (np.arange(count) + .5) / count * (1 - 2 * margin)
    axis = np.asarray(positions, dtype=float) if positions is not None else np.arange(n, dtype=float)
    if len(axis) != n or not np.isfinite(axis).all() or np.any(np.diff(axis) < 0):
        raise ValueError("Positions must be finite, sorted and match series length")
    targets = axis[0] + fractions * (axis[-1] - axis[0])
    return np.abs(axis[:, None] - targets[None, :]).argmin(axis=0)


def canonicalize(image: np.ndarray, orientation: tuple[float, ...] | None, plane: str) -> np.ndarray:
    if orientation is None:
        return image
    desired = {"Sagittal": (1, 2), "Coronal": (0, 2), "Axial": (0, 1)}[plane]
    horizontal = np.array(orientation[:3])
    vertical = np.array(orientation[3:])
    axes = (int(np.abs(horizontal).argmax()), int(np.abs(vertical).argmax()))
    if axes == desired[::-1]:
        image = image.T
        horizontal, vertical = vertical, horizontal
        axes = axes[::-1]
    if axes != desired:
        LOG.warning("Oblique orientation cannot be axis-canonicalized for %s; preserving pixel grid", plane)
        return image
    if horizontal[desired[0]] < 0:
        image = image[:, ::-1]
    if vertical[desired[1]] < 0:
        image = image[::-1, :]
    return image


def decode_slice(header: SliceHeader, plane: str) -> np.ndarray:
    try:
        ds = pydicom.dcmread(header.path)
        image = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Cannot decode {header.path.name}; check transfer syntax and offline pydicom codecs") from exc
    if image.ndim != 2 or int(ds.get("SamplesPerPixel", 1)) != 1:
        raise ValueError("Only single-frame monochrome MRI is supported")
    image = image * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        image = image.max() + image.min() - image
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite DICOM pixels after modality rescale")
    return np.ascontiguousarray(canonicalize(image, header.orientation, plane))


def normalize_images(images: list[np.ndarray], reference_images: list[np.ndarray] | None = None) -> list[np.ndarray]:
    values = np.concatenate([image.ravel() for image in (reference_images if reference_images is not None else images)])
    nonzero = values[values != 0]
    lo, hi = np.percentile(nonzero if len(nonzero) else values, [1, 99])
    if hi - lo < 1e-6:
        return [np.zeros_like(image, dtype=np.float32) for image in images]
    return [np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32) for image in images]


def letterbox(image: np.ndarray, size: int) -> torch.Tensor:
    height, width = image.shape
    scale = size / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)[0]
    top, left = (size - new_h) // 2, (size - new_w) // 2
    return F.pad(tensor, (left, size - new_w - left, top, size - new_h - top))


def load_selected_series(directory: Path, series_uid: str, plane: str, count: int, size: int,
                         strategy: str, central_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    headers, ordering = index_series(directory, series_uid)
    positions = [h.position for h in headers] if ordering == "geometry" else None
    indices = sample_indices(len(headers), count, strategy, positions, central_fraction)
    # Decode each selected file only once. Mask duplicate padding in short volumes.
    unique = list(dict.fromkeys(indices.tolist()))
    # Fixed uniform calibration slices give both experimental conditions the
    # same intensity scale, even when they select different model input slices.
    references = list(dict.fromkeys(sample_indices(len(headers), 3, "uniform", positions).tolist()))
    decoded = {index: decode_slice(headers[index], plane) for index in dict.fromkeys([*unique, *references])}
    images = normalize_images([decoded[index] for index in unique], [decoded[index] for index in references])
    tensors = {index: letterbox(image, size) for index, image in zip(unique, images)}
    seen = set()
    mask = []
    for index in indices:
        mask.append(index not in seen)
        seen.add(index)
    return torch.stack([tensors[index] for index in indices]), torch.tensor(mask, dtype=torch.bool)
