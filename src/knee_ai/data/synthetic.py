from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, RLELossless

from knee_ai.constants import PLANES, SERIES_ID, STUDY_ID, TARGETS
from knee_ai.utils.reproducibility import write_json


def write_dicom(path: Path, image: np.ndarray, study_uid: str, series_uid: str,
                instance: int, plane: str = "Sagittal", rle: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sop_uid = f"{series_uid}.{instance}"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.9999"
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, sop_uid
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.PatientName, ds.PatientID = "SYNTHETIC^FIXTURE", "SYNTHETIC"
    ds.Modality = "MR"
    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.InstanceNumber = instance
    orientation = {"Sagittal": [0, 1, 0, 0, 0, 1], "Coronal": [1, 0, 0, 0, 0, 1], "Axial": [1, 0, 0, 0, 1, 0]}[plane]
    ds.ImageOrientationPatient = orientation
    normal = np.cross(orientation[:3], orientation[3:])
    ds.ImagePositionPatient = (normal * instance * 3).tolist()
    ds.PixelSpacing, ds.SliceThickness = [.7, .7], 3
    ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    ds.PixelData = np.ascontiguousarray(image, dtype="<u2").tobytes()
    if rle:
        ds.compress(RLELossless)
    ds.save_as(path, enforce_file_format=True)


def create_synthetic(root: str | Path, train_studies: int = 32, labelled_count: int = 24,
                     test_studies: int = 4, slices: int = 10, seed: int = 123) -> dict:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Preserve existing data at {root}; choose a new synthetic directory")
    if not 4 <= labelled_count <= train_studies or test_studies < 1 or slices < 1:
        raise ValueError("Invalid synthetic fixture dimensions")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, train_series, test_series, submission = [], [], [], []
    yy, xx = np.mgrid[-1:1:64j, -1:1:48j]
    for i in range(train_studies + test_studies):
        uid = f"1.2.826.0.1.3680043.10.9999.{i + 1}"
        labels = rng.integers(0, 2, size=12)
        training = i < train_studies
        if training:
            row = {STUDY_ID: uid, "Report": f"Synthetic fixture report {i}; not clinical text."}
            row.update({target: int(value) if i < labelled_count else np.nan for target, value in zip(TARGETS, labels)})
            rows.append(row)
        else:
            submission.append({STUDY_ID: uid, **dict.fromkeys(TARGETS, .5)})
        for plane_id, plane in enumerate(PLANES):
            sid = f"{uid}.{plane_id + 1}"
            metadata = {STUDY_ID: uid, SERIES_ID: sid, "Fluid_Sensitive": 1,
                        "Fat_Suppression": 1, "Anatomical_Plane": plane}
            (train_series if training else test_series).append(metadata)
            for z in range(slices):
                radius = .6 + .1 * np.cos(z / slices * np.pi)
                tissue = (xx**2 + (yy / 1.1)**2) < radius**2
                image = 250 * np.exp(-((xx + .25)**2 + yy**2) / .1)
                image += 700 * np.exp(-((xx - .2)**2 + (yy - .2)**2) / .06)
                # Nonclinical structured signal allows mechanical optimization tests.
                for j in range(4):
                    label = labels[plane_id * 4 + j]
                    center = -.45 + j * .3
                    image += label * 180 * np.exp(-((xx - center)**2 + (yy + .3)**2) / .008)
                image += rng.normal(60, 12, image.shape)
                image = np.clip(image * tissue, 0, 4095).astype(np.uint16)
                # Filename order is deliberately reversed; geometry controls ordering.
                path = root / ("train_images" if training else "test_images") / uid / sid / f"slice_{slices - z:03d}.dcm"
                write_dicom(path, image, uid, sid, z + 1, plane, rle=z == 0)
    pd.DataFrame(rows, columns=[STUDY_ID, "Report", *TARGETS]).to_csv(root / "train.csv", index=False)
    pd.DataFrame(train_series).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame(test_series).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame(submission, columns=[STUDY_ID, *TARGETS]).to_csv(root / "sample_submission.csv", index=False)
    result = {"synthetic": True, "seed": seed, "train_studies": train_studies, "labelled_count": labelled_count,
              "test_studies": test_studies, "slices_per_series": slices,
              "dicom_files": (train_studies + test_studies) * 3 * slices,
              "transfer_syntaxes": [str(ExplicitVRLittleEndian), str(RLELossless)],
              "warning": "MRI-like engineering fixtures; no medical performance can be inferred."}
    write_json(root / "SYNTHETIC.json", result)
    return result
