import numpy as np
import pydicom
import pytest
import torch

from knee_ai.constants import SERIES_ID, STUDY_ID
from knee_ai.data.dataset import StudyDataset
from knee_ai.data.metadata import load_series, load_train
from knee_ai.preprocessing.dicom import canonicalize, decode_slice, index_series, letterbox, normalize_images, sample_indices


def first_series(fixtures):
    row = load_series(fixtures / "train_series.csv").iloc[0]
    return row, fixtures / "train_images" / row[STUDY_ID] / row[SERIES_ID]


def test_geometry_ignores_filename_order(fixtures):
    row, directory = first_series(fixtures)
    headers, source = index_series(directory, row[SERIES_ID])
    assert source == "geometry"
    assert [header.instance for header in headers] == [1, 2, 3, 4, 5]
    assert headers[0].path.name > headers[-1].path.name


def test_native_rle_and_uncompressed_decode(fixtures):
    row, directory = first_series(fixtures)
    headers, _ = index_series(directory, row[SERIES_ID])
    assert pydicom.dcmread(headers[0].path).file_meta.TransferSyntaxUID.is_compressed
    for header in (headers[0], headers[1]):
        pixels = decode_slice(header, row.Anatomical_Plane)
        assert pixels.shape == (64, 48)
        assert np.isfinite(pixels).all() and pixels.max() > 0


def test_instance_fallback_and_ambiguous_order(fixtures, tmp_path):
    _, directory = first_series(fixtures)
    for path in directory.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        del ds.ImagePositionPatient
        ds.save_as(tmp_path / path.name, enforce_file_format=True)
    headers, source = index_series(tmp_path)
    assert source == "instance"
    path = headers[0].path
    ds = pydicom.dcmread(path)
    ds.InstanceNumber = 2
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="unambiguous"):
        index_series(tmp_path)


def test_rejects_multiframe(fixtures, tmp_path):
    _, directory = first_series(fixtures)
    ds = pydicom.dcmread(next(directory.glob("*.dcm")))
    ds.NumberOfFrames = 2
    ds.save_as(tmp_path / "enhanced.dcm", enforce_file_format=True)
    with pytest.raises(ValueError, match="multi-frame"):
        index_series(tmp_path)


def test_corrupted_headers_and_missing_series(tmp_path):
    (tmp_path / "broken.dcm").write_bytes(b"not dicom")
    with pytest.raises(FileNotFoundError):
        index_series(tmp_path)


def test_sampling_physical_spacing():
    positions = [0, 1, 2, 10, 11]
    uniform = sample_indices(5, 2, "uniform", positions)
    central = sample_indices(5, 2, "central", positions)
    assert uniform.tolist() == [2, 3]
    assert central.tolist() == [2, 3]
    assert sample_indices(20, 4, "uniform").tolist() == [2, 7, 12, 17]
    assert sample_indices(20, 4, "central").tolist() == [5, 8, 11, 14]
    assert sample_indices(1, 4).tolist() == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        sample_indices(0, 4)


def test_normalization_letterbox_orientation():
    image = np.arange(24, dtype=np.float32).reshape(4, 6)
    normalized = normalize_images([image])[0]
    assert normalized.min() == 0 and normalized.max() == 1
    assert letterbox(normalized, 16).shape == (1, 16, 16)
    assert np.count_nonzero(normalize_images([np.ones((3, 4))])[0]) == 0
    flipped = canonicalize(image, (-1, 0, 0, 0, -1, 0), "Axial")
    np.testing.assert_array_equal(flipped, image[::-1, ::-1])


def test_fixed_reference_normalization_shared_across_conditions():
    reference = [np.linspace(1, 100, 100, dtype=np.float32).reshape(10, 10)]
    low = np.full((4, 4), 30, dtype=np.float32)
    high = np.full((4, 4), 70, dtype=np.float32)
    all_images = normalize_images([low, high], reference)
    np.testing.assert_array_equal(normalize_images([low], reference)[0], all_images[0])
    np.testing.assert_array_equal(normalize_images([high], reference)[0], all_images[1])


def test_modality_rescale_and_monochrome1(fixtures, tmp_path):
    row, directory = first_series(fixtures)
    ds = pydicom.dcmread(next(directory.glob("*.dcm")))
    ds.decompress() if ds.file_meta.TransferSyntaxUID.is_compressed else None
    original = ds.pixel_array.astype(np.float32)
    ds.RescaleSlope, ds.RescaleIntercept = 2, -10
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.save_as(tmp_path / "slice.dcm", enforce_file_format=True)
    headers, _ = index_series(tmp_path)
    result = decode_slice(headers[0], row.Anatomical_Plane)
    scaled = original * 2 - 10
    np.testing.assert_array_equal(result, scaled.max() + scaled.min() - scaled)


def test_dataset_shapes_short_volumes_masks_and_cache(fixtures, tmp_path):
    train, series = load_train(fixtures / "train.csv"), load_series(fixtures / "train_series.csv")
    data = StudyDataset(train, series, fixtures / "train_images", slices_per_series=8, image_size=32, cache_dir=tmp_path)
    a, b = data[0], data[0]
    assert a["images"].shape == (24, 1, 32, 32)
    assert int(a["mask"].sum()) == 15
    assert a["labels"].shape == (12,)
    assert torch.equal(a["images"], b["images"])
    assert len(list(tmp_path.glob("*.pt"))) == 1
    central = StudyDataset(train, series, fixtures / "train_images", 8, 32, "central", cache_dir=tmp_path)
    central[0]
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_missing_plane_masked_but_empty_study_fails(fixtures):
    train, series = load_train(fixtures / "train.csv"), load_series(fixtures / "train_series.csv")
    only = series.loc[series.Anatomical_Plane.eq("Sagittal")]
    data = StudyDataset(train, only, fixtures / "train_images", 2, 32)
    assert int(data[0]["mask"].sum()) == 2
    with pytest.raises(FileNotFoundError):
        StudyDataset(train, series, fixtures / "missing", 2, 32)[0]
    with pytest.raises(RuntimeError, match="No usable"):
        StudyDataset(train, series, fixtures / "missing", 2, 32, error_policy="skip_series")[0]
