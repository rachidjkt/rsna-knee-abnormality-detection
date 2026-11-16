import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pydicom
import pytest
import torch
from pydicom.uid import ImplicitVRLittleEndian, JPEG2000Lossless, JPEGLSLossless

from knee_ai.constants import SERIES_ID, STUDY_ID
from knee_ai.data.metadata import load_series
from knee_ai.inference.predict import load_model
from knee_ai.models.multislice import MultiSliceClassifier
from knee_ai.preprocessing.dicom import decode_slice, index_series
from knee_ai.utils.kaggle import find_competition_root, find_image_root, find_checkpoint


@pytest.mark.parametrize("syntax", [JPEG2000Lossless, JPEGLSLossless])
def test_synthetic_lossless_codec_roundtrip(fixtures, tmp_path, syntax):
    row = load_series(fixtures / "train_series.csv").iloc[0]
    directory = fixtures / "train_images" / row[STUDY_ID] / row[SERIES_ID]
    source = pydicom.dcmread(next(directory.glob("*.dcm")))
    if source.file_meta.TransferSyntaxUID.is_compressed:
        source.decompress()
    truth = source.pixel_array.copy()
    source.compress(syntax)
    source.save_as(tmp_path / "compressed.dcm", enforce_file_format=True)
    headers, _ = index_series(tmp_path)
    pixels = decode_slice(headers[0], row.Anatomical_Plane)
    np.testing.assert_array_equal(pixels, truth)


def test_implicit_vr_decode(fixtures, tmp_path):
    row = load_series(fixtures / "train_series.csv").iloc[0]
    source = pydicom.dcmread(next((fixtures / "train_images" / row[STUDY_ID] / row[SERIES_ID]).glob("*.dcm")))
    if source.file_meta.TransferSyntaxUID.is_compressed:
        source.decompress()
    truth = source.pixel_array.copy()
    source.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    source.save_as(tmp_path / "implicit.dcm", enforce_file_format=True)
    headers, _ = index_series(tmp_path)
    np.testing.assert_array_equal(decode_slice(headers[0], row.Anatomical_Plane), truth)


def test_official_lossless_jpeg_decoder_available():
    from pydicom.pixels import get_decoder
    from pydicom.uid import JPEGLossless, JPEGLosslessSV1
    assert "pylibjpeg" in get_decoder(JPEGLossless).available_plugins
    assert "pylibjpeg" in get_decoder(JPEGLosslessSV1).available_plugins


def test_notebooks_embed_current_source_without_runtime_downloads():
    sources = {p.relative_to(Path("src")).as_posix(): p.read_text(encoding="utf-8")
               for p in sorted(Path("src/knee_ai").rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata.knee_ai_source_sha256 == digest
        assert notebook.metadata.kaggle.isInternetEnabled is False
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path}:{index}", "exec")
        embedded = notebook.cells[2].source
        payload = re.search(r"b64decode\('([^']+)'\)", embedded).group(1)
        assert json.loads(gzip.decompress(base64.b64decode(payload))) == sources


def test_mount_discovery_handles_official_folder_layout(tmp_path, fixtures):
    root = tmp_path / "competitions/rsna-knee-abnormality-detection"
    root.mkdir(parents=True)
    for name in ("train.csv", "train_series.csv", "test_series.csv", "sample_submission.csv"):
        (root / name).write_bytes((fixtures / name).read_bytes())
    (root / "train_series").mkdir()
    (root / "test_series").mkdir()
    assert find_competition_root(tmp_path) == root
    assert find_image_root(root, "train") == root / "train_series"
    assert find_image_root(root, "test") == root / "test_series"
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tmp_path)


def test_pretrained_weights_load_without_download_and_freeze():
    weights = Path("artifacts/pretrained/resnet18-f37072fd.pth")
    if not weights.exists():
        pytest.skip("Optional public ImageNet weights not prepared; run fetch_encoder_weights.py")
    model = MultiSliceClassifier("resnet18", encoder_weights=weights, freeze_encoder=True)
    model.train()
    assert not model.encoder.training
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    output = model(torch.rand(2, 2, 1, 48, 48), torch.ones(2, 2, dtype=torch.bool))
    assert output.shape == (2, 12)
