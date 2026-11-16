"""Generate synthetic compressed DICOMs, then decode in a separate NumPy-1 environment.

The caller supplies an isolated interpreter with the four pinned codec packages
and NumPy 1.26.4, but no pyjpegls or pylibjpeg-rle. No downloads occur here.
The parent uses its existing fixture encoders; its environment is never changed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CHILD = r'''
import importlib.metadata as metadata
import importlib.util
import json, platform, socket, sys
from pathlib import Path
import numpy as np
import pydicom
from pydicom.pixels import get_decoder, pixel_array
from pydicom.uid import JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, RLELossless

def blocked(*args, **kwargs):
    raise RuntimeError("Networking forbidden in codec verification")
socket.create_connection = blocked
socket.socket.connect = blocked
assert np.__version__ == "1.26.4"
assert importlib.util.find_spec("jpeg_ls") is None, "pyjpegls must be absent"
assert importlib.util.find_spec("rle") is None, "external RLE plugin must be absent"
folder = Path(sys.argv[1])
truth = np.load(folder / "truth.npy")
results = []
for name, uid, plugin, tolerance in (
    ("jpeg_ls_lossless", JPEGLSLossless, "pylibjpeg", 0),
    ("jpeg_ls_near_lossless", JPEGLSNearLossless, "pylibjpeg", 1),
    ("jpeg2000_lossless", JPEG2000Lossless, "pylibjpeg", 0),
    ("rle_lossless", RLELossless, "pydicom", 0),
):
    assert plugin in get_decoder(uid).available_plugins
    actual = pixel_array(folder / (name + ".dcm"), decoding_plugin=plugin)
    error = int(np.abs(actual.astype(np.int32) - truth.astype(np.int32)).max())
    assert error <= tolerance, (name, error)
    results.append({"fixture": name, "transfer_syntax": str(uid), "plugin": plugin,
                    "shape": list(actual.shape), "max_absolute_pixel_error": error, "passed": True})
report = {"synthetic": True, "genuine_competition_images": False, "networking": "blocked",
          "python": sys.version.split()[0], "platform": platform.platform(),
          "versions": {p: metadata.version(p) for p in ("numpy", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg")},
          "pyjpegls_installed": False, "pylibjpeg_rle_installed": False, "fixtures": results}
(folder / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, help="Isolated NumPy 1.26.4 interpreter")
    parser.add_argument("--output", default="outputs/numpy126_codec_fixtures")
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid,
                             JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, RLELossless)
    y, x = np.indices((64, 64))
    truth = ((y * 17 + x * 29) % 4096).astype(np.uint16)
    np.save(folder / "truth.npy", truth)
    for name, uid in (("jpeg_ls_lossless", JPEGLSLossless), ("jpeg_ls_near_lossless", JPEGLSNearLossless),
                      ("jpeg2000_lossless", JPEG2000Lossless), ("rle_lossless", RLELossless)):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.Rows, dataset.Columns = truth.shape
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16; dataset.BitsStored = 12; dataset.HighBit = 11
        dataset.PixelRepresentation = 0
        dataset.PixelData = truth.tobytes()
        options = {"jls_error": 1} if uid == JPEGLSNearLossless else {}
        dataset.compress(uid, **options)
        dataset.save_as(folder / (name + ".dcm"), enforce_file_format=True)
    subprocess.run([str(Path(args.python).resolve()), "-I", "-c", CHILD, str(folder)], check=True)


if __name__ == "__main__":
    main()
