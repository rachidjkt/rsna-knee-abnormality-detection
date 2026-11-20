# Attribution and license scope

## Included baseline source

The available Git history begins at `2a4facc`, importing the existing `knee_ai` baseline, scripts, tests, and source-embedded notebooks. Headers and that history do not identify an upstream copied baseline or include a third-party source notice. Earlier implementation history and any pre-import snippet provenance are not included. Confirm whether the imported `src/knee_ai/` and supporting scripts adapt external code and provide any applicable upstream notices before selecting a project-wide license. This review does not infer ownership from the Git author field.

The ResNet18 implementation is imported from torchvision in `src/knee_ai/models/multislice.py`, not copied into this repository. Notebooks embed the repository package for offline execution; they are not evidence of a separate model's authorship. Synthetic DICOM fixtures exercise software interfaces and contain no clinical performance evidence.

## Third-party resources

| Resource | Use and upstream source |
| --- | --- |
| PyTorch | Model/training runtime; [upstream](https://github.com/pytorch/pytorch) |
| torchvision / ResNet18 | Encoder implementation; [BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| ResNet18 ImageNet weights | External `IMAGENET1K_V1` checkpoint; [torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) |
| pydicom | DICOM reader and native codecs; [upstream](https://github.com/pydicom/pydicom) |
| pylibjpeg and plugins / pyjpegls | Optional codecs; [pylibjpeg](https://github.com/pydicom/pylibjpeg), [pyjpegls](https://github.com/pydicom/pyjpegls); plugins retain their respective terms |
| NumPy, pandas, scikit-learn, Pillow | Numerical, tabular, evaluation, and image dependencies; [NumPy](https://github.com/numpy/numpy), [pandas](https://github.com/pandas-dev/pandas), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [Pillow](https://github.com/python-pillow/Pillow) |
| RSNA competition inputs | Separately authorized data; [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) |

`pyproject.toml` and `requirements*.txt` record runtime and verification dependencies; their installed packages carry their own notices. No dependency-wide license replacement is intended.

`scripts/fetch_encoder_weights.py` identifies the official `resnet18-f37072fd.pth` URL, checks its hash prefix, and writes a SHA-256 provenance manifest when deliberately run. It explicitly leaves competition eligibility unestablished. The torchvision code license does not by itself establish all weight, ImageNet data, or competition-use permissions. No weights or competition inputs are published here.

## Excluded research and remaining decision

Later borrowed ensemble workflows and checkpoints described in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) are outside this repository. Their upstream notebook/checkpoint identities, licenses, and competition eligibility are not established by the included baseline history. Any future publication of those separate assets needs a source-specific review; their unresolved provenance is not assigned to torchvision's included API usage.

No project-wide license is added. The owner needs to confirm pre-import source/snippet provenance and choose a license for the contributions they have authority to license. Any identified upstream copyright notices must be retained under the applicable terms. This documentation review does not establish permission to redistribute excluded ensembles, data, or weights.
