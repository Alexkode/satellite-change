# Geographic satellite change detection

[![Verify project](https://github.com/Alexkode/satellite-change/actions/workflows/verify.yml/badge.svg)](https://github.com/Alexkode/satellite-change/actions/workflows/verify.yml)

Independent difference-threshold, early-fusion UNet and shared-encoder Siamese UNet baselines on the original OSCD satellite dataset. Predict urban change from two registered dates using four native 10m bands (blue, green, red, near infrared).

## Run

```sh
python -m pip install -r requirements.txt
python scripts/download.py
python -m pytest -q
bash scripts/run_cpu.sh
```

Downloads total about 489 MiB from the author's mirrors. Archive MD5 values are checked against [TorchGeo's published dataset implementation](https://docs.torchgeo.org/en/v0.7.2/_modules/torchgeo/datasets/oscd.html); SHA-256 and source URLs are retained in the split manifest. The old train-label domain has moved to `partage.imt.fr`.

Saclay East is excluded because neighboring Saclay West is in the official test set. The thirteen remaining official training cities are sorted; their last three become validation, and ten train the models. The ten official test cities are held out. No random-pixel split. Validation chooses model epoch and decision threshold; test evaluates every pixel in the common native image/label extent by tiles. Unsupported outer borders are excluded and counted in the manifest. The four-band inputs were checked against the official rectified rasters and are identical inside these extents. Three measured seeds per neural architecture, a difference baseline, confusion counts, micro/macro metrics and city errors appear in [report](results/REPORT.md) and [summary](results/summary.json).

These small models are baselines, with 15 epochs and 12 random patches per city. Four bands omit the other spectral channels. This does not reproduce a paper's full training recipe or claim a state-of-the-art score. Discuss class imbalance, geographic shift, threshold selection, shared temporal encoding and tile seams in an interview.

## Attribution and license

Code was developed with AI assistance and is MIT licensed. The bundled original dataset README identifies the images as modified Copernicus Sentinel data from 2015–2018 and releases change labels under Creative Commons BY-NC-SA (without specifying a version). These terms are separate from the code license. They also apply to label-derived `results/example.png` and `results/*/example.npz`. Raw images and model weights are excluded. Retain attribution and consult the authors for commercial reuse of labels.

Dataset and paper: [Daudt, Le Saux, Boulch and Gousseau (IGARSS 2018)](https://rcdaudt.github.io/oscd/), [author's reference code](https://github.com/rcdaudt/fully_convolutional_change_detection).

## Installation and verification

Clone this repository and run commands from its root. Each repository has its own dependencies and tests.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/verify.sh
```

GitHub Actions checks the code contracts and the consistency of published experimental records. It does not rerun training or cluster benchmarks. See [experiment provenance](ENVIRONMENT.md) and [measured report](results/REPORT.md).

For Linux CPU-only PyTorch, install its CPU wheel before `requirements.txt`: `python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu`.
