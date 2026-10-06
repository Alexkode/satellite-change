# Geographic satellite change detection

Independent difference-threshold, early-fusion UNet and shared-encoder Siamese UNet baselines on the original OSCD satellite dataset. Predict urban change from two registered dates using four native 10m bands (blue, green, red, near infrared).

## Run

```sh
python -m pip install -r requirements.txt
python scripts/download.py
python -m pytest -q
bash scripts/run_cpu.sh
```

Downloads total about 489 MiB from the author's mirrors. Archive MD5 values are checked against [TorchGeo's published dataset implementation](https://docs.torchgeo.org/en/v0.7.2/_modules/torchgeo/datasets/oscd.html); SHA-256 and source URLs are retained in the split manifest. The old train-label domain has moved to `partage.imt.fr`.

Saclay East is excluded because neighboring Saclay West is in the official test set. The thirteen remaining official training cities are sorted; their last three become validation, and ten train the models. The ten official test cities are held out. No random-pixel split. Validation chooses model epoch and decision threshold; test evaluates all pixels by tiles. Three measured seeds per neural architecture, a simple difference baseline, confusion counts, micro and macro metrics, and per-city errors are published in [report](results/REPORT.md) and [summary](results/summary.json).

These small models are baselines, with 15 epochs and 12 random patches per city. Four bands omit the other spectral channels. This does not reproduce a paper's full training recipe or claim a state-of-the-art score. Discuss class imbalance, geographic shift, threshold selection, shared temporal encoding and tile seams in an interview.

## Attribution and license

Code was developed with AI assistance and is MIT licensed. OSCD images, labels, and derived `results/example.png` and `results/*/example.npz` are **separate third-party data**, under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) as identified by the dataset distributor. Raw images and model weights are excluded. Retain attribution when reusing visual results.

Dataset and paper: [Daudt, Le Saux, Boulch and Gousseau (IGARSS 2018)](https://rcdaudt.github.io/oscd/), [author's reference code](https://github.com/rcdaudt/fully_convolutional_change_detection).
