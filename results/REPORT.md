# Satellite change detection experiment

Official OSCD: 10 training cities, 3 validation cities, 10 held-out test cities. Four native 10m Sentinel-2 bands. Small CPU models, 15 epochs, width 8, 12 random 64×64 patches per training city per seed.

| Model | Micro F1 | Micro IoU | Macro city F1 |
|---|---:|---:|---:|
| difference | 0.279 ± 0.000 | 0.162 | 0.260 |
| unet | 0.139 ± 0.011 | 0.075 | 0.121 |
| siamese | 0.271 ± 0.094 | 0.160 | 0.252 |

Lowest F1 cities for Siamese seed 0: valencia (0.011), norcia (0.048), saclay_w (0.052).

The threshold and epoch are chosen on validation cities only. Every pixel in the common native image/label extent of each test city is evaluated. Unsupported outer borders exclude 11,183 labeled pixels; per-city extents are retained in the manifest. Random crops never cross split boundaries. Model seed variability and per-city scores expose geographic generalization limits.

These are small independent baseline implementations, not reproductions of published benchmark training. Nonoverlapping inference tiles can create seams; four-band inputs omit nine available channels. Limited patches and epochs constrain the results.

Image-derived artifacts in this results directory are subject to the OSCD dataset terms, separately from the MIT code license. Attribution: Daudt, Le Saux, Boulch and Gousseau, Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks, IGARSS 2018.
