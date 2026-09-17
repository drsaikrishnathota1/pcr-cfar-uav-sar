# Table 5

Robustness on 20 held-out scenes at SCR = 8 dB. Training used two-look K texture with shape 1.6.

| Study | Setting | CA Pd / Pfa | CNN Pd / Pfa | PCR Pd / Pfa |
|---|---|---:|---:|---:|
| Looks | L=1 | 0.483 / 1.17e-2 | 0.975 / 1.99e-3 | 1.000 / 1.75e-3 |
| Looks | L=2 | 0.475 / 5.55e-3 | 1.000 / 1.57e-3 | 1.000 / 6.80e-4 |
| Looks | L=4 | 0.558 / 2.45e-3 | 1.000 / 1.18e-3 | 1.000 / 3.30e-4 |
| Texture | nu=0.8 | 0.608 / 1.25e-2 | 1.000 / 1.95e-3 | 1.000 / 1.91e-3 |
| Texture | nu=3.2 | 0.575 / 2.10e-3 | 1.000 / 1.10e-3 | 1.000 / 2.83e-4 |
| Crowding | 12 targets | 0.500 / 5.50e-3 | 1.000 / 1.56e-3 | 1.000 / 6.75e-4 |
| GO-2L | alpha L=2 | 1.000 / 3.51e-2 | — | — |
| OS-2L | alpha L=2 | 1.000 / 2.33e-2 | — | — |
| Box F1 | centroid r=3 | 0.022 | 0.153 | 0.297 |
| Public TerraSAR-X | 8 chips, injected 8 dB | 0.938 / 5.10e-6 | 1.000 / 6.12e-5 | 1.000 / 5.87e-5 |
| CA window | 13x13 | 0.425 / 5.40e-3 | — | — |
| CA window | 21x21 | 0.475 / 5.55e-3 | — | — |
| CA window | 31x31 | 0.500 / 5.65e-3 | — | — |
