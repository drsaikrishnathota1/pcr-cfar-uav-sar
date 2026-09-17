# Table 2

Ablation at the same 8 dB operating point.

| Variant | \(P_d\) | Empirical \(P_{fa}\) |
|---|---:|---:|
| CA-CFAR (exponential prior) | 0.500 | \(5.52\times 10^{-3}\) |
| CA-CFAR, two-look gamma scale | 1.000 | \(4.35\times 10^{-2}\) |
| CA-CFAR, looks-adaptive L | 0.721 | \(6.62\times 10^{-3}\) |
| Ungated residual (gate ≡ 1) | 1.000 | \(6.70\times 10^{-4}\) |
| Black-box CNN-CFAR (no CA prior) | 1.000 | \(1.57\times 10^{-3}\) |
| PCR-CFAR (gated residual + Pfa loss) | 1.000 | \(6.62\times 10^{-4}\) |
