# NIFTY Options IV Surface Reconstruction

**Competition:** Finance Club, IIT Roorkee — Open Project  
**Best Score:** `0.0000368505` MSE  
**Metric:** Mean Squared Error on held-out IV values

---

## Problem Statement

Given a partially observed Implied Volatility (IV) dataset for NIFTY 50 options (Jan 7–27, 2026), reconstruct the complete IV surface by predicting missing values across strikes and expiries. The dataset spans **975 timestamps** at 5-minute intervals (09:15–15:25) across 28 CE and PE strike columns for a single expiry (Jan 27, 2026).

---

## Key EDA Findings

- IV smile is **smooth and monotone in log-moneyness** space — well suited for polynomial fitting
- Always **≥ 6 observed strikes per timestamp** → polynomial fitting always well-conditioned
- **Expiry day (Jan 27)** shows sharp end-of-day IV spikes in OTM puts and instability in OTM calls as time-to-expiry approaches zero
- Term structure confirms IV rises as DTE shrinks (expiry effect)
- Feature correlations with IV:

| Feature | Correlation with IV | Role in model |
|---------|---------------------|---------------|
| `iv_neighbor_mean` | **0.996** | Implicitly (cross-sectional uses same-row neighbors) |
| `iv_lag_1` | 0.988 | Via temporal PCHIP |
| `iv_roll_mean_5/10` | 0.963–0.974 | Via temporal PCHIP |
| `log_moneyness` | 0.042 | x-axis of polynomial |
| `days_to_expiry` | −0.421 | Separate expiry-day treatment |
| `session_progress` | 0.086 | Implicitly via temporal PCHIP |

---

## Modelling Approach

The solution is a **three-step pipeline** with zero look-ahead bias at every stage.

### Step 1 — Cross-Sectional Fit (per timestamp)

For each timestamp independently, using only the observed IV values at that row:

- **Interpolation** (target inside observed strike range): Gaussian-weighted degree-3 polynomial in log-moneyness `k = log(K/S)`, `σ_frac = 0.62`
- **Extrapolation** (target outside range): SVI model (80%) + 2-point linear safety net (20%); falls back to Gaussian poly (σ_frac = 1.2) + 50% linear if SVI fails

**Why log-moneyness?** Raw strikes ignore the moving spot price — the same raw strike has different financial meaning as NIFTY moves. Log-moneyness normalises K relative to S, giving a consistent smile shape across all days.

**Why Gaussian weights?** Feature analysis shows `iv_neighbor_mean` has correlation 0.996 with IV — local strikes are by far the strongest predictors. Gaussian weights `w_i = exp(-½·((k_i − k_target)/σ)²)` with `σ = 0.62 × half-range` formalise this locality.

**Why SVI for wing extrapolation?** The SVI model (Gatheral 2004):
```
w(k) = a + b·[ρ·(k−m) + √((k−m)² + σ²)]
```
is specifically designed for wing behaviour — linear in |k| at tails with correct asymptotic behaviour. A no-butterfly condition is enforced during fitting to avoid arbitrage.

### Step 2 — Intra-Day Temporal Smoothing (PCHIP)

For each strike column independently, a **PCHIP interpolator** is fit within each 75-row trading day window using only same-day observed values. PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is monotone-preserving — it never overshoots between observations, which is important for IV that cannot go negative.

### Step 3 — Adaptive Blend

Final predictions combine both signals:

| Day type | Cross-sectional weight | Temporal weight |
|---|---|---|
| Regular days | 95% | 5% |
| Expiry day (Jan 27) | 90% | 10% |

The higher temporal weight on expiry day captures the IV collapse toward close — a time-dimension effect the cross-sectional polynomial alone cannot model.

---

## No Look-Ahead Bias

| Step | Data used |
|------|-----------|
| Cross-sectional interpolation | Same timestamp only |
| SVI extrapolation | Same timestamp only |
| Temporal PCHIP | Same trading day (75 rows), prior observations only |
| Blend | — |

**Validation:** Leave-One-Out (LOO) — one observed cell removed, predicted from the remaining cells at the same timestamp. This exactly mirrors the competition test structure.

---

## Score Progression

| Approach | MSE |
|----------|-----|
| Baseline PCHIP + linear | 0.000085 |
| Unweighted poly3 in log-moneyness | 0.000045 |
| Gaussian-weighted poly3 + 5% temporal | 0.0000372 |
| SVI extrapolation + intra-day PCHIP | 0.0000369 |
| **Best submission** | **0.0000368** |

---

## Project Structure

```
PROJECT_OP_FINANCE/
├── .gitignore
├── readme.md
│
├── # Data
├── dataset.csv                                              # Original data with missing IV values
├── filled_dataset.csv                                       # Fully reconstructed IV surface (975 × 30)
├── submission.csv                                           # Competition submission (id || value format)
│
├── # Notebook
├── NIFTY_IV_Reconstruction.ipynb                           # Full pipeline: EDA → modelling → visualisation
│
└── # EDA & Result Visualisations
    ├── Missing IV % by Strike.png                          # Missingness bar chart by column
    ├── Missingness heatmap — 200 sampled timestamps × all strikes.png
    ├── NIFTY 50 Spot Price — Jan 7 to Jan 27, 2026.png    # Spot price with expiry marker
    ├── Pearson Correlation with IV.png                     # Feature correlation bar chart
    ├── Term Structure of IV — as DTE shrinks, IV rises (expiry effect).png
    ├── CE andPE smile.png                                  # Volatility smile at 11:30 each day
    ├── PE Smile.png
    ├── OTM CE IV — Expiry Day.png                         # Expiry day IV behaviour
    ├── OTM PE IV — Expiry Day.png
    ├── CE — Expiry Day Evolution.png                       # Smile evolution across expiry session
    └── PE — Expiry Day Evolution.png
```

---

## Libraries

`pandas` · `numpy` · `scipy` (PchipInterpolator, least_squares) · `matplotlib` · `seaborn`
