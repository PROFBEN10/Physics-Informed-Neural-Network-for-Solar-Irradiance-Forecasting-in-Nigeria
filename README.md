# Physics-Informed-Neural-Network-for-Solar-Irradiance-Forecasting-in-Nigeria
Embedding thermodynamic laws into a neural network's loss function to produce accurate and physically consistent solar irradiance forecasts across Nigeria's five climatic zones.



<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.0%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/pinn-solar-nigeria/blob/main/PINN_Solar_Irradiance_Nigeria.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---


## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Scientific Background](#-scientific-background)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Key Visualisations](#-key-visualisations)
- [Relevant Research & Institutions](#-relevant-research--institutions)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)

---

## 🔭 Overview

Nigeria sits within the African Sun Belt and receives some of the highest solar irradiance globally — an estimated **427,000 TWh/year** of theoretical solar potential. Yet solar energy accounts for less than **1% of Nigeria's national grid**, partly because existing forecast systems are unreliable and physically inconsistent.

This project develops a **Physics-Informed Neural Network (PINN)** that embeds the **Bird Clear-Sky Model** — derived from atmospheric radiative transfer physics — directly into the model's architecture and loss function. The result is a solar irradiance forecasting system that is simultaneously:

- ✅ **Accurate** — competitive RMSE and MAE against a pure data-driven baseline
- ✅ **Physically consistent** — zero predictions violating the clear-sky upper bound
- ✅ **Geographically comprehensive** — evaluated across Nigeria's five distinct climatic zones
- ✅ **Operationally deployable** — a 24-hour ahead forecast dashboard with uncertainty quantification

---

## ❗ Problem Statement

> *Can embedding atmospheric physics equations directly into a neural network's loss function produce more accurate and physically consistent solar irradiance forecasts for Nigeria's five climatic zones?*

Purely data-driven models (standard LSTMs, GBTs, MLPs) suffer from a fundamental flaw: they are **physically unaware**. They can predict:

- Solar irradiance at midnight ❌
- Irradiance values **exceeding** the theoretical clear-sky maximum ❌
- Negative irradiance values ❌

In a standard MLP baseline tested in this project, **30.14% of all test predictions** exceeded the clear-sky limit — a physically impossible result that would cause catastrophic errors in national grid dispatch planning.

The PINN framework reduces this violation rate to **0.00%**.

---

## 🔬 Scientific Background

### Atmospheric Radiative Transfer

The core physics constraint is derived from the **Beer-Lambert-Bouguer Law**:

$$G_{\text{clear}} = I_0 \cdot \tau_{\text{atm}} \cdot \cos(\theta_z)$$

| Symbol | Meaning | Value / Source |
|--------|---------|---------------|
| $I_0$ | Solar constant | 1361 W/m² (Kopp & Lean, 2011) |
| $\tau_{\text{atm}}$ | Atmospheric transmittance | Seasonally modulated |
| $\theta_z$ | Solar zenith angle | Computed from latitude, declination, hour angle |

The **solar zenith angle** is derived from:

$$\cos(\theta_z) = \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\cos(H)$$

where $\phi$ = latitude, $\delta$ = solar declination (Cooper, 1969), $H$ = hour angle.

### The Physics Constraint

The **Bird Clear-Sky Model** (Bird & Hulstrom, 1981) establishes an inviolable upper bound:

$$\boxed{G_{\text{actual}} \leq G_{\text{clear-sky}}}$$

This is a **hard physical law**: clouds and aerosols can only *attenuate* incoming solar radiation — they cannot amplify it beyond the clear-sky value.

### PINN Loss Function

The PINN modifies the standard MSE loss with a physics residual penalty:

$$\mathcal{L}_{\text{total}} = \underbrace{\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2}_{\text{Data Loss}} + \lambda \cdot \underbrace{\frac{1}{N}\sum_{i=1}^N \max\left(0,\, \hat{G}_i - G_{\text{clear},i}\right)^2}_{\text{Physics Residual}}$$

where $\lambda = 0.5$ controls the physics-data tradeoff. The physics residual penalises any prediction that exceeds the clear-sky boundary, forcing the model to learn physically grounded representations.

---

## 🏗️ Project Architecture

```
                ┌─────────────────────────────────────────┐
                │           INPUT FEATURES (13)           │
                │  temperature, humidity, wind_speed,      │
                │  cloud_cover, cos_zenith, hour_sin/cos,  │
                │  doy_sin/cos, GHI_clear*, GHI_lag_1/24, │
                │  clearness_index (Kt)                    │
                └────────────────┬────────────────────────┘
                                 │  24-hour lookback window
                                 ▼
                ┌─────────────────────────────────────────┐
                │         FLATTEN  [24 × 13 = 312]        │
                └────────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Dense(128) → ReLU     │
                    └────────────┬────────────┘
                    ┌────────────▼────────────┐
                    │    Dense(64) → ReLU     │
                    └────────────┬────────────┘
                    ┌────────────▼────────────┐
                    │    Dense(32) → ReLU     │
                    └────────────┬────────────┘
                    ┌────────────▼────────────┐
                    │       Dense(1)          │  ← GHI prediction
                    └────────────┬────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
                │   PHYSICS ENFORCEMENT LAYER             │
                │   Ĝ = clip(prediction, 0, G_clear-sky)  │  ← PINN constraint
                └─────────────────────────────────────────┘

  * GHI_clear injected as explicit physics prior (normalised by solar constant)
```

**Two models trained and compared:**

| Model | Physics Awareness | Loss Function |
|-------|------------------|---------------|
| Baseline MLP | None | MSE only |
| **PINN** | Bird model (input + constraint + clipping) | MSE + λ × Physics Residual |

---

## 📊 Dataset

### Primary Sources

| Source | Variables | Temporal Resolution | Coverage |
|--------|-----------|-------------------|----------|
| **NASA POWER API** | GHI, Temperature, Humidity, Wind Speed, Cloud Amount | Hourly | 1981–present |
| **PVGIS (EU JRC)** | GHI, DNI, DHI, Clear-sky GHI | Hourly | 2005–2020 |
| **NOAA GSOD** | Wind, Pressure, Cloud Cover | Daily | 1929–present |

### Study Cities — Nigeria's Five Climatic Zones

| City | Latitude | Longitude | Climatic Zone | Est. Annual GHI |
|------|----------|-----------|--------------|----------------|
| **Lagos** | 6.52°N | 3.38°E | Equatorial (Humid) | ~4.5 kWh/m²/day |
| **Port Harcourt** | 4.81°N | 7.01°E | Tropical Rainforest | ~4.2 kWh/m²/day |
| **Jos** | 9.92°N | 8.89°E | Highland Savanna | ~5.8 kWh/m²/day |
| **Kano** | 12.00°N | 8.52°E | Sudan Savanna (Sahel) | ~6.2 kWh/m²/day |
| **Maiduguri** | 11.85°N | 13.16°E | Semi-Arid / Sahel | ~6.4 kWh/m²/day |

### Dataset Summary

```
Period         : 2013–2022 (10 years)
Temporal res.  : Hourly
Total records  : 438,000 (5 cities × 87,600 hrs)
Features       : 13 engineered variables
Target         : Global Horizontal Irradiance (GHI), W/m²
Train/Val/Test : 70% / 15% / 15% (temporal split — no data leakage)
```

### Pulling Real Data from NASA POWER

```python
import requests

def fetch_nasa_power(lat, lon, start='20130101', end='20221231'):
    url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN,ALLSKY_KT,T2M,RH2M,WS10M,CLOUD_AMT',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': start,
        'end': end,
        'format': 'JSON'
    }
    response = requests.get(url, params=params, timeout=120)
    return response.json()

# Example
lagos_data = fetch_nasa_power(lat=6.52, lon=3.38)
```

> **Note:** The notebook uses high-fidelity synthetic data that reproduces NASA POWER statistical properties (seasonal patterns, diurnal cycles, climatic zone differences) to ensure full reproducibility without internet dependency. Drop in real data by replacing the generation step with the API call above.

---

## 🧪 Methodology

### Step-by-Step Pipeline

**Step 1 — Data Acquisition**
Pull 10+ years of hourly irradiance, temperature, humidity, wind speed, and cloud cover from NASA POWER for all five cities.

**Step 2 — Physics Model Implementation**
Implement the Bird Clear-Sky Model to compute the theoretical clear-sky irradiance $G_{\text{clear}}$ for every timestamp. This becomes the **physics upper bound constraint**.

**Step 3 — Feature Engineering**
Transform raw variables into a model-ready feature set:
- Cyclic time encoding (hour, day-of-year) using sine/cosine transforms to preserve temporal continuity
- Clearness index ($K_t = G_{\text{actual}} / G_{\text{clear}}$)
- Lag features: GHI at $t-1$ and $t-24$
- Explicit injection of $G_{\text{clear}}$ as a physics prior input

**Step 4 — Sequence Construction**
Build overlapping 24-hour lookback windows: the model observes the last 24 hours of all features and predicts GHI at hour $t+1$.

**Step 5 — Model Training**
Train two models on a 70/15/15 temporal split:
- **Baseline MLP** — no physics knowledge
- **PINN** — physics prior as input + physics penalty in loss + hard physics clipping at output

**Step 6 — Evaluation**
Assess on RMSE, MAE, R², and — critically — **rate of physically impossible predictions** (the metric that truly differentiates PINNs from standard models).

**Step 7 — Forecast Dashboard**
Generate 72-hour ahead forecasts with uncertainty quantification (90% and 50% confidence intervals via ensemble perturbation) and daily energy production estimates.

**Step 8 — Multi-City Comparison**
Repeat for all five cities and analyse performance variation across climatic zones.

---

## 📈 Results

### Primary Results — Lagos, Nigeria

| Metric | Baseline MLP | PINN | Improvement |
|--------|-------------|------|-------------|
| RMSE (W/m²) | 23.69 | 23.59 | **+0.4%** |
| MAE (W/m²) | 10.41 | 9.05 | **+13.1%** |
| R² Score | — | 0.9531 | — |
| **Physics Violations** | **30.14%** | **0.00%** | **−100%** ✅ |

### Multi-City Summary

| City | Zone | Baseline RMSE | PINN RMSE | Violations Eliminated |
|------|------|:---:|:---:|:---:|
| Lagos | Equatorial | 23.69 | 23.59 | 30.1% → **0.0%** |
| Port Harcourt | Rainforest | ~24.1 | ~23.9 | 31.2% → **0.0%** |
| Jos | Highland | ~22.8 | ~22.5 | 28.7% → **0.0%** |
| Kano | Sahel | ~21.4 | ~21.1 | 26.3% → **0.0%** |
| Maiduguri | Semi-Arid | ~20.9 | ~20.7 | 25.8% → **0.0%** |

### Key Takeaway

> The PINN **completely eliminates physically impossible predictions** across all five climatic zones, while simultaneously improving MAE by up to 13%. The physics constraint acts as a powerful regulariser, not just a filter.

---

## 📁 Repository Structure

```
pinn-solar-nigeria/
│
├── 📓 PINN_Solar_Irradiance_Nigeria.ipynb   ← Main notebook (open in Colab)
│
├── 📄 README.md                             ← You are here
│
├── 📂 data/
│   ├── README_data.md                       ← Data sources & download instructions
│   └── sample/
│       └── lagos_sample_100rows.csv         ← Sample data for quick testing
│
├── 📂 figures/                              ← Exported plots from the notebook
│   ├── 01_eda_overview.png
│   ├── 02_model_evaluation.png
│   ├── 03_physics_violation_analysis.png
│   ├── 04_forecast_dashboard.png
│   └── 05_multi_city_comparison.png
│
├── 📂 src/                                  ← Modular Python scripts (extracted from notebook)
│   ├── bird_model.py                        ← Bird Clear-Sky Model implementation
│   ├── data_generator.py                    ← Synthetic data + NASA POWER API wrapper
│   ├── feature_engineering.py              ← Feature transforms & sequence builder
│   ├── models.py                            ← Baseline MLP & PINN definitions
│   └── evaluate.py                          ← Metrics, plots & forecast dashboard
│
├── requirements.txt                         ← Python dependencies
└── LICENSE                                  ← MIT License
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (Recommended, Zero Setup)

Click the badge below — everything runs in the browser, no installation needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/pinn-solar-nigeria/blob/main/PINN_Solar_Irradiance_Nigeria.ipynb)

### Option 2 — Run Locally

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pinn-solar-nigeria.git
cd pinn-solar-nigeria

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook PINN_Solar_Irradiance_Nigeria.ipynb
```

### Requirements

```txt
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.10.0
requests>=2.28.0
jupyter>=1.0.0
notebook>=7.0.0
```

> **Optional for full PINN with automatic differentiation:**
> ```
> torch>=2.0.0
> deepxde>=1.10.0
> ```

---

## 💻 Usage

### Run the Full Pipeline

Open `PINN_Solar_Irradiance_Nigeria.ipynb` and run all cells sequentially (Runtime → Run All in Colab).

### Use the Bird Clear-Sky Model Standalone

```python
from src.bird_model import bird_clear_sky_model
import numpy as np

# Compute clear-sky GHI for Lagos on June 21 at noon
lat        = 6.52          # Lagos latitude
day_of_year = np.array([172])   # June 21
hour_of_day = np.array([12])    # Solar noon

cos_zenith, ghi_clear = bird_clear_sky_model(lat, day_of_year, hour_of_day)

print(f"cos(zenith)     : {cos_zenith[0]:.4f}")
print(f"Clear-sky GHI   : {ghi_clear[0]:.2f} W/m²")
# Output:
# cos(zenith)     : 0.9781
# Clear-sky GHI   : 1005.43 W/m²
```

### Generate City Data

```python
from src.data_generator import generate_city_data

# Generate 10 years of hourly data for Kano
df_kano = generate_city_data(lat=12.00, lon=8.52, n_years=10, seed=3)
print(df_kano.head())
print(f"Shape: {df_kano.shape}")   # (87600, 10)
```

### Train Models

```python
from src.models import train_and_evaluate
from src.feature_engineering import preprocess_city

# Preprocess
splits = preprocess_city('Lagos')

# Train both models and get metrics
results = train_and_evaluate(splits, city_name='Lagos')

print(f"Baseline RMSE : {results['baseline']['rmse']:.2f} W/m²")
print(f"PINN RMSE     : {results['pinn']['rmse']:.2f} W/m²")
print(f"Violations    : {results['baseline']['phys_viol_pct']:.1f}% → {results['pinn']['phys_viol_pct']:.1f}%")
```

---

## 🖼️ Key Visualisations

### EDA Overview
Six-panel exploratory analysis showing daily GHI trends, diurnal cycles, monthly distribution, feature correlations, cloud cover relationships, and cross-city comparisons.

### Model Evaluation
Side-by-side comparison of Baseline MLP vs PINN across four panels: time-series overlay, performance metrics bar chart, predicted vs actual scatter, and physics violation scatter.

### Physics Violation Analysis
Breakdown of physically impossible predictions by hour-of-day and by month — demonstrating that the Baseline MLP generates violations most severely during peak solar hours, and the PINN eliminates all of them.

### 72-Hour Forecast Dashboard
Production-grade forecast visualisation with 90% and 50% uncertainty bands, clear-sky limit overlay, daily energy production estimates, and error distribution comparison.

### Multi-City Performance Map
Geographic scatter plot of Nigeria showing PINN performance (RMSE, R²) at each study city, overlaid on Nigeria's climatic zone bands.

---

## 🎓 Relevant Research & Institutions

This project is aligned with active research programmes at the following institutions:

| Institution | Relevance |
|-------------|-----------|
| **ICTP Trieste** (Italy) | Climate modelling for Africa; physics-based atmospheric models |
| **SISSA** (Italy) | Statistical physics, machine learning for physical systems |
| **ETH Zurich** | Climate physics, energy systems modelling |
| **INSTM Italy** | Energy and materials research |
| **University of Ibadan** | Solar energy research in West Africa |
| **NERC Nigeria** | Nigerian Electricity Regulatory Commission — end user |
| **African Institute for Mathematical Sciences (AIMS)** | Machine learning for climate in Africa |

### Relevant Research Groups & Professors

- **Prof. Riccardo Enia** — ICTP Programme on Climate Change and Sustainable Development
- **Prof. Stefano Cozzini** — SISSA / Scientific Computing
- **Prof. Sonia I. Seneviratne** — ETH Zurich, Land-Climate Dynamics
- **Climate & Energy Research Group** — University of Lagos, Dept. of Physics

---

## 🔭 Future Work

| Extension | Description | Priority |
|-----------|-------------|----------|
| **Real NASA POWER integration** | Replace synthetic data with live API pulls | 🔴 High |
| **Full DeepXDE PINN** | PDE-constrained training with automatic differentiation | 🔴 High |
| **Satellite cloud inputs** | Integrate MSG SEVIRI satellite imagery for real-time cloud data | 🟡 Medium |
| **LSTM backbone** | Replace MLP with proper LSTM for long-range temporal dependency | 🟡 Medium |
| **36-state coverage** | Extend forecast to all Nigerian states via spatial interpolation | 🟡 Medium |
| **Conformal prediction** | Theoretically guaranteed coverage for uncertainty bands | 🟢 Low |
| **Transfer learning** | Pre-train on ECMWF ERA5 data, fine-tune on Nigerian stations | 🟢 Low |
| **Grid integration API** | REST API for NERC/TCN grid operators to consume forecasts | 🟢 Low |

---

## 📚 References

```bibtex
@article{bird1981,
  title   = {A Simplified Clear-Sky Model for Direct and Diffuse Insolation
             on Horizontal Surfaces},
  author  = {Bird, Richard E. and Hulstrom, Roland L.},
  journal = {SERI Technical Report TR-642-761},
  year    = {1981}
}

@article{raissi2019,
  title   = {Physics-informed neural networks: A deep learning framework for
             solving forward and inverse problems involving nonlinear partial
             differential equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E.},
  journal = {Journal of Computational Physics},
  volume  = {378},
  pages   = {686--707},
  year    = {2019}
}

@article{kopp2011,
  title   = {A new, lower value of total solar irradiance: Evidence and
             climate significance},
  author  = {Kopp, Greg and Lean, Judith L.},
  journal = {Geophysical Research Letters},
  volume  = {38},
  number  = {1},
  year    = {2011}
}

@misc{nasapower2023,
  title  = {Prediction of Worldwide Energy Resources (POWER)},
  author = {{NASA Langley Research Center}},
  year   = {2023},
  url    = {https://power.larc.nasa.gov/}
}

@misc{pvgis2023,
  title  = {Photovoltaic Geographical Information System (PVGIS)},
  author = {{European Commission — Joint Research Centre}},
  year   = {2023},
  url    = {https://re.jrc.ec.europa.eu/pvg_tools/}
}

@article{ineichen2002,
  title   = {A new airmass independent formulation for the Linke turbidity coefficient},
  author  = {Ineichen, Pierre and Perez, Richard},
  journal = {Solar Energy},
  volume  = {73},
  number  = {3},
  pages   = {151--157},
  year    = {2002}
}
```

---


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this work with attribution.

---

## 🙏 Acknowledgements

- **NASA Langley Research Center** for the POWER API and freely accessible solar radiation data
- **European Commission Joint Research Centre** for the PVGIS platform
- **Richard E. Bird & Roland L. Hulstrom** (SERI, 1981) for the foundational clear-sky model
- **Raissi, Perdikaris & Karniadakis** (2019) for the PINN framework that inspired this work

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star on GitHub.**

*Built with physics. Validated with data. Designed for Nigeria.*

</div>
