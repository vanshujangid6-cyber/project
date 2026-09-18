# Materials Science & Machine Learning Projects

Two portfolio projects combining materials engineering, physically informed synthetic data, scientific computing, and interpretable machine learning.

## Projects

### 1. ML-Assisted Optimization of Alumina-Magnesia Spinel Refractories
Focus: refractory chemistry, composition-property modeling, phase stability.

- Physically informed MgO-Al2O3-SiO2 dataset generation.
- Random Forest regression for spinel expansion and slag-corrosion behavior.
- 5-fold cross-validation, held-out testing, and feature importance.
- MgO-Al2O3-MgAl2O4 phase-stability analysis with pymatgen.
- Honest documentation that the training data are synthetic and designed to prototype the workflow before real laboratory/plant data are available.

Project README: topic1_spinel_refractory/README.md

### 2. Data-Driven Modeling of Leaching Kinetics & Porosity in Fused Silica Ceramic Cores
Focus: ceramic processing, alkaline leaching, kinetics, and interpretable ML.

- Models porosity, modulus of rupture, and KOH dissolution rate.
- Fits a shrinking-core kinetic model with scipy.optimize.curve_fit.
- Cross-checks kinetics with scipy.integrate.solve_ivp.
- Uses Random Forest regression and held-out permutation importance.
- Uses synthetic-but-physically-grounded data with documented provenance.

Project README: topic2_ceramic_core/README.md

## Tech Stack

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, joblib, pymatgen.

## Reproducibility

The datasets are synthetic and physically informed because proprietary experimental datasets are not publicly available. The source scripts document the assumptions and can be adapted to real experimental data.

Install dependencies with: pip install -r requirements.txt

Run each project's data-generation script first, then the analysis/modeling scripts described in its README.
