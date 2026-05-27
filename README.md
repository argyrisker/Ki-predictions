# Ki Predictions from Molecular Structure

![Python](https://img.shields.io/badge/Python-3.11-blue)
![RDKit](https://img.shields.io/badge/RDKit-cheminformatics-informational)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-informational)
![Status](https://img.shields.io/badge/status-reproducible%20portfolio%20project-success)

A reproducible Applied Pharmaceutical Bioinformatics project for modelling molecular binding affinity from SMILES strings. The target is `Ki (nM)`, transformed to `log10(Ki)` for regression and thresholded for activity classification.

The repository is structured as a code-first PhD-application portfolio project: clean source modules, reproducible environment files, a runnable smoke check, and a notebook workflow. It avoids inflated claims: random cross-validation is useful internal validation, not proof of prospective drug-discovery performance.

## Portfolio summary

| Item | Detail |
|---|---|
| Research task | QSAR-style prediction of Ki from molecular structure |
| Dataset | 7,562 SMILES/Ki records supplied for an exam project |
| Curation | RDKit parsing, largest-fragment selection, canonical SMILES, median aggregation of repeats |
| Final modelling set | 4,820 unique canonical molecules |
| Descriptors | ECFP4, MACCS keys, RDKit descriptors, custom SMARTS-derived structural features |
| Best regression model | ExtraTrees regressor, R² = 0.722 ± 0.055 on `log10(Ki)` |
| Main limitation | Random CV does not prove scaffold-level generalisation |

## Project structure

```text
.
├── data/                         # example data; full exam dataset can be added locally
├── notebooks/                    # notebook workflow location
├── src/apb_ki_modelling/         # reusable cleaning and descriptor code
├── scripts/                      # smoke check and notebook execution helpers
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Methods

1. Load raw SMILES and Ki values.
2. Remove missing and non-positive Ki values.
3. Parse molecules with RDKit.
4. For salts and multi-fragment records, keep the largest covalent fragment.
5. Convert molecules to canonical isomeric SMILES.
6. Aggregate repeated canonical molecules using median Ki.
7. Transform Ki to `log10(Ki)` for regression.
8. Generate ECFP4, MACCS, RDKit, and custom descriptors.
9. Compare regression and classification models using 10-fold cross-validation.
10. Run descriptor ablation to test where predictive signal comes from.

## Key results

### Cleaning

| Step | Result |
|---|---:|
| Raw records | 7,562 |
| Valid Ki records after cleaning | 7,556 |
| Invalid RDKit SMILES | 0 |
| Multi-fragment SMILES | 552 |
| Unique canonical molecules after aggregation | 4,820 |

### Regression on `log10(Ki)`

| Model | R² | MSE |
|---|---:|---:|
| ExtraTrees regressor | 0.722 ± 0.055 | 0.476 ± 0.102 |
| HistGradientBoosting regressor | 0.694 ± 0.065 | 0.524 ± 0.124 |
| Random Forest regressor | 0.669 ± 0.055 | 0.569 ± 0.106 |
| Ridge regression | 0.463 ± 0.106 | 0.909 ± 0.116 |

### Classification

| Threshold | Best model by AUC | AUC | F1 |
|---|---|---:|---:|
| Ki < 1000 nM | HistGradientBoosting / ExtraTrees | 0.942 ± 0.013 / 0.942 ± 0.018 | 0.966 ± 0.005 / 0.965 ± 0.006 |
| Ki < median Ki | ExtraTrees | 0.891 ± 0.017 | 0.802 ± 0.025 |

The `Ki < 1000 nM` task is chemically interpretable but imbalanced: about 90% of canonical molecules are active. The median split is less chemically meaningful but gives a more honest classifier stress test.

### Descriptor ablation

| Descriptor set | Features | R² mean | MSE mean |
|---|---:|---:|---:|
| ECFP4 + MACCS + RDKit + custom | 1236 | 0.721753 | 0.475985 |
| ECFP4 + RDKit core | 1037 | 0.707132 | 0.502200 |
| ECFP4 only | 1024 | 0.696819 | 0.520654 |
| RDKit core only | 13 | 0.504880 | 0.851078 |

ECFP4 carries most of the signal. The extended descriptor set helps, but the gain is incremental, not transformative.

## Reproducibility

Use Conda or Mamba. RDKit is the dependency most likely to break in a casual pip-only setup.

```bash
mamba env create -f environment.yml
mamba activate apb-ki-modelling
make smoke
```

Run the notebook after adding it to `notebooks/Ki_modelling.ipynb`:

```bash
make notebook
```

## Critical limitation

Random 10-fold cross-validation after duplicate aggregation is acceptable for an exam project, but it can still leak analogue-series information across folds. A research-grade next step would use Bemis-Murcko scaffold splitting, external validation, uncertainty estimates, and error analysis by chemical series.

## Application framing

A strong way to describe this project in PhD applications:

> Built a reproducible RDKit/scikit-learn workflow for modelling molecular Ki values from SMILES, including molecular standardisation, duplicate aggregation, multi-family descriptor generation, cross-validated regression/classification, and descriptor ablation. The project emphasises methodological transparency and explicitly discusses scaffold-leakage risk in random validation.

## Data note

The dataset was supplied for an exam project. The code is reusable; dataset reuse outside the course context may require checking the original data rights.
