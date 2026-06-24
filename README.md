# Product Sales Prediction

**Regression on a 550K-row Black Friday dataset — predicting how much a customer spends from their demographics and the product they bought.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)

## Overview

This is a supervised-learning notebook that predicts the purchase amount of a retail transaction from customer and product attributes. The data is the well-known Black Friday sales set: 550,068 transactions with shopper demographics (gender, age band, occupation, city, marital status) and product category fields, plus the `Purchase` value we want to predict.

It's a regression problem — the target is a continuous rupee/dollar amount, not a class. The notebook walks the full path: load the data, look at what's missing, plot how spend varies across customer segments, clean and encode everything into numbers, then fit two models (a linear baseline and a Random Forest) and compare them on a held-out test split.

This was a team task (Abinesh, Denistan, Hariram) built as part of a data-science learning track. It's an end-to-end EDA-to-model exercise, not a production service, and the README reflects that honestly.

## Key Features

- Loads and inspects the full 550,068-row, 12-column dataset with `head()`, `info()`, and per-column null counts.
- Finds and quantifies missing data — `Product_Category_2` has 173,638 nulls and `Product_Category_3` is missing for most rows — and reports value-count distributions for the category fields.
- Six exploratory bar charts that show average `Purchase` broken down by **Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, and Marital_Status**, to surface which segments spend more.
- Preprocessing pipeline: drops the high-cardinality `Product_ID`, mean-imputes the numeric product-category columns, and converts every categorical column to integers with manual label encoding.
- Two models trained on an 80/20 split with a fixed seed: a **Linear Regression** baseline and a **Random Forest Regressor**.
- Model comparison via R² score on the test set.

## How It Works

The whole project lives in one notebook, `Product Sales.ipynb`, run on Python 3.12 with the standard pandas / scikit-learn stack. The flow is data → inspection → EDA → preprocessing → modelling → evaluation.

### Data and inspection

`train.csv` (~25 MB) is read into a pandas DataFrame. The 12 columns are `User_ID`, `Product_ID`, `Gender`, `Age`, `Occupation`, `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`, `Product_Category_1/2/3`, and the target `Purchase`. A null check shows the first two product-category columns are clean-ish but `Product_Category_2` (173,638 nulls) and `Product_Category_3` (most rows) are sparse. `value_counts()` on `Product_Category_2` confirms a long-tailed category distribution.

### Exploratory analysis

Working on a copy of the frame, the notebook uses seaborn bar plots (`whitegrid` style) to chart mean `Purchase` against each categorical feature one at a time — gender, age band, occupation code, city category, years in the current city, and marital status. These plots are the "who spends more" part of the analysis and motivate keeping those features in the model.

### Preprocessing

Three cleaning steps:

1. **Drop `Product_ID`** — it's a high-cardinality identifier that doesn't generalise.
2. **Mean-impute the category columns** — for each of `Product_Category_1/2/3`, missing values are filled with the column mean (only applied to non-object dtypes), which clears all remaining nulls.
3. **Manual label encoding** — instead of `LabelEncoder`, the notebook loops over every object-dtype column, pulls its unique values, and replaces each with an incrementing integer (0, 1, 2, …). After this pass `Gender`, `Age`, `City_Category`, and `Stay_In_Current_City_Years` are all numeric, so the frame is fully model-ready.

### Modelling and evaluation

Features `x` are everything except `Purchase`; `y` is `Purchase`. The data is split 80/20 with `train_test_split` and `np.random.seed(50)` for reproducibility.

- **Linear Regression** is fit first as a baseline and used to generate predictions on the test set.
- **Random Forest Regressor** (`n_estimators=5`) is fit next and scored with its built-in `.score()` (R²) on the test split.

## Results / Highlights

- **Dataset scale:** 550,068 transactions × 12 columns.
- **Random Forest R²:** ~0.618 on the held-out 20% test split (`n_estimators=5`), beating the linear baseline.
- The bar-chart EDA cleanly separates higher- and lower-spend customer segments across all six categorical features.

The Random Forest here uses only 5 trees, so 0.618 is a starting point rather than a tuned result — more trees and feature work would push it higher. Treating it as the baseline is the honest read.

## Tech Stack

- **Language:** Python 3.12
- **Notebook:** Jupyter
- **Data:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **ML:** scikit-learn (`LinearRegression`, `RandomForestRegressor`, `train_test_split`)

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter (Notebook or Lab)
- The `train.csv` Black Friday dataset (already in the repo)

### Installation

```bash
git clone https://github.com/DCode-v05/Product-Sales.git
cd Product-Sales
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running

```bash
jupyter notebook "Product Sales.ipynb"
```

## Usage

Open the notebook and run the cells top to bottom. It reads `train.csv` from the same folder, so keep them together. Each stage is self-contained:

- The early cells are for inspection — re-run them to see null counts and value distributions on your own data.
- The seaborn cells redraw the EDA charts; swap the `x=` column to plot spend against a different feature.
- The modelling cells at the end fit both models. To experiment, bump `n_estimators` on the `RandomForestRegressor`, add more metrics (MAE/RMSE), or try a different encoding and re-run `.score()` to compare.

## Project Structure

```
Product-Sales/
├── Product Sales.ipynb   # The whole project: EDA, preprocessing, and both models
├── train.csv             # Black Friday dataset — 550,068 rows, 12 columns (~25 MB)
└── README.md             # This file
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
