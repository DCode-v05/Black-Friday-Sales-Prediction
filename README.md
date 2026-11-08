# Black Friday Sales Prediction

**Regression on a 550K-row Black Friday dataset: predicting how much a customer spends from their demographics and the product they bought.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)

## Overview

This is a supervised-learning notebook that predicts the purchase amount of a retail transaction from customer and product attributes. The data is the well-known Black Friday sales set: 550,068 transactions with shopper demographics (gender, age band, occupation, city, marital status) and product category fields, plus the `Purchase` value to predict.

It's a regression problem, since the target is a continuous amount rather than a class. The notebook goes from loading the data to checking what's missing, plotting how spend varies across customer segments, cleaning and encoding everything into numbers, and then fitting two models (a linear baseline and a Random Forest) on a held-out test split.

This was a team task (Abinesh, Denistan, Hariram) done as part of a data-science learning track. It's an EDA-to-model exercise, not a production service.

## Key Features

- Loads and inspects the 550,068-row, 12-column dataset with `head()`, `info()`, and per-column null counts.
- Finds missing data (`Product_Category_2` has 173,638 nulls and `Product_Category_3` is missing for most rows) and prints value-count distributions for the category fields.
- Six exploratory bar charts of average `Purchase` by **Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, and Marital_Status**, to show which segments spend more.
- Preprocessing: drops the high-cardinality `Product_ID`, mean-imputes the numeric product-category columns, and converts every categorical column to integers with manual label encoding.
- Two models trained on an 80/20 split with a fixed seed: a **Linear Regression** baseline and a **Random Forest Regressor**.
- The Random Forest is scored with R² on the test set.

## How It Works

The whole project lives in one notebook, `Product Sales.ipynb`, run on Python 3.12 with the standard pandas / scikit-learn stack. The flow is data, inspection, EDA, preprocessing, modelling, evaluation.

### Data and inspection

`train.csv` (~25 MB) is read into a pandas DataFrame. The 12 columns are `User_ID`, `Product_ID`, `Gender`, `Age`, `Occupation`, `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`, `Product_Category_1/2/3`, and the target `Purchase`. A null check shows that `Product_Category_2` (173,638 nulls) and `Product_Category_3` (most rows) are sparse. `value_counts()` on `Product_Category_2` shows a long-tailed category distribution.

### Exploratory analysis

Working on a copy of the frame, the notebook uses seaborn bar plots (`whitegrid` style) to chart mean `Purchase` against each categorical feature one at a time: gender, age band, occupation code, city category, years in the current city, and marital status. These plots are the "who spends more" part of the analysis.

### Preprocessing

Three cleaning steps:

1. **Drop `Product_ID`**: it's a high-cardinality identifier that doesn't generalise.
2. **Mean-impute the category columns**: for each of `Product_Category_1/2/3`, missing values are filled with the column mean (only applied to non-object dtypes), which clears all remaining nulls.
3. **Manual label encoding**: instead of `LabelEncoder`, the notebook loops over every object-dtype column, takes its unique values, and replaces each with an incrementing integer (0, 1, 2, ...). After this, `Gender`, `Age`, `City_Category`, and `Stay_In_Current_City_Years` are all numeric.

`User_ID` is left in as a numeric feature.

### Modelling and evaluation

Features `x` are everything except `Purchase`; `y` is `Purchase`. The data is split 80/20 with `train_test_split` and `np.random.seed(50)`.

- **Linear Regression** is fit first as a baseline and used to generate predictions on the test set.
- **Random Forest Regressor** (`n_estimators=5`) is fit next and scored with its built-in `.score()` (R²) on the test split.

## Results / Highlights

- **Dataset scale:** 550,068 transactions × 12 columns.
- **Random Forest R²:** ~0.618 on the held-out 20% test split (`n_estimators=5`). This is the only score the notebook records; the linear baseline is not scored.
- The bar-chart EDA shows higher- and lower-spend customer segments across all six categorical features.

The Random Forest uses only 5 trees and there is no tuning, so 0.618 is a starting point rather than a tuned result. More trees and feature work would likely push it higher.

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
git clone https://github.com/DCode-v05/Black-Friday-Sales-Prediction.git
cd Black-Friday-Sales-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running

```bash
jupyter notebook "Product Sales.ipynb"
```

## Usage

Open the notebook and run the cells top to bottom. It reads `train.csv` from the same folder, so start Jupyter from the repo root. Each stage is self-contained:

- The early cells are for inspection. Re-run them to see null counts and value distributions.
- The seaborn cells redraw the EDA charts; swap the `x=` column to plot spend against a different feature.
- The modelling cells at the end fit both models. To experiment, bump `n_estimators` on the `RandomForestRegressor`, add more metrics (MAE/RMSE), or try a different encoding and re-run `.score()` to compare.

## Project Structure

```
Black-Friday-Sales-Prediction/
├── Product Sales.ipynb   # The whole project: EDA, preprocessing, and both models
├── train.csv             # Black Friday dataset, 550,068 rows, 12 columns (~25 MB)
├── .gitignore
└── README.md             # This file
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
