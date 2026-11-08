# Product Sales

A notebook that predicts how much a customer spends on a purchase from the Black Friday sales dataset (550,068 transactions).

## Data

`train.csv` has one row per purchase: user and product IDs, gender, age band, occupation, city category, years in the current city, marital status, three product category columns and `Purchase`, the amount spent, which is the target. `Product_Category_2` and `Product_Category_3` have many missing values.

## What the notebook does

1. Loads `train.csv` and checks for missing values.
2. Drops `Product_ID` and fills the missing product category values with the column mean.
3. Plots average `Purchase` by gender, age, occupation, city category, years in city and marital status.
4. Encodes the text columns as integers.
5. Fits a `LinearRegression` and a `RandomForestRegressor` (5 trees) on a train/test split.

The only score the notebook records is the random forest R2 on the test split, 0.618. It is a quick baseline: no tuning, and `User_ID` is kept as a numeric feature.

The first cell credits Abinesh, Denistan and Hariram.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook "Product Sales.ipynb"
```

The notebook reads `train.csv` from its own folder, so start Jupyter from the repo root.
