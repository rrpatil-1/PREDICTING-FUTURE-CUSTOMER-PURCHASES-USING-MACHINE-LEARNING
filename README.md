# Customer Segmentation and Future Purchase Prediction

This project uses the Online Retail dataset to:

- explore transaction and customer behavior,
- calculate Recency, Frequency, and Monetary Value (RFM) features,
- segment customers with K-means,
- predict whether a customer will make at least one purchase during the final three-month holdout window, and
- compare Logistic Regression, Random Forest, and XGBoost with and without segment names.

## Dataset

Place `Online Retail.xlsx` in the project root. The notebooks read this file directly:

```python
df_ori = pd.read_excel("Online Retail.xlsx")
```

The expected columns are `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

## Requirements

The project dependencies are listed in `requirements.txt`

## Setup

Create and activate a virtual environment, then install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name PREDICTING-FUTURE-CUSTOMER-PURCHASES-USING-MACHINE-LEARNING --display-name "Marketing Research"
```

The notebook imports include Plotly and SciPy in addition to the packages listed in `requirements.txt`. If these packages are not already available in the environment, install them before running the notebooks:

```powershell
python -m pip install plotly scipy
```

## Running the notebooks

Open VS Code, select the **Marketing Research** kernel, and run the notebooks in this order:

1. `data_analysis.ipynb` — data cleaning, exploratory analysis, RFM preparation, and K-means exploration.
2. `model_training.ipynb` — future-purchase target creation, customer segmentation, and predictive model comparison.

Run all cells with **Run All**. Both notebooks assume `Online Retail.xlsx` is in the project root.

## Data preparation

The notebooks validate and clean the transaction data by:

- removing rows with missing `CustomerID` values,
- removing rows with zero or negative `Quantity`,
- removing rows with zero or negative `UnitPrice`,
- identifying cancelled invoices, and
- converting `CustomerID` to an integer type.

The notebooks calculate sales amount as:

```text
Amount = UnitPrice × Quantity
```

## Exploratory analysis

`data_analysis.ipynb` analyzes:

- dataset structure and summary statistics,
- United Kingdom versus other markets,
- sales amount by country,
- top customers by sales amount and purchase frequency,
- top products by sales amount and purchase frequency,
- RFM distributions and skewness,
- data transformations for clustering, and
- K-means clustering with cluster-count evaluation.

Product descriptions with the same `StockCode` are standardized to a consistent description in the exploratory notebook.

## Temporal setup and RFM features

The final available month is used to determine the holdout period. The holdout starts at the first day of the month three calendar months before the latest month and ends at the latest transaction date.

Transactions before the holdout start form the historical observation window. For each historical customer, the notebook calculates:

- **Recency** — days between the customer's latest purchase and the holdout start,
- **Frequency** — number of unique invoices, and
- **Monetary Value** — total transaction amount.

The RFM distributions are skewed, so the notebooks apply transformations:

- Recency: square root,
- Frequency: log1p,
- Monetary Value: log1p.

The transformed features are normalized with `MinMaxScaler` before K-means clustering.

## Future-purchase target

`FuturePurchase` is the binary target used in `model_training.ipynb`:

- `1` — the historical customer has at least one unique invoice in the holdout window.
- `0` — the customer has no qualifying purchase in the holdout window.

The target is constructed with the holdout transactions only, while model features use only historical transactions.

## Segmentation

The notebooks evaluate K-means models using:

- Elbow/inertia,
- silhouette score,
- Davies-Bouldin score,
- Calinski-Harabasz score, and
- cluster-size checks.

Both notebooks fit a two-cluster K-means model. The example segment mapping used in `model_training.ipynb` is:

- Cluster `1`: `Inactive/Low-Value`
- Cluster `0`: `Loyal/High-Value`

The test customers receive cluster labels from the model fitted on the training customers.

## Predictive modeling

The customer data is split into 80% training and 20% testing using a fixed random seed and stratification on `FuturePurchase`.

The notebooks compare three classifiers:

- Logistic Regression,
- Random Forest, and
- XGBoost.

Each classifier is evaluated in two configurations:

1. **Without Segment** — Recency, Frequency, and Monetary Value only.
2. **With Segment** — RFM features plus the one-hot-encoded segment name.

Model selection uses 5-fold stratified cross-validation and grid search with ROC-AUC as the scoring metric. The test set is evaluated with probability-based ROC-AUC and classification metrics including precision, recall, F1, and the 0.5 probability threshold.

## Notebook-generated outputs

Running the notebooks can display:

- data summaries and cleaning counts,
- sales and customer rankings,
- RFM distribution plots,
- cluster metric tables,
- 3D cluster visualizations,
- future-purchase class counts,
- model performance tables and classification reports.

## Reproducibility notes

- Data splits use `random_state=42`.
- K-means clustering uses `random_state=42`.
- Cross-validation uses 5 stratified folds with shuffling and `random_state=42`.
- The two-cluster analysis is specified explicitly in the notebooks.
