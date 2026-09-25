# FMCG Smart Launch: Product Cannibalization & Demand Forecasting

An end-to-end FMCG analytics project built to support product launch, portfolio monitoring, and short-term demand planning.

The project combines exploratory data analysis, product lifecycle analysis, price response analysis, product cannibalization analysis, and demand forecasting using daily FMCG sales data from 2022–2024.

> **Project title:** FMCG Smart Launch: Mitigasi Kanibalisasi Produk dan Prediksi Permintaan Pasar Makro
>
> **Note:** The current dataset does not contain external macroeconomic variables. The demand forecasting part therefore focuses on sales history, product timing, promotion, and calendar patterns available in the dataset.

---

## 1. Business Context

Launching a new FMCG product can create additional sales, but it can also change demand across products that already exist in the portfolio. At the same time, a launch decision needs a realistic view of expected demand so the business can prepare inventory and plan promotions.

This project was designed around two practical questions:

1. **What happens to existing products when new SKUs enter the portfolio?**
2. **How accurately can future demand be estimated from the sales information already available?**

Rather than treating every sales decline as cannibalization, the analysis compares existing-product performance before and after a new SKU appears and checks the broader segment movement at the same time.

For forecasting, the project uses a chronological train-test split so that future observations are not used to predict the past.

---

## 2. Project Objectives

- Understand sales and product performance across time, category, channel, and region.
- Analyze how product performance develops over the observed product lifecycle.
- Examine the observed relationship between price and demand.
- Identify potential signs of product cannibalization after new SKU introductions.
- Build a demand forecasting model using historical sales patterns.
- Translate analytical results into practical product launch and demand planning insights.

---

## 3. Dataset

**Dataset:** FMCG Daily Sales Data 2022–2024  
**Source:** Kaggle  
**Period:** 21 January 2022 – 31 December 2024  
**Rows:** 190,757  
**Columns:** 14  
**Unique SKUs:** 30

Dataset source:
https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024

### Main variables

| Variable | Description |
|---|---|
| `date` | Daily transaction date |
| `sku` | Product identifier |
| `brand` | Brand associated with the SKU |
| `segment` | Product segment |
| `category` | Product category |
| `channel` | Sales channel |
| `region` | Sales region |
| `pack_type` | Product packaging type |
| `price_unit` | Unit price |
| `promotion_flag` | Promotion indicator (0/1) |
| `delivery_days` | Delivery duration |
| `stock_available` | Available stock |
| `delivered_qty` | Delivered quantity |
| `units_sold` | Units sold; main demand target |

---

## 4. Analytical Workflow

```text
FMCG Daily Sales Data
        |
        v
Data Understanding
        |
        v
Data Cleaning & Validation
        |
        v
Exploratory Data Analysis
        |
        v
Product Lifecycle Analysis
        |
        v
Price Elasticity Analysis
        |
        v
Product Cannibalization Analysis
        |
        v
Demand Forecasting
        |
        v
Model Evaluation
        |
        v
Business Insights & Recommendations
```

---

# 5. Data Cleaning & Validation

The raw dataset was checked before any business analysis was performed.

### Checks performed

- Converted `date` from object to `datetime64[ns]`.
- Checked missing values.
- Checked duplicate rows.
- Reviewed unique values for key categorical fields.
- Checked negative values in core numerical variables.
- Confirmed `promotion_flag` structure.
- Preserved the original observations instead of removing records simply to make the dataset easier to model.

### Cleaning result

| Check | Result |
|---|---:|
| Final observations | 190,757 |
| Variables | 14 |
| Duplicate rows | 0 |
| Missing values | 0 |
| Date data type | `datetime64[ns]` |

The cleaned dataset was saved as:

```text
data/processed/FMCG_clean.csv
```

---

# 6. Exploratory Data Analysis

EDA was used to understand the sales structure before moving into the more specific launch and forecasting analyses.

The analysis covered:

- Daily and monthly sales trends
- SKU performance
- Category performance
- Channel performance
- Regional performance
- Promotion vs. non-promotion demand
- Price vs. units sold
- Product structure across brands, segments, categories, channels, regions, and pack types

### Example questions answered

**How does demand move over time?**  
Monthly and daily sales trends were reviewed to identify changes in demand and recurring patterns.

**Which products carry the most volume?**  
SKU-level aggregation was used to compare total units sold, sales value, average price, and other operating measures.

**Does promotion coincide with different sales levels?**  
Average demand was compared between promoted and non-promoted observations. This is treated as a descriptive relationship, not proof that promotion caused the difference.

**How does price relate to demand?**  
A price-demand view was created as the starting point for the elasticity analysis.

---

# 7. Product Lifecycle Analysis

The dataset contains **30 SKUs**.

Based on the first observed sales date:

- **20 SKUs** were first observed in 2022.
- **10 SKUs** were first observed in 2023.
- No SKU had a first observed date in 2024.

The earliest observed product date was **21 January 2022**, while the latest observation in the dataset was **31 December 2024**.

The average observed SKU duration was **831.5 days**, with a median of **823 days**.

### Product age pattern

Demand varied across product age groups.

| Product age | Average units sold |
|---|---:|
| 0–30 days | 16.75 |
| 31–90 days | 22.45 |
| 91–180 days | **25.45** |
| 181–365 days | 19.83 |
| 366–730 days | 18.99 |
| 730+ days | 18.22 |

The highest observed average demand occurred during the **91–180 day** group.

This is interpreted as an observed pattern rather than proof that product age itself causes demand to rise or fall.

### Important definition

The dataset does not contain official product launch dates. For this reason, the project uses the **first observed sales date as a proxy for product introduction**.

---

# 8. Price Elasticity Analysis

Price elasticity was analyzed in two stages.

## 8.1 Baseline analysis

A simple SKU-level log-log model initially produced a near-zero relationship between price and units sold. The pooled baseline produced:

- Elasticity: **-0.0010**
- R-squared: **~0.0000**

This indicated that a simple price-only relationship was not sufficient for the dataset.

## 8.2 Controlled model

A more complete log-log regression was then used to control for:

- SKU
- Promotion
- Channel
- Region
- Month
- Year

The model estimated:

| Metric | Result |
|---|---:|
| Price elasticity | **-0.001905** |
| P-value | **0.366912** |
| R-squared | **0.296913** |
| Observations | **186,892** |

The elasticity coefficient is close to zero, and the p-value does not provide sufficient statistical evidence of a meaningful price-demand relationship within this model.

The result should not be read as “price has no effect on demand” in a broader business sense. It means that **this dataset and this model do not provide strong statistical evidence of a meaningful price-demand relationship after the included controls are considered**.

The elasticity estimate is treated as an observed conditional relationship, not a causal estimate.

---

# 9. Product Cannibalization Analysis

Cannibalization is different from price elasticity.

- **Price elasticity** asks how demand changes alongside price changes.
- **Cannibalization** asks whether the introduction of a new product coincides with weaker demand for existing products in the same portfolio.

## Method

The analysis focused on SKUs first observed in **2023**, because 2023 introductions provide a reasonable pre- and post-introduction window within the available data.

For each new SKU:

1. Identify existing SKUs in the same segment.
2. Use a 90-day pre-introduction period.
3. Use a 90-day post-introduction period.
4. Compare average daily demand of the comparable existing SKUs.
5. Compare the result with segment-level demand excluding the newly introduced SKUs.

The logic was intentionally conservative.

```text
Existing peer demand declines
            +
Segment demand is stable/increasing
            |
            v
Potential cannibalization indication
```

A decline in an existing product was **not** automatically classified as cannibalization.

## Results

There were **10 SKUs first observed in 2023**.

| Result | Events |
|---|---:|
| New SKU introduction events | 10 |
| Events with sufficient comparison | 8 |
| No clear cannibalization indication | 6 |
| Insufficient comparison | 2 |
| Potential cannibalization indication | **0** |

Two introduction events showed a decline in comparable peer demand, but the corresponding segment demand also declined. Therefore, they did not meet the preliminary criterion for a potential cannibalization indication.

Two other events did not have enough comparable existing SKUs and were therefore classified as **insufficient comparison**, rather than being interpreted as evidence of no cannibalization.

### Interpretation

The analysis did not identify a clear potential cannibalization indication under the defined framework. This does **not** mean cannibalization cannot occur; it means that the available data and analytical rule did not provide sufficient evidence for an identified event.

The 90-day window is an analytical choice and different windows may produce different results.

---

# 10. Demand Forecasting

The forecasting target is **daily total units sold**.

The forecasting model uses information that would reasonably be available from historical sales records, including:

- 1-day lag
- 7-day lag
- 14-day lag
- 28-day lag
- 7-day rolling mean
- 14-day rolling mean
- 28-day rolling mean
- Calendar features
- Promotion rate

### Why time-aware validation?

A random train-test split was avoided because forecasting is a time-dependent problem. The dataset was split chronologically so that the model learns from earlier observations and is tested on a later period.

### Model

**HistGradientBoostingRegressor** was selected for the baseline machine learning forecast because it can capture nonlinear relationships while remaining straightforward to reproduce with scikit-learn.

A simple previous-day demand forecast was used as the naive baseline.

---

# 11. Forecasting Results

The machine learning model improved all three error metrics compared with the naive baseline.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive Baseline | 227.06 | 288.59 | 5.37% |
| HistGradientBoosting | **206.38** | **259.85** | **5.00%** |

### Improvement over baseline

- MAE improved by approximately **9.10%**.
- RMSE improved by approximately **9.96%**.
- MAPE improved by approximately **6.97%**.

The final model evaluation on the chronological test set was:

| Metric | Result |
|---|---:|
| MAE | **206.38 units** |
| RMSE | **259.85 units** |
| MAPE | **5.00%** |
| R-squared | **0.7639** |

The model's mean forecast error was **-128.78 units**, indicating a tendency to over-forecast demand on average during the test period.

---

# 12. Feature Importance

Permutation importance was used to understand which features contributed most to model performance.

| Feature | Importance |
|---|---:|
| `rolling_mean_14` | **68.63** |
| `rolling_mean_7` | **66.53** |
| `month_cos` | **39.98** |
| `rolling_mean_28` | 16.84 |
| `promotion_rate` | 14.93 |
| `lag_1` | 10.92 |
| `lag_28` | 8.08 |
| `lag_14` | 3.62 |
| `lag_7` | 3.52 |
| `month_sin` | -0.28 |
| `dow_sin` | -1.89 |
| `dow_cos` | -2.56 |

The strongest signals came from the **7-day and 14-day rolling demand patterns**, suggesting that recent demand history carried more predictive information than individual lag variables alone in this model.

`promotion_rate` also contributed to model performance, although feature importance should not be interpreted as proof that a feature causes demand changes.

---

# 13. Key Business Insights

### Product portfolio

The portfolio contains 30 SKUs with new product introductions concentrated in 2022 and 2023. Product age is associated with different observed demand levels, so launch monitoring should consider how long a SKU has been in the observed market.

### Pricing

The controlled elasticity model produced a coefficient close to zero and was not statistically significant. Within this dataset, pricing alone does not provide a strong statistical explanation of demand after the included controls.

### Cannibalization

No analyzed event met the preliminary criterion for potential cannibalization. This reinforces the importance of checking broader segment demand before interpreting a decline in an existing SKU as a portfolio cannibalization effect.

### Forecasting

Historical demand patterns were the strongest signals in the tested forecasting model. The machine learning model improved on the naive baseline across MAE, RMSE, and MAPE and achieved a test-set MAPE of approximately 5%.

---

# 14. Recommendations

### 1. Use recent demand patterns in launch planning

Recent rolling demand can provide useful information for short-term demand planning. Forecasting should therefore be part of the launch preparation process rather than an afterthought.

### 2. Monitor new and existing SKUs together

When launching a new SKU, monitor related existing products at the same time. A decline in an existing SKU should be reviewed alongside segment-level demand before being treated as a potential cannibalization issue.

### 3. Do not rely on price alone

The tested price elasticity model did not show a statistically significant price-demand relationship. Pricing decisions should therefore be reviewed together with demand history, promotions, channel, product characteristics, and timing.

### 4. Track forecast bias

The model showed an average forecast error of -128.78 units. Forecast monitoring should therefore include bias tracking so the model can be recalibrated when systematic over-forecasting continues.

### 5. Re-evaluate the model as new data arrives

The forecasting model was trained and tested on the observed 2022–2024 period. Its performance should be checked periodically as new sales data and different market conditions become available.

---

# 15. Project Structure

```text
fmcg-smart-launch-intelligence/
│
├── data/
│   ├── raw/
│   │   └── FMCG_2022_2024.csv
│   └── processed/
│       └── FMCG_clean.csv
│
├── notebooks/
│   └── FMCG_Smart_Launch_Analysis.ipynb
│
├── results/
│   ├── sku_lifecycle.csv
│   ├── cannibalization_analysis.csv
│   ├── forecast_evaluation.csv
│   ├── forecast_feature_importance.csv
│   └── forecast_model_comparison.csv
│
├── assets/
│   ├── eda_daily_units_sold_trend.png
│   ├── eda_monthly_units_sold_trend.png
│   ├── eda_top_15_sku.png
│   ├── eda_units_sold_by_category.png
│   ├── eda_units_sold_by_channel.png
│   ├── eda_units_sold_by_region.png
│   ├── eda_promotion_vs_no_promotion.png
│   ├── eda_price_vs_units_sold.png
│   ├── lifecycle_new_sku_by_year.png
│   ├── lifecycle_monthly_new_sku.png
│   ├── lifecycle_units_by_product_age.png
│   ├── elasticity_average_price_vs_units.png
│   ├── elasticity_by_sku.png
│   ├── cannibalization_peer_demand_change.png
│   ├── cannibalization_segment_demand_change.png
│   ├── forecast_actual_vs_predicted.png
│   ├── evaluation_forecast_error_distribution.png
│   ├── evaluation_absolute_error_over_time.png
│   └── evaluation_feature_importance.png
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 16. Tools & Technologies

- **Python** — data preparation, analysis, modeling
- **Pandas / NumPy** — data manipulation and numerical analysis
- **Matplotlib / Seaborn** — exploratory visualization
- **Scikit-learn** — regression, forecasting, metrics, and permutation importance
- **Statsmodels** — controlled regression for price elasticity
- **Jupyter Notebook** — analysis workflow and documentation
- **Streamlit** — dashboard layer planned for project deployment

---

# 17. Analytical Outputs

The project produces several reusable outputs for reporting and dashboard development:

```text
results/
├── sku_lifecycle.csv
├── cannibalization_analysis.csv
├── forecast_evaluation.csv
├── forecast_feature_importance.csv
└── forecast_model_comparison.csv
```

The `assets/` folder contains the charts generated throughout the analysis so that the results can be reviewed without opening the notebook.

---

# 18. Limitations

This project is intentionally based on one FMCG sales dataset, so several limitations apply.

1. **No official launch date**  
   First observed sales date is used as a proxy for product introduction.

2. **No external macroeconomic variables**  
   The current forecasting model does not include inflation, GDP, interest rates, consumer confidence, or other external macro indicators.

3. **Cannibalization is not causal inference**  
   The analysis identifies patterns around SKU introduction but does not prove that a new SKU caused a decline in another SKU.

4. **90-day event window**  
   Cannibalization results depend partly on the selected pre- and post-introduction period.

5. **Price elasticity is model-specific**  
   The elasticity coefficient describes the conditional relationship captured by the tested regression rather than a controlled experiment.

6. **Forecast performance is period-dependent**  
   The reported metrics describe the chronological test period from the available dataset and may change when market conditions change.

---

# 19. Next Development

The next stage of the project is to turn the analysis into an interactive Streamlit dashboard so a recruiter or business user can explore the results without opening the notebook.

The planned dashboard will focus on:

- Executive KPI overview
- Sales trend monitoring
- SKU and category performance
- Product lifecycle monitoring
- Price and promotion analysis
- Cannibalization event monitoring
- Forecasted vs. actual demand
- Forecast model performance

The goal is to keep the dashboard focused on business questions rather than simply displaying every available metric.

---

## 20. What This Project Demonstrates

This project demonstrates an end-to-end analytics workflow:

```text
Raw Data
→ Cleaning
→ Validation
→ EDA
→ Business Metrics
→ Statistical Analysis
→ Portfolio Analysis
→ Forecasting
→ Model Evaluation
→ Business Recommendations
```

The emphasis is on making the analysis useful for an actual business decision: understanding portfolio changes, checking potential product interactions, and improving short-term demand planning with a reproducible forecasting approach.

---

## Author

**Naufal Fakhri**  
Data Analytics & Business Intelligence Portfolio
