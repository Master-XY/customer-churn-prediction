# Telco Customer Churn Prediction — Project Progress

## 1. Project
- **Title:** Customer Churn Prediction Using Machine Learning
- **Level:** Beginner Data Science / ML semester project
- **Goal:** Build an understandable customer churn prediction project for a resume and explain every step to the professor.
- **Approach:** Keep the project simple and understandable; do not over-engineer.
- **Planned model:** Logistic Regression (binary classification).
- **Planned libraries:** NumPy, Pandas, Matplotlib, Scikit-learn.

## 2. Dataset
- **Dataset:** Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Rows:** 7043
- **Columns:** 21
- **Target:** `Churn` (`Yes` / `No`)

Columns:
`customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn`

## 3. Environment
- Project repository: `customer-churn-prediction`
- Notebook: `customer-churn.ipynb`
- Local workflow: VS Code + Jupyter Notebook
- Local project path: `C:\Desktop\Master_XY\customer-churn-prediction`
- Dataset is stored locally under `data/`.
- DataFrame changes happen in RAM; the original CSV is unchanged unless explicitly saved.

Recommended structure:
```text
customer-churn-prediction/
├── README.md
├── PROJECT_PROGRESS.md
├── customer-churn.ipynb
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── .gitignore
```

## 4. Teaching / workflow preference
- Teach like a teacher: ask questions and give hints before giving answers.
- Do not provide complete project code unless explicitly requested.
- User should write the code and test it.
- Give small syntax/function hints when needed.
- Keep explanations short and practical.
- Focus on understanding what operation/function is needed, rather than memorizing exact syntax.
- Avoid unnecessary graphs, libraries, or complexity.

Preferred workflow:
`user decides goal → user attempts code → error/stuck → AI explains concept/hints → user fixes code → test → continue`

## 5. Data cleaning completed

### TotalCharges
- `df.info()` initially showed `TotalCharges` as `object`.
- `df.isnull().sum()` initially showed no nulls because the problematic values were blank strings.
- Converting with `pd.to_numeric(..., errors="coerce")` revealed **11 invalid/blank values** in `TotalCharges`.
- All 11 occurred where `tenure == 0`.
- Decision: treat these blank `TotalCharges` values as `0`, because zero tenure means no accumulated tenure-based charges.
- Cleaning completed conceptually as:
  `TotalCharges → numeric → invalid values become NaN → NaN filled with 0`
- `TotalCharges` is now `float64`.
- Do not claim the dataset proves why those customers had blank charges; only the observed data supports the cleaning decision.

### Customer IDs / duplicates
- `customerID` is intended to uniquely identify customers and is a candidate for dropping before modeling because it is an identifier, not a useful predictive feature.
- Repeated values in a column do NOT automatically mean duplicate rows.
- `df["Dependents"].duplicated().sum()` returned 7041 because `Dependents` only contains repeated Yes/No values; this does not mean 7041 duplicate rows.
- For true duplicate-row checking, use the whole DataFrame, not one categorical column.

### Categorical values
- `DeviceProtection`, `OnlineBackup`, and `InternetService` contain meaningful categories such as `No internet service`.
- Decision: do not delete or arbitrarily transform those categories during cleaning.
- Encoding will be handled later during preprocessing.
- Do not infer socioeconomic status or similar attributes from `No internet service`.

## 6. Basic numerical EDA completed

### MonthlyCharges
- Minimum: 18.25
- Maximum: 118.75
- Mean: 64.76169246059918
- Median: 70.35

### TotalCharges
- Minimum: 0.0
- Maximum: 8684.8
- Mean: 2279.7343035638223
- Median: 1394.55

Interpretation:
- `MonthlyCharges` has a reasonable observed range.
- `TotalCharges` has mean substantially above median, suggesting a right-skewed distribution.
- High `TotalCharges` values are not automatically outliers because longer-tenure customers can legitimately accumulate higher charges.
- `TotalCharges` is naturally related to `MonthlyCharges` and `tenure`.

## 7. EDA: tenure and TotalCharges
- A scatter plot of `tenure` vs `TotalCharges` was made.
- The relationship was obvious and expected: `TotalCharges` generally increases as `tenure` increases.
- This graph is optional; a written finding is sufficient for this project because it does not add much beyond the obvious relationship.

## 8. EDA: Contract vs tenure
`Contract` categories:
- `Month-to-month`
- `One year`
- `Two year`

A box plot of `Contract` vs `tenure` was made.

Observed pattern:
- Month-to-month customers generally have shorter tenure.
- One-year customers generally have longer tenure.
- Two-year customers generally have the longest tenure.
- There are potential boxplot outliers; these are not automatically errors and may be legitimate customers.

Important lesson:
- Do NOT encode contract categories as `0, -1, -2` merely for visualization. That creates an artificial numerical ordering/distance.
- `Contract` is categorical and will be encoded appropriately during preprocessing.

## 9. EDA: Churn distribution
`df["Churn"].value_counts()`:

```text
No     5174
Yes    1869
```

Total = 7043.

Approximate distribution:
- `No`: 73.5%
- `Yes`: 26.5%

Finding:
- The target classes are imbalanced; there are substantially more non-churned customers than churned customers.

A bar chart was considered but is optional because the counts themselves communicate the finding.

## 10. Current EDA: Contract vs Churn

Hypothesis:
> Customers with longer contracts may have lower churn rates.

We correctly selected churn values for a contract category using:

```python
df[df["Contract"] == "Month-to-month"]["Churn"]
```

Then counted them with:

```python
df[df["Contract"] == "Month-to-month"]["Churn"].value_counts()
```

The same should be done for `One year` and `Two year`.

Observed so far:
- Churn appears to decrease as contract duration increases.

Important:
- Do not compare only raw churn counts because the contract groups contain different numbers of customers.
- We need **churn rate within each contract group**:

`churned customers in contract group / total customers in that contract group × 100`

Current task:
- Calculate the churn percentage for Month-to-month, One year, and Two year.
- User was considering finding the total number of rows for each contract category and dividing the `Yes` count by that total.

## 11. Planned project pipeline

1. Load dataset
2. Understand dataset
3. Clean data
4. Exploratory Data Analysis
5. Encode categorical variables
6. Define features `X` and target `y`
7. Train/test split
8. Train Logistic Regression
9. Make predictions
10. Evaluate with:
   - Accuracy
   - Confusion matrix
   - Precision
   - Recall
   - F1-score
11. Interpret results
12. Conclusion

## 12. Graph policy for this project
Do not create a graph for every variable.

Use a graph only when it materially helps:
- discover a relationship,
- compare groups,
- understand a distribution,
- or explain an important result.

Currently:
- `tenure` vs `TotalCharges` scatter plot: optional
- `Churn` bar chart: optional
- `Contract` vs `tenure` box plot: useful
- Future graphs should be chosen based on an actual question.

## 13. Important concepts already learned
- `df["column"].unique()` → unique values in one Series.
- `df.nunique()` → number of unique values in columns.
- `.value_counts()` → frequency of each unique value.
- `.duplicated()` on one column checks repeated values in that column, not duplicate rows.
- Boolean filtering:
  `df[df["Contract"] == "Month-to-month"]`
- Selecting a column after filtering:
  `df[df["Contract"] == "Month-to-month"]["Churn"]`
- `pd.to_numeric(..., errors="coerce")` converts numeric-looking strings and turns invalid values into `NaN`.
- `.fillna(0)` replaces missing values with zero.
- DataFrame operations are case-sensitive.
- In Jupyter, separate expressions in one cell may only visibly display the last expression unless printed/combined.

## 14. Next immediate task
Finish **Contract vs Churn rate** for all three contract types.

After that:
- decide whether any other EDA questions are genuinely useful,
- finish preprocessing,
- then move to encoding and model preparation.

Do not jump directly to a large code block. Continue with the teacher-style, hint-first workflow.
