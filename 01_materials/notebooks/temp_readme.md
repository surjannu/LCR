## Handling "Unknown" Values in Multiple Variables for Classification

In this dataset, several categorical variables contain `"unknown"` values. Proper handling of these is important because they can negatively affect model performance if treated incorrectly.

### Why Not Leave "Unknown" As-Is?
- Many machine learning algorithms interpret each category as meaningful.
- `"unknown"` is not a real category — it represents **missing information**, not a valid class.
- Treating it as a normal category can introduce noise and misleading patterns.

---

### Recommended Approaches

#### 1. Separate Binary Flag (Best Practice)
Create an additional feature indicating whether the value was originally unknown.

**Example**
Original column:
```
job: ["admin", "technician", "unknown", "services"]
```

Transformed:
```
job: ["admin", "technician", "services"]
job_unknown_flag: [0, 0, 1, 0]
```

**Why this works well**
- Preserves missingness information.
- Lets the model learn whether missingness itself is predictive.
- Avoids mixing unknowns with real categories.

---

#### 2. Replace with Mode (Simple Imputation)
Replace `"unknown"` with the most frequent value in that column.

**Pros**
- Easy and fast.

**Cons**
- May introduce bias.
- Assumes unknowns behave like the majority class.

---

#### 3. Treat Unknown as Separate Category
Keep `"unknown"` as its own category.

**When to use**
- If unknown is common.
- If missingness might carry meaning.

---

### Recommended Strategy for This Dataset
For classification tasks:

> **Use Binary Flag + Imputation (Hybrid Approach)**

Steps:
1. Create `_unknown_flag` column for each variable with unknowns.
2. Replace `"unknown"` with mode (or `"missing"` label).
3. Encode categorical features normally.

This approach typically improves performance because:
- The model sees both the cleaned category
- AND whether the value was originally missing.

---

### Example Implementation (Pandas)

```python
for col in categorical_cols:
    df[f"{col}_unknown_flag"] = (df[col] == "unknown").astype(int)
    mode_val = df[col].replace("unknown", pd.NA).mode()[0]
    df[col] = df[col].replace("unknown", mode_val)
```

---

### Summary

| Method | Recommended? | Reason |
|------|-------------|------|
Leave as "unknown" | ❌ | Adds noise |
Replace with mode | ⚠️ | May bias data |
Binary flag + replace | ✅ Best | Keeps information + cleans data |
Separate category | ✅ Sometimes | Good if unknown is frequent |

---

**Final Recommendation:**  
For most classification models, the **binary flag + replacement** method gives the best balance between data cleanliness and information retention.




# Feature Selection Report — Bank Marketing ML Project

## Objective
Identify the most relevant predictors for determining whether a customer will subscribe to a term deposit (`y`). The goal is to improve model accuracy, reduce noise, prevent overfitting, and enhance interpretability for business stakeholders.

---

## Dataset Summary
| Attribute | Value |
|--------|------|
Total Records | 41,188 |
Total Features | 21 |
Target | y (Yes/No) |
Numerical Features | 10 |
Categorical Features | 11 |
Explicit Missing Values | None |
Implicit Missing Values | Present as `"unknown"` |

---

## Data Quality Assessment
### Strengths
- Large dataset suitable for ML training
- Balanced mix of demographic, behavioral, and macroeconomic variables
- No null values

### Limitations
- Multiple categorical variables contain `"unknown"` values
- Target leakage risk from `duration` feature (call duration known only after call)
- Macroeconomic features may be correlated

---

## Preprocessing Performed
- Created binary flag columns for `"unknown"` values
- Replaced `"unknown"` with mode for modeling
- Label encoded categorical features
- Removed constant/near-constant columns
- Checked multicollinearity via correlation matrix
- Dropped highly correlated features (>0.85 threshold)

---

## Feature Selection Techniques Used

### Filter Methods
- Chi-Square Test
- Mutual Information

Purpose: Measure statistical dependency between each feature and target.

---

### Embedded Methods
- Random Forest Feature Importance

Purpose: Identify features that contribute most to prediction during training.

---

### Wrapper Methods
- Recursive Feature Elimination (RFE)

Purpose: Iteratively select optimal subset of predictors.

---

## Selected Features
*(Update after running your notebook)*

### Note:
Although duration is highly predictive, it causes data leakage because it is only known after the call. Therefore it must be excluded from the production model to ensure realistic predictions.
