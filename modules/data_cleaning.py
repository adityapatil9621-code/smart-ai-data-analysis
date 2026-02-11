import pandas as pd
import numpy as np

# =========================================
# GLOBAL CONFIG
# =========================================
SAFE_MODE = True   # If True, preserves missing values when imputation may bias analysis


# =========================================
# SEMANTIC ROLE DETECTION
# =========================================
def detect_column_role(col_name, dtype):
    name = col_name.lower()

    if "id" in name:
        return "identifier"
    if "age" in name:
        return "age"
    if any(k in name for k in ["salary", "amount", "price", "cost"]):
        return "monetary"
    if np.issubdtype(dtype, np.number):
        return "numeric"
    if dtype == object:
        return "categorical"
    if np.issubdtype(dtype, np.datetime64):
        return "datetime"

    return "unknown"


# =========================================
# MAIN CLEANING FUNCTION
# =========================================
def clean_data(df):
    cleaned_df = df.copy()
    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

    explanation = []
    warnings = []
    impact_report = {}

    total_rows = len(cleaned_df)

    # =====================================
    # 1️⃣ DUPLICATE REMOVAL
    # =====================================
    dup_count = cleaned_df.duplicated().sum()
    if dup_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        explanation.append(
            f"Removed {dup_count} duplicate rows to prevent statistical bias."
        )

    # =====================================
    # 2️⃣ COLUMN-WISE INTELLIGENT CLEANING
    # =====================================
    for col in cleaned_df.columns:

        missing_before = cleaned_df[col].isna().sum()
        missing_ratio = missing_before / total_rows
        dtype = cleaned_df[col].dtype
        role = detect_column_role(col, dtype)

        impact_report[col] = {
            "role": role,
            "missing_before": int(missing_before),
            "values_modified": 0,
            "action": None,
            "confidence": None
        }

        if missing_before == 0:
            impact_report[col]["confidence"] = "High"
            continue

        # =================================
        # IDENTIFIERS → NEVER IMPUTE
        # =================================
        if role == "identifier":
            explanation.append(
                f"Column '{col}' identified as identifier → missing values preserved."
            )
            impact_report[col]["action"] = "preserved_missing"
            impact_report[col]["confidence"] = "High"
            continue

        # =================================
        # AGE (DOMAIN-CONSTRAINED)
        # =================================
        if role == "age":
            median_age = round(cleaned_df[col].median())
            new_col = cleaned_df[col].fillna(median_age)

            # Enforce real-world constraints
            new_col = new_col.clip(lower=0, upper=120).astype(int)

            impact_report[col]["values_modified"] = int(
                new_col.ne(cleaned_df[col]).sum()
            )

            cleaned_df[col] = new_col

            explanation.append(
                f"Column '{col}' treated as Age → median imputation, integer enforced, valid range applied."
            )

            impact_report[col]["action"] = "median_int_domain_enforced"
            impact_report[col]["confidence"] = "High"
            continue

        # =================================
        # MONETARY VALUES
        # =================================
        if role == "monetary":
            skew = cleaned_df[col].skew()
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((cleaned_df[col] < q1 - 1.5 * iqr) |
                        (cleaned_df[col] > q3 + 1.5 * iqr)).sum()

            fill_value = cleaned_df[col].median()
            new_col = cleaned_df[col].fillna(fill_value)

            impact_report[col]["values_modified"] = int(
                new_col.ne(cleaned_df[col]).sum()
            )

            cleaned_df[col] = new_col

            explanation.append(
                f"Column '{col}' treated as monetary → median used "
                f"(skew={round(skew,2)}, outliers={outliers}) to reduce distortion."
            )

            impact_report[col]["action"] = "median_outlier_safe"
            impact_report[col]["confidence"] = "High"
            continue

        # =================================
        # GENERIC NUMERIC
        # =================================
        if role == "numeric":
            if missing_ratio < 0.05:
                value = cleaned_df[col].mean()
                method = "mean"
                confidence = "High"
            else:
                value = cleaned_df[col].median()
                method = "median"
                confidence = "Medium"

            new_col = cleaned_df[col].fillna(value)

            impact_report[col]["values_modified"] = int(
                new_col.ne(cleaned_df[col]).sum()
            )

            cleaned_df[col] = new_col

            explanation.append(
                f"Column '{col}' numeric → filled using {method}."
            )

            impact_report[col]["action"] = method
            impact_report[col]["confidence"] = confidence
            continue

        # =================================
        # CATEGORICAL / BEHAVIORAL
        # =================================
        if role == "categorical":

            # LOW MISSING
            if missing_ratio < 0.10:
                value = cleaned_df[col].mode().iloc[0]
                new_col = cleaned_df[col].fillna(value)

                impact_report[col]["values_modified"] = int(
                    new_col.ne(cleaned_df[col]).sum()
                )

                cleaned_df[col] = new_col

                explanation.append(
                    f"Column '{col}' categorical → filled using mode ('{value}')."
                )

                impact_report[col]["action"] = "mode"
                impact_report[col]["confidence"] = "High"

            # MODERATE MISSING
            elif 0.10 <= missing_ratio <= 0.30:
                numeric_cols = cleaned_df.select_dtypes(include=np.number).columns

                if len(numeric_cols) > 0:
                    ref_col = numeric_cols[0]

                    new_col = cleaned_df.groupby(
                        pd.qcut(cleaned_df[ref_col], 4, duplicates="drop"),
                        observed=True
                    )[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)
                    )

                    impact_report[col]["values_modified"] = int(
                        new_col.ne(cleaned_df[col]).sum()
                    )

                    cleaned_df[col] = new_col

                    explanation.append(
                        f"Column '{col}' categorical → conditional mode applied to preserve behavior."
                    )

                    impact_report[col]["action"] = "conditional_mode"
                    impact_report[col]["confidence"] = "Medium"
                else:
                    if SAFE_MODE:
                        warnings.append(
                            f"Column '{col}' has moderate missing values but no numeric reference → values preserved."
                        )
                        impact_report[col]["action"] = "preserved_missing"
                        impact_report[col]["confidence"] = "Low"
                    else:
                        new_col = cleaned_df[col].fillna("Unknown")
                        impact_report[col]["values_modified"] = int(
                            new_col.ne(cleaned_df[col]).sum()
                        )
                        cleaned_df[col] = new_col
                        impact_report[col]["action"] = "unknown_fallback"
                        impact_report[col]["confidence"] = "Low"

            # HIGH MISSING
            else:
                warnings.append(
                    f"Column '{col}' has high missing ratio ({round(missing_ratio*100,1)}%). "
                    f"Imputation skipped to avoid analytical bias."
                )
                impact_report[col]["action"] = "preserved_missing"
                impact_report[col]["confidence"] = "Low"

            continue

        # =================================
        # DATETIME
        # =================================
        if role == "datetime":
            new_col = cleaned_df[col].fillna(method="ffill")

            impact_report[col]["values_modified"] = int(
                new_col.ne(cleaned_df[col]).sum()
            )

            cleaned_df[col] = new_col

            explanation.append(
                f"Column '{col}' datetime → forward fill applied for temporal continuity."
            )

            impact_report[col]["action"] = "forward_fill"
            impact_report[col]["confidence"] = "Medium"

    explanation.append("Real-world intelligent data cleaning completed.")
    return cleaned_df, explanation, warnings, impact_report
def compute_data_quality_score(impact_report, total_rows):
    column_scores = {}
    total_score = 0

    for col, info in impact_report.items():
        score = 100

        missing_ratio = info["missing_before"] / total_rows
        modified_ratio = info["values_modified"] / total_rows

        # Missing data penalties
        if missing_ratio > 0.30:
            score -= 30
        elif 0.10 <= missing_ratio <= 0.30:
            score -= 15

        # Modification penalty
        if modified_ratio > 0.20:
            score -= 20

        # Action-based penalty
        if info["action"] in ["preserved_missing"]:
            score -= 15

        # Confidence-based penalty
        if info["confidence"] == "Low":
            score -= 10
        elif info["confidence"] == "Medium":
            score -= 5

        column_scores[col] = max(score, 0)
        total_score += column_scores[col]

    dataset_score = round(total_score / len(column_scores), 2)
    return column_scores, dataset_score
def compute_drift_metrics(before_df, after_df):
    drift_report = {}

    for col in before_df.columns:
        if col not in after_df.columns:
            continue

        # Numeric drift
        if pd.api.types.is_numeric_dtype(before_df[col]):
            before_mean = before_df[col].mean()
            after_mean = after_df[col].mean()

            before_std = before_df[col].std()
            after_std = after_df[col].std()

            drift_report[col] = {
                "mean_change_%": None if before_mean == 0 else round(
                    ((after_mean - before_mean) / before_mean) * 100, 2
                ),
                "std_change_%": None if before_std == 0 else round(
                    ((after_std - before_std) / before_std) * 100, 2
                )
            }

        # Categorical drift
        elif before_df[col].dtype == object:
            before_dist = before_df[col].value_counts(normalize=True)
            after_dist = after_df[col].value_counts(normalize=True)

            categories = set(before_dist.index).union(after_dist.index)
            drift = sum(
                abs(before_dist.get(cat, 0) - after_dist.get(cat, 0))
                for cat in categories
            )

            drift_report[col] = {
                "distribution_drift": round(drift, 3)
            }

    return drift_report
