# app.py
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import Nystroem

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils import clean_columns, TARGET_COL


# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="MyValuation DSS", layout="wide")
DATA_PATH = "data/sample.csv"


# ----------------- Helper: sklearn OneHotEncoder compatibility -----------------
def make_ohe():
    # Compatibility across scikit-learn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ----------------- Data Loading -----------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Missing data/sample.csv")
    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)
    return df


# ----------------- Regression Models (Train & Compare) -----------------
@st.cache_resource
def train_regression_models(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Check utils.py TARGET_COL.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )

    # Kernel Ridge Nystroem (KRN)
    krn = Pipeline([
        ("nystroem", Nystroem(kernel="rbf", gamma=0.2, n_components=300, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest (Small)": RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Kernel Ridge (Nystroem)": krn,
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained = {}
    rows = []

    for idx, (name, model) in enumerate(models.items(), start=1):
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        trained[name] = pipe
        rows.append({
            "No.": idx,
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
        })

    results_df = pd.DataFrame(rows)
    feature_cols = list(X.columns)
    return trained, results_df, feature_cols


# ----------------- Market Insights: Classification + Association Rules -----------------
@st.cache_resource
def build_market_insights(df: pd.DataFrame):
    """
    1) Classification: convert Transaction Price into price segments (Low/Mid/High) and train a classifier.
    2) Association Rules: mine rules that commonly lead to each price segment.
    """
    seg_labels = ["Low", "Mid", "High"]
    df_ins = df.copy()
    df_ins["Price_Segment"] = pd.qcut(df_ins[TARGET_COL], q=3, labels=seg_labels)

    # ---------- Classification ----------
    X = df_ins.drop(columns=[TARGET_COL])
    y_cls = df_ins["Price_Segment"].astype(str)

    if "Price_Segment" in X.columns:
        X = X.drop(columns=["Price_Segment"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    cls_pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    cls_pipe.fit(X_train, y_train)

    y_pred = cls_pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=seg_labels)
    report = classification_report(
        y_test, y_pred, labels=seg_labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Metric"})

    # ---------- Association Rules (Apriori via mlxtend) ----------
    rules_df = pd.DataFrame()
    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        df_ar = df_ins.copy()

        obj_cols = df_ar.select_dtypes(include=["object"]).columns.tolist()
        num_cols_all = df_ar.select_dtypes(exclude=["object"]).columns.tolist()
        if TARGET_COL in num_cols_all:
            num_cols_all.remove(TARGET_COL)

        num_keep = num_cols_all[:6]

        # Discretize numeric cols into 3 bins
        for c in num_keep:
            try:
                df_ar[c + "_Band"] = pd.qcut(df_ar[c], q=3, labels=["Low", "Mid", "High"])
            except Exception:
                pass

        keep_cols = []
        keep_cols += obj_cols
        keep_cols += [c for c in df_ar.columns if c.endswith("_Band")]
        keep_cols += ["Price_Segment"]

        df_arm = df_ar[keep_cols].copy()

        # Tokenize as "Col=Value"
        for c in df_arm.columns:
            df_arm[c] = c + "=" + df_arm[c].astype(str)

        onehot = pd.get_dummies(df_arm)

        itemsets = apriori(onehot, min_support=0.03, use_colnames=True)
        if not itemsets.empty:
            rules = association_rules(itemsets, metric="lift", min_threshold=1.05)

            rules = rules.copy()
            rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
            rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

            rules = rules[rules["consequents"].str.contains("Price_Segment=")]
            rules = rules.sort_values(["lift", "confidence"], ascending=False)

            rules_df = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(50)

    except Exception:
        rules_df = pd.DataFrame()

    return {
        "cls_model": cls_pipe,
        "cls_accuracy": acc,
        "cls_cm": cm,
        "cls_report_df": report_df,
        "seg_labels": seg_labels,
        "rules_df": rules_df,
    }


# ======================= UI =======================
st.title("🏡 MyValuation Decision Support System (DSS)")

df = load_data()

property_type_options = (
    sorted(df["Property_Type"].dropna().astype(str).unique().tolist())
    if "Property_Type" in df.columns else []
)
tenure_options = (
    sorted(df["Tenure"].dropna().astype(str).unique().tolist())
    if "Tenure" in df.columns else []
)
state_options = (
    sorted(df["state"].dropna().astype(str).unique().tolist())
    if "state" in df.columns else []
)
district_options = (
    sorted(df["District"].dropna().astype(str).unique().tolist())
    if "District" in df.columns else []
)

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Dataset", "Train & Compare", "Predict", "Market Insights"]
)


# ---------------- Overview ----------------
if menu == "Overview":
    st.subheader("Introduction")

    st.write(
        "**MyValuation DSS** is a web-based decision support system designed to help Malaysian netizens and "
        "average homebuyers estimate a **fair property transaction price** using data-driven methods."
    )

    st.write(
        "The system supports the project title: **“Replicating Market Stratification and Forecasting Future Property Trends "
        "in Malaysia (2021–2024)”** by learning market patterns from historical data."
    )

    st.markdown(
        """
**Main goal (for users):**
- Enter key property details and socio-economic indicators to obtain an estimated transaction price.

**Decision support capabilities:**
- **Forecasting (Regression):** predicts a continuous transaction price.
- **Market Interpretation (Insights):** classifies properties into Low/Mid/High segments and discovers common patterns using association rules.
        """
    )



# ---------------- Dataset ----------------
elif menu == "Dataset":
    st.subheader("Dataset Overview")

    st.info(
        "To keep the application responsive, the table below displays only the **first N rows** "
        "of the dataset in their original order. By default, **50 rows** are shown. "
        "You may increase this number, but displaying very large amounts of data may cause slower performance."
    )

    max_rows = int(df.shape[0])
    n = st.slider("Number of rows to display", 10, max_rows, 50, step=10)
    st.dataframe(df.head(n), use_container_width=True)

    st.write("Column data types:")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

    st.write("Basic statistics (numerical features):")
    st.dataframe(df.describe().T, use_container_width=True)


# ---------------- Train & Compare ----------------
elif menu == "Train & Compare":
    st.subheader("Model Training & Comparison (Regression)")

    with st.spinner("Training regression models (first run only)..."):
        trained, results_df, feature_cols = train_regression_models(df)

    st.success("Models trained and cached successfully.")
    st.dataframe(results_df, use_container_width=True)

    st.plotly_chart(
        px.bar(
            results_df,
            x="Model",
            y="RMSE",
            title="RMSE Comparison (Lower is Better)",
        ),
        use_container_width=True,
    )


# ---------------- Predict ----------------
elif menu == "Predict":
    st.subheader("Predict Transaction Price")

    with st.spinner("Loading best model (Random Forest)..."):
        trained, _, feature_cols = train_regression_models(df)

    # Always use best model
    model = trained["Random Forest (Small)"]

    st.caption(
        "Predictions are generated using the **Random Forest model**, which generally produces more realistic price "
        "relationships across property types."
    )

    user_input = {}
    for col in feature_cols:
        if col == "Property_Type":
            user_input[col] = st.selectbox("Property Type", property_type_options) if property_type_options else st.text_input("Property_Type", "")
        elif col == "Tenure":
            user_input[col] = st.selectbox("Tenure", tenure_options) if tenure_options else st.text_input("Tenure", "")
        elif col == "state":
            user_input[col] = st.selectbox("State", state_options) if state_options else st.text_input("state", "")
        elif col == "District":
            user_input[col] = st.selectbox("District", district_options) if district_options else st.text_input("District", "")
        elif col == "Month_Year":
            user_input[col] = st.text_input("Month_Year (YYYY-MM-DD)", "2024-01-01")
        else:
            user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_cols)
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Transaction Price: RM {pred:,.2f}")


# ---------------- Market Insights (Classification + Association Rules) ----------------
else:
    st.subheader("Market Insights")

    st.write(
        "This page provides additional **market interpretation** beyond a single price prediction:\n"
        "- **Classification**: groups properties into Low/Mid/High price segments.\n"
        "- **Association Rules**: shows common feature combinations that often lead to specific price segments."
    )

    with st.spinner("Building insights (first run only)..."):
        insights = build_market_insights(df)

    # ---- Classification ----
    st.markdown("### 1) Price Segment Classification (Low / Mid / High)")
    st.write(f"**Classifier Accuracy:** {insights['cls_accuracy']:.3f}")

    seg_labels = insights["seg_labels"]
    cm_df = pd.DataFrame(
        insights["cls_cm"],
        index=[f"Actual {s}" for s in seg_labels],
        columns=[f"Pred {s}" for s in seg_labels],
    )
    st.write("Confusion Matrix:")
    st.dataframe(cm_df, use_container_width=True)

    with st.expander("Show detailed classification report"):
        st.dataframe(insights["cls_report_df"], use_container_width=True)

    # ---- Association Rules ----
    st.markdown("### 2) Association Rules (Patterns leading to price segments)")

    if insights["rules_df"].empty:
        st.warning(
            "Association rules are not available right now. "
            "If you want this feature, install **mlxtend** (pip install mlxtend) and redeploy."
        )
    else:
        seg = st.selectbox(
            "View rules where the outcome (consequent) is:",
            ["All", "Price_Segment=Low", "Price_Segment=Mid", "Price_Segment=High"]
        )

        rules_view = insights["rules_df"].copy()
        if seg != "All":
            rules_view = rules_view[rules_view["consequents"].str.contains(seg)]

        with st.expander("What do these rules mean? (Interpretation guide)"):
            st.markdown(
                """
**Association Rules** show common patterns found in the dataset.

- **Antecedents** = the “IF” part (conditions that occur together)  
- **Consequents** = the “THEN” part (what often happens with those conditions)

✅ Example:
- IF `District=Johor Bahru` AND `Land/Parcel_Area_Band=High`  
- THEN `Price_Segment=High`

**Metrics explained:**
- **Support**: how frequently the pattern appears in the dataset (higher = more common)  
- **Confidence**: probability of the consequent given the antecedent (higher = more reliable)  
- **Lift**: strength compared to random chance  
  - **> 1** = useful positive association  
  - **≈ 1** = near chance  
  - **< 1** = negative association
                """
            )

        st.info(
            "Tip: Focus on rules with **high lift** and **high confidence** for stronger and more meaningful patterns."
        )

        # ✅ Renumber from 1..N (instead of original index)
        rules_view = rules_view.reset_index(drop=True)
        rules_view.insert(0, "No.", range(1, len(rules_view) + 1))

        st.dataframe(rules_view.head(20), use_container_width=True)
