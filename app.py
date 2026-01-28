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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import Nystroem

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import clean_columns, TARGET_COL

st.set_page_config(page_title="Property Price Forecasting", layout="wide")

# ✅ Use sample or real dataset
# DATA_PATH = "data/sample.csv"
DATA_PATH = "data/Norm_Fused_Dataset.csv"   # <-- switch to your real dataset when ready


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Add it to your repo.")
    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)
    return df


@st.cache_resource
def train_models(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Check utils.py TARGET_COL.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Compatibility: use sparse=False (works across older sklearn versions)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ],
        remainder="drop",
    )

    # ✅ Kernel Ridge Nystroem (KRN) = Nystroem + Ridge
    krn_model = Pipeline([
        ("nystroem", Nystroem(kernel="rbf", gamma=0.2, n_components=300, random_state=42)),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])

    model_defs = {
        "Linear Regression": LinearRegression(),

        "Random Forest (Small)": RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),

        "KNN": KNeighborsRegressor(n_neighbors=5),

        "Kernel Ridge (Nystroem)": krn_model,

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
    results = {}

    for name, model in model_defs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        trained[name] = pipe
        results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    return trained, results, list(X.columns)


# ---------- UI ----------
st.title("🏠 Property Price Forecasting (Train-in-Cloud)")

df = load_data()

# dropdown options from dataset
property_type_options = sorted(df["Property_Type"].dropna().astype(str).unique().tolist()) if "Property_Type" in df.columns else []
tenure_options = sorted(df["Tenure"].dropna().astype(str).unique().tolist()) if "Tenure" in df.columns else []

menu = st.sidebar.radio("Navigation", ["Overview", "Dataset", "Train & Compare", "Predict"])


if menu == "Overview":
    st.subheader("What this application does")

    st.write(
        "This application is a decision-support prototype (MyValuation) that forecasts Malaysian property "
        "transaction prices by combining **transaction records** with **socio-economic indicators** such as "
        "income and population."
    )

    st.markdown(
        """
**Why it matters (from the project draft):**
- The market faces an affordability gap, including the “Ghost Town Paradox” where high-end supply appears in lower-income areas.
- Traditional valuation can miss local purchasing power; fusing census indicators with transactions helps reduce this “valuation blind spot”.
- The goal is an affordability-aware “Fair Price” baseline through an interactive Streamlit interface.
        """
    )

elif menu == "Dataset":
    st.subheader("Dataset")

    st.write("Preview:")
    st.dataframe(df.head(50), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Column types:")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)

    with c2:
        st.write("Missing values (top):")
        miss = df.isna().sum().sort_values(ascending=False)
        st.dataframe(miss[miss > 0].head(20), use_container_width=True)

    st.write("Numeric summary:")
    num_df = df.select_dtypes(exclude=["object"])
    if not num_df.empty:
        st.dataframe(num_df.describe().T, use_container_width=True)

elif menu == "Train & Compare":
    st.subheader("Train & Compare Models (cached)")

    with st.spinner("Training models (first run only)..."):
        trained, results, feature_cols = train_models(df)

    st.success("Done ✅ Models cached in memory.")

    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(res_df, use_container_width=True)

    st.plotly_chart(
        px.bar(res_df, x="Model", y="RMSE", title="RMSE Comparison (lower is better)"),
        use_container_width=True,
    )

else:
    st.subheader("Predict Transaction Price")

    with st.spinner("Loading trained models..."):
        trained, results, feature_cols = train_models(df)

    model_name = st.selectbox("Choose model", list(trained.keys()))
    model = trained[model_name]

    st.caption("Fill in inputs below. Property Type and Tenure are dropdowns from dataset values.")

    user_input = {}

    for col in feature_cols:
        if col == "Property_Type":
            user_input[col] = st.selectbox("Property Type", property_type_options) if property_type_options else st.text_input("Property_Type", "")

        elif col == "Tenure":
            user_input[col] = st.selectbox("Tenure", tenure_options) if tenure_options else st.text_input("Tenure", "")

        elif col in ["District", "state"]:
            user_input[col] = st.text_input(col, value="")

        elif col == "Month_Year":
            user_input[col] = st.text_input("Month_Year (YYYY-MM-DD)", value="2024-01-01")

        else:
            user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_cols)
        pred = model.predict(input_df)[0]
        st.success(f"✅ Predicted Transaction Price: RM {pred:,.2f}")
