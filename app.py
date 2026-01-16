# app.py
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import clean_columns, TARGET_COL

st.set_page_config(page_title="Property Price Forecasting", layout="wide")

DATA_PATH = "data/sample.csv"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Missing data/sample.csv. Add it to your repo.")
    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)
    return df


@st.cache_resource
def train_models(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Keep Random Forest SMALL so training is fast in cloud
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
        "SVM (LinearSVR)": LinearSVR(max_iter=10000, random_state=42),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=42
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

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        trained[name] = pipe
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return trained, results, list(X.columns)


# ---------- UI ----------
st.title("🏠 Property Price Forecasting (Train-in-Cloud)")

df = load_data()

# Dropdown options (built from dataset)
property_type_options = sorted(df["Property_Type"].dropna().unique().tolist()) if "Property_Type" in df.columns else []
tenure_options = sorted(df["Tenure"].dropna().unique().tolist()) if "Tenure" in df.columns else []

menu = st.sidebar.radio("Navigation", ["Overview", "EDA", "Train & Compare", "Predict"])


if menu == "Overview":
    st.write(
        "Regression task: predict **Transaction Price** based on income, population, and property attributes."
    )
    st.info(f"Loaded sample: {df.shape[0]:,} rows × {df.shape[1]} columns")

elif menu == "EDA":
    st.subheader("EDA (Sample)")
    st.dataframe(df.head(30), use_container_width=True)

    if "income" in df.columns:
        st.plotly_chart(
            px.scatter(df, x="income", y=TARGET_COL, title="Income vs Transaction Price"),
            use_container_width=True,
        )
    if "population" in df.columns:
        st.plotly_chart(
            px.scatter(df, x="population", y=TARGET_COL, title="Population vs Transaction Price"),
            use_container_width=True,
        )

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

    st.caption(
        "Enter values. Property Type and Tenure use dropdowns (from dataset). Other categorical fields are text for now."
    )

    user_input = {}
    for col in feature_cols:
        if col == "Property_Type":
            if property_type_options:
                user_input[col] = st.selectbox("Property Type", property_type_options)
            else:
                user_input[col] = st.text_input("Property_Type", value="")

        elif col == "Tenure":
            if tenure_options:
                user_input[col] = st.selectbox("Tenure", tenure_options)
            else:
                user_input[col] = st.text_input("Tenure", value="")

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
