import io
import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------
# PAGE CONFIG + THEME
# ------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn AI | Retention Center",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #14243b; }
    [data-testid="stSidebar"] * { color: #f4f7fb; }
    .hero {
        padding: 1.35rem 1.55rem; border-radius: 18px;
        background: linear-gradient(115deg,#14243b,#1c4961);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 1.8rem; }
    .hero p { color: #d6e5ef; margin: .35rem 0 0 0; }
    .metric-card {
        background: white; border: 1px solid #e7ebf2; border-radius: 15px;
        padding: 17px 18px; box-shadow: 0 3px 12px rgba(20,36,59,.04);
    }
    .metric-label { color: #64748b; font-size: .82rem; font-weight: 600; }
    .metric-value { color: #14243b; font-size: 1.75rem; font-weight: 700; margin-top: 5px; }
    .metric-note { color: #8491a3; font-size: .75rem; margin-top: 2px; }
    .section-title { color: #14243b; font-size: 1.15rem; font-weight: 700; margin: .4rem 0 .7rem; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .stButton button { border-radius: 9px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
MODEL_PATH = "models/model.pkl"
THRESHOLD_PATH = "models/threshold.pkl"
DB_PATH = "followups.db"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "TechSupport", "TenureGroup", "ChargeLevel"
]
REQUIRED_MODEL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ------------------------------------------------------------
# DATA + MODEL HELPERS
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def normalize_churn(series: pd.Series) -> pd.Series:
    """Convert common Yes/No churn labels to 1/0."""
    return (
        series.astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    )


def engineer_features(df: pd.DataFrame, charge_edges=None) -> pd.DataFrame:
    data = df.copy()

    required = ["tenure", "MonthlyCharges", "TotalCharges"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["MonthlyCharges"] * data["tenure"]
    )
    data["tenure"] = data["tenure"].fillna(data["tenure"].median())
    data["MonthlyCharges"] = data["MonthlyCharges"].fillna(
        data["MonthlyCharges"].median()
    )
    data["AvgMonthlySpend"] = data["TotalCharges"] / (data["tenure"] + 1)

    data["TenureGroup"] = pd.cut(
        data["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )
    data["TenureGroup"] = data["TenureGroup"].astype("object").fillna("0-1yr")

    if charge_edges is not None:
        # Use training-set quantile edges for both batch and single-customer scoring.
        data["ChargeLevel"] = pd.cut(
            data["MonthlyCharges"], bins=charge_edges,
            labels=["Low", "Medium", "High"][:len(charge_edges) - 1],
            include_lowest=True,
        )
    else:
        data["ChargeLevel"] = pd.qcut(
            data["MonthlyCharges"], q=3,
            labels=["Low", "Medium", "High"], duplicates="drop",
        )
    data["ChargeLevel"] = data["ChargeLevel"].astype("object").fillna("Medium")

    for col in ["Contract", "InternetService", "TechSupport"]:
        if col not in data.columns:
            raise ValueError(f"CSV is missing required column: {col}")
        data[col] = data[col].astype("object").fillna("Unknown")
    return data


def add_customer_features(customer: dict, charge_edges) -> pd.DataFrame:
    """Build the exact feature columns used by the fitted preprocessor."""
    return engineer_features(pd.DataFrame([customer]), charge_edges)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


@st.cache_resource(show_spinner="Loading model and rebuilding preprocessing…")
def load_model_and_preprocessor(csv_bytes: bytes, model_mtime: float, threshold_mtime: float):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Put your existing model.pkl in the models folder."
        )

    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH)) if os.path.exists(THRESHOLD_PATH) else 0.40

    raw = pd.read_csv(io.BytesIO(csv_bytes))
    if "Churn" not in raw.columns:
        raise ValueError("The uploaded training CSV must contain the Churn column.")
    y = normalize_churn(raw["Churn"])
    valid = y.notna()
    raw = raw.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    featured_initial = engineer_features(raw)
    X = featured_initial[REQUIRED_MODEL_COLUMNS]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Rebuild the same training-derived monthly-charge quantile boundaries.
    charge_edges = pd.qcut(
        X_train["MonthlyCharges"], q=3, retbins=True, duplicates="drop"
    )[1]
    if len(charge_edges) < 4:
        raise ValueError(
            "MonthlyCharges has too many duplicate values to rebuild three ChargeLevel bins."
        )

    X_train_features = engineer_features(X_train, charge_edges)
    preprocessor = make_preprocessor()
    preprocessor.fit(X_train_features[REQUIRED_MODEL_COLUMNS])

    featured = engineer_features(raw, charge_edges)
    transformed = preprocessor.transform(featured[REQUIRED_MODEL_COLUMNS])
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and transformed.shape[1] != expected:
        raise ValueError(
            f"Preprocessing produced {transformed.shape[1]} features, but model.pkl expects {expected}. "
            "The CSV or preprocessing definitions may differ from the original notebook."
        )

    probabilities = model.predict_proba(transformed)
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    churn_probability = probabilities[:, positive_index]

    results = featured.copy()
    results["ChurnActual"] = y.values
    results["ChurnProbability"] = churn_probability
    results["PredictedChurn"] = (churn_probability >= threshold).astype(int)
    results["RiskLevel"] = pd.cut(
        churn_probability, bins=[-0.000001, 0.30, 0.60, 1.000001],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return model, threshold, results, preprocessor, charge_edges


def get_recommendation(row: pd.Series) -> str:
    actions = []
    contract = str(row.get("Contract", ""))
    internet = str(row.get("InternetService", ""))
    support = str(row.get("TechSupport", ""))
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    tenure = float(row.get("tenure", 0) or 0)

    if contract.lower() == "month-to-month":
        actions.append("Offer an annual-contract discount")
    if monthly >= 75:
        actions.append("Review plan pricing and offer a suitable lower-cost plan")
    if tenure <= 12:
        actions.append("Provide an onboarding check-in or new-customer benefit")
    if internet.lower() != "no" and support.lower() == "no":
        actions.append("Offer a technical-support add-on or support consultation")
    if not actions:
        actions.append("Schedule a satisfaction check-in and review service experience")
    return "; ".join(actions)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS followups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                risk_level TEXT,
                churn_probability REAL,
                action TEXT,
                contact_date TEXT,
                followup_date TEXT,
                status TEXT DEFAULT 'Pending',
                notes TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()


def save_followup(record: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO followups
            (customer_id, risk_level, churn_probability, action, contact_date,
             followup_date, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["customer_id"], record["risk_level"],
                record["churn_probability"], record["action"],
                str(record["contact_date"]), str(record["followup_date"]),
                record["status"], record["notes"], datetime.now().isoformat(timespec="seconds")
            ),
        )
        conn.commit()


def get_followups() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM followups ORDER BY id DESC", conn)


def update_followup_status(record_id: int, status: str, notes: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE followups SET status=?, notes=? WHERE id=?",
            (status, notes, record_id),
        )
        conn.commit()


def create_answer(question: str) -> str:
    """Rule-based conversational replies with context from the latest prediction."""
    q = question.lower().strip()
    prediction = st.session_state.get("last_prediction")
    previous_topic = st.session_state.get("last_topic", "")

    if q in ["why?", "why", "explain", "explain it", "tell me more", "how?", "what does that mean?"]:
        if prediction and previous_topic == "prediction":
            return (
                f"This customer has a churn probability of {prediction['probability']:.1%}. "
                f"The prediction is {prediction['prediction']}. A higher probability means "
                "the model sees a greater likelihood of leaving. The customer's contract is "
                f"{prediction['Contract']}, tenure is {prediction['tenure']} months, and "
                f"monthly charge is {prediction['MonthlyCharges']:.2f}. These are useful "
                "factors to review, but this app cannot explain individual feature contributions."
            )
        return "Could you tell me which part you want me to explain?"

    if any(x in q for x in ["my prediction", "this customer", "customer result", "risk", "will they churn", "churn probability", "what is the result"]):
        if prediction:
            st.session_state["last_topic"] = "prediction"
            return (
                f"The churn probability is {prediction['probability']:.1%}. "
                f"The model result is: {prediction['prediction']}. This is a model estimate, not a guarantee."
            )
        return "Please make a customer churn prediction first. Then I can discuss the result."

    if any(x in q for x in ["what is churn", "meaning of churn", "define churn"]):
        st.session_state["last_topic"] = "churn"
        return "Customer churn means a customer stops using a company's service or cancels their subscription. In telecom, it usually means disconnecting or switching providers."
    if any(x in q for x in ["reduce", "prevent", "stop churn", "retain customers"]):
        st.session_state["last_topic"] = "retention"
        return "A telecom company can reduce churn by improving customer support, resolving network issues quickly, offering suitable plans, rewarding loyalty, and contacting customers who show signs of leaving."
    if any(x in q for x in ["why", "reason", "cause"]):
        st.session_state["last_topic"] = "reasons"
        return "Common reasons for telecom churn include high monthly charges, month-to-month contracts, poor support, service interruptions, and better competitor offers."
    if "tenure" in q:
        st.session_state["last_topic"] = "tenure"
        return "Tenure is the number of months a customer has stayed with the telecom company. Customers with shorter tenure can be at higher churn risk because they have had less time to build loyalty."
    if "contract" in q:
        st.session_state["last_topic"] = "contract"
        return "Month-to-month customers can leave more easily than customers on one-year or two-year contracts. Longer contracts are often associated with lower churn, but service quality still matters."
    if any(x in q for x in ["hello", "hi", "hey"]):
        st.session_state["last_topic"] = "greeting"
        return "Hello! I'm your telecom churn assistant. You can ask me about churn, prediction results, customer retention, tenure, or contracts."
    if "thank" in q:
        return "You're welcome! Feel free to ask a follow-up question."
    return ("I can help with churn definitions, churn reasons, customer retention, tenure, contracts, "
            "and your latest prediction. You can ask follow-up questions such as 'Why?' or 'How can we reduce it?'.")


def speak_text(text: str):
    import pyttsx3
    engine = pyttsx3.init()
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as audio_file:
            return audio_file.read()
    finally:
        engine.stop()
        if os.path.exists(wav_path):
            os.remove(wav_path)


def transcribe_audio(audio_file) -> str:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        audio_bytes = audio_file.getvalue()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception as exc:
        raise RuntimeError(
            "Could not transcribe audio. Check your internet connection and microphone "
            f"recording format. Details: {exc}"
        )


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
init_db()
with st.sidebar:
    st.markdown("## 📞 TELCO INSIGHTS")
    st.caption("Customer intelligence & retention")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Predict a Customer", "Churn Analytics", "Customer Explorer",
         "Retention Center", "Follow-up Tracker", "Voice AI Assistant"],
        label_visibility="visible",
        key="page"
    )
    st.divider()
    st.markdown("### Upload your dataset")
    uploaded_csv = st.file_uploader("Telco CSV", type=["csv"], key="telco_csv")
    st.caption("Upload your dataset")
    st.divider()
    st.caption("Decision threshold: loaded from threshold.pkl (default 0.40)")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Telco Churn AI</h1>
      <p>Understand customer risk, prioritize retention, and manage follow-ups.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_csv is None:
    st.info(
        "Upload your original Telco Customer Churn CSV in the sidebar to load "
        "customer insights and predictions."
    )

    st.stop()

csv_bytes = uploaded_csv.getvalue()
try:
    model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    threshold_mtime = os.path.getmtime(THRESHOLD_PATH) if os.path.exists(THRESHOLD_PATH) else 0
    model, threshold, customers, preprocessor, charge_edges = load_model_and_preprocessor(
        csv_bytes, model_mtime, threshold_mtime
    )
except Exception as exc:
    st.error(f"Could not load predictions: {exc}")
    st.stop()

if "customerID" not in customers.columns:
    customers["customerID"] = [f"Customer-{i+1}" for i in range(len(customers))]

# Common filters
with st.sidebar:
    st.markdown("### Filters")
    risk_filter = st.multiselect(
        "Risk level", ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    contract_options = sorted(customers["Contract"].dropna().astype(str).unique())
    contract_filter = st.multiselect("Contract", contract_options, default=contract_options)

filtered = customers[
    customers["RiskLevel"].isin(risk_filter)
    & customers["Contract"].astype(str).isin(contract_filter)
].copy()

total_customers = len(customers)
actual_churn_rate = customers["ChurnActual"].mean() if total_customers else 0
predicted_churn_count = int(customers["PredictedChurn"].sum())
high_risk_count = int((customers["RiskLevel"] == "High").sum())
avg_monthly = customers["MonthlyCharges"].mean() if total_customers else 0


def metric(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
if page == "Dashboard":
    st.markdown('<div class="section-title">Business overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        metric("Total customers", f"{total_customers:,}", "Customers in uploaded dataset")
    with cols[1]:
        metric("Actual churn rate", f"{actual_churn_rate:.1%}", "Based on the Churn column")
    with cols[2]:
        metric("Predicted churn", f"{predicted_churn_count:,}", f"Threshold: {threshold:.2f}")
    with cols[3]:
        metric("High-risk customers", f"{high_risk_count:,}", "Model probability above 60%")

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Actual churn distribution")
        churn_counts = customers["ChurnActual"].map({0: "Stayed", 1: "Churned"}).value_counts().reset_index()
        churn_counts.columns = ["Customer status", "Customers"]
        fig = px.pie(
            churn_counts, names="Customer status", values="Customers",
            hole=.62, color="Customer status",
            color_discrete_map={"Stayed": "#20a486", "Churned": "#e56b6f"},
        )
        fig.update_layout(margin=dict(t=15, b=10, l=5, r=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Predicted risk segments")
        risk_counts = customers["RiskLevel"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        ).rename_axis("Risk").reset_index(name="Customers")
        fig = px.bar(
            risk_counts, x="Risk", y="Customers", color="Risk",
            color_discrete_map={"High": "#e56b6f", "Medium": "#e5a84b", "Low": "#20a486"},
        )
        fig.update_layout(showlegend=False, margin=dict(t=15, b=10, l=5, r=5))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Customers to prioritize")
    priority = customers[customers["RiskLevel"] == "High"].sort_values(
        "ChurnProbability", ascending=False
    ).head(10).copy()
    priority["ChurnProbability"] = (priority["ChurnProbability"] * 100).round(1).astype(str) + "%"
    priority["Suggested action"] = priority.apply(get_recommendation, axis=1)
    display_cols = ["customerID", "ChurnProbability", "Contract", "tenure", "MonthlyCharges", "Suggested action"]
    st.dataframe(priority[display_cols], use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# PREDICT A CUSTOMER
# ------------------------------------------------------------
elif page == "Predict a Customer":
    st.markdown("### 1. Predict a customer")
    st.caption("Enter a customer's service details to estimate churn risk using your existing XGBoost model.")
    with st.form("single_customer_prediction"):
        p1, p2, p3 = st.columns(3)
        with p1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
        with p2:
            total = st.number_input("Total charges", min_value=0.0, value=840.0, step=10.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with p3:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        predict_clicked = st.form_submit_button("🔍 Predict churn", type="primary", use_container_width=True)

    if predict_clicked:
        customer = {
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
            "Contract": contract, "InternetService": internet, "TechSupport": support,
        }
        try:
            customer_features = add_customer_features(customer, charge_edges)
            matrix = preprocessor.transform(customer_features[REQUIRED_MODEL_COLUMNS])
            probability_array = model.predict_proba(matrix)[0]
            model_classes = list(getattr(model, "classes_", [0, 1]))
            positive_idx = model_classes.index(1) if 1 in model_classes else 1
            probability = float(probability_array[positive_idx])
            result = "Likely to Churn" if probability >= threshold else "Not Likely to Churn"
            st.session_state.last_prediction = {
                "probability": probability, "prediction": result, **customer
            }
            st.session_state.last_topic = "prediction"
            left, right = st.columns([1, 1.4])
            with left:
                st.metric("Churn probability", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
            with right:
                if probability >= threshold:
                    st.error(f"⚠️ {result}")
                else:
                    st.success(f"✅ {result}")
                st.caption(f"Decision threshold: {threshold:.2f}")
                def go_to_voice_assistant():
                    st.session_state["page"] = "Voice AI Assistant"
                
                st.success("Prediction completed! You can now discuss this customer with the Voice AI Assistant.")

                
                st.button(
                    "🎙️ Go to Voice AI Assistant",
                    use_container_width=True,
                    on_click=go_to_voice_assistant
                )

        except Exception as exc:
            st.error(f"Prediction failed. Check that the uploaded CSV matches the notebook preprocessing. Details: {exc}")

# ------------------------------------------------------------
# CHURN ANALYTICS
# ------------------------------------------------------------
elif page == "Churn Analytics":
    st.markdown("### Explore churn patterns")
    c1, c2 = st.columns(2)
    with c1:
        contract_churn = customers.groupby("Contract", observed=False)["ChurnActual"].mean().mul(100).reset_index()
        contract_churn.columns = ["Contract", "Churn rate (%)"]
        fig = px.bar(contract_churn, x="Contract", y="Churn rate (%)", color="Contract")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tenure_data = customers.copy()
        tenure_data["Tenure band"] = pd.cut(
            tenure_data["tenure"], bins=[-1, 12, 24, 48, 72],
            labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
        )
        tenure_churn = tenure_data.groupby("Tenure band", observed=False)["ChurnActual"].mean().mul(100).reset_index()
        tenure_churn.columns = ["Tenure band", "Churn rate (%)"]
        fig = px.bar(tenure_churn, x="Tenure band", y="Churn rate (%)", color="Tenure band")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        internet_churn = customers.groupby("InternetService", observed=False)["ChurnActual"].mean().mul(100).reset_index()
        internet_churn.columns = ["Internet service", "Churn rate (%)"]
        fig = px.bar(internet_churn, x="Internet service", y="Churn rate (%)", color="Internet service")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(
            customers, x="MonthlyCharges", color=customers["ChurnActual"].map({0: "Stayed", 1: "Churned"}),
            barmode="overlay", opacity=.7,
            labels={"color": "Actual status"},
        )
        fig.update_layout(legend_title_text="Actual status")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Churn probability distribution")
    fig = px.histogram(customers, x="ChurnProbability", color="RiskLevel",
                       color_discrete_map={"High": "#e56b6f", "Medium": "#e5a84b", "Low": "#20a486"})
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# CUSTOMER EXPLORER
# ------------------------------------------------------------
elif page == "Customer Explorer":
    st.markdown("### Search and inspect customers")
    search = st.text_input("Search customer ID", placeholder="Enter a customer ID…")
    view = filtered.copy()
    if search:
        view = view[view["customerID"].astype(str).str.contains(search, case=False, na=False)]
    view = view.sort_values("ChurnProbability", ascending=False)
    st.dataframe(
        view[["customerID", "RiskLevel", "ChurnProbability", "Contract",
              "tenure", "MonthlyCharges", "InternetService", "TechSupport"]],
        use_container_width=True, hide_index=True,
        column_config={
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0, max_value=1, format="%.1%%"
            )
        },
    )
    ids = view["customerID"].astype(str).tolist()
    if ids:
        selected_id = st.selectbox("Select a customer for details", ids)
        row = view[view["customerID"].astype(str) == selected_id].iloc[0]
        a, b, c = st.columns(3)
        with a:
            metric("Risk category", str(row["RiskLevel"]), "Based on model probability")
        with b:
            metric("Churn probability", f"{row['ChurnProbability']:.1%}", "Estimated by XGBoost")
        with c:
            metric("Monthly charges", f"${row['MonthlyCharges']:.2f}", "Customer monthly bill")
        st.markdown("**Recommended retention action**")
        st.info(get_recommendation(row))
        if st.button("Create follow-up for this customer", type="primary"):
            st.session_state["followup_customer"] = selected_id
            st.session_state["followup_action"] = get_recommendation(row)
            st.success("Customer selected. Open Follow-up Tracker to save the record.")
    else:
        st.warning("No customers match the current filters/search.")

# ------------------------------------------------------------
# RETENTION CENTER
# ------------------------------------------------------------
elif page == "Retention Center":
    st.markdown("### Retention action center")
    st.caption("Prioritize high-risk customers and use these suggestions as decision support.")
    high_risk = customers[customers["RiskLevel"].isin(["High", "Medium"])].copy()
    high_risk["Suggested action"] = high_risk.apply(get_recommendation, axis=1)
    high_risk = high_risk.sort_values("ChurnProbability", ascending=False)
    st.dataframe(
        high_risk[["customerID", "RiskLevel", "ChurnProbability", "Contract",
                   "tenure", "MonthlyCharges", "Suggested action"]],
        use_container_width=True, hide_index=True,
        column_config={
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0, max_value=1, format="%.1%%"
            )
        },
    )
    selected = st.selectbox(
        "Choose customer to create a retention follow-up",
        high_risk["customerID"].astype(str).tolist() if len(high_risk) else [],
    )
    if selected:
        row = high_risk[high_risk["customerID"].astype(str) == selected].iloc[0]
        st.info(f"Suggested action: {get_recommendation(row)}")
        if st.button("Use this recommendation", type="primary"):
            st.session_state["followup_customer"] = selected
            st.session_state["followup_action"] = get_recommendation(row)
            st.success("Recommendation selected. Go to Follow-up Tracker to save it.")

# ------------------------------------------------------------
# FOLLOW-UP TRACKER
# ------------------------------------------------------------
elif page == "Follow-up Tracker":
    st.markdown("### Retention follow-up records")
    followups = get_followups()
    if not followups.empty:
        f1, f2, f3 = st.columns(3)
        with f1:
            metric("Total records", f"{len(followups):,}")
        with f2:
            metric("Pending", int((followups["status"] == "Pending").sum()))
        with f3:
            metric("Retained", int((followups["status"] == "Retained").sum()))
    else:
        st.info("No follow-up records yet. Create one from Customer Explorer or Retention Center.")

    with st.expander("＋ Create a follow-up record", expanded=not bool(followups.empty)):
        default_id = st.session_state.get("followup_customer", "")
        default_action = st.session_state.get("followup_action", "")
        with st.form("new_followup_form", clear_on_submit=True):
            customer_id = st.text_input("Customer ID", value=default_id)
            risk_level = st.selectbox("Risk level", ["High", "Medium", "Low"])
            prob = st.slider("Churn probability", 0.0, 1.0, 0.70, 0.01)
            action = st.text_area("Retention action", value=default_action)
            contact_date = st.date_input("Contact date", value=date.today())
            followup_date = st.date_input("Follow-up date", value=date.today() + timedelta(days=3))
            status = st.selectbox("Status", ["Pending", "Contacted", "Retained", "Lost"])
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save follow-up", type="primary")
            if submitted:
                if not customer_id.strip():
                    st.error("Please enter a customer ID.")
                else:
                    save_followup({
                        "customer_id": customer_id.strip(),
                        "risk_level": risk_level,
                        "churn_probability": prob,
                        "action": action,
                        "contact_date": contact_date,
                        "followup_date": followup_date,
                        "status": status,
                        "notes": notes,
                    })
                    st.success("Follow-up record saved.")
                    st.rerun()

    followups = get_followups()
    if not followups.empty:
        st.markdown("### Update an existing record")
        record_ids = followups["id"].astype(int).tolist()
        record_id = st.selectbox("Record ID", record_ids)
        current = followups[followups["id"] == record_id].iloc[0]
        new_status = st.selectbox(
            "Update status", ["Pending", "Contacted", "Retained", "Lost"],
            index=["Pending", "Contacted", "Retained", "Lost"].index(current["status"])
        )
        new_notes = st.text_area("Update notes", value=str(current["notes"] or ""))
        if st.button("Update record"):
            update_followup_status(int(record_id), new_status, new_notes)
            st.success("Record updated.")
            st.rerun()

        st.markdown("### All follow-up records")
        st.dataframe(followups, use_container_width=True, hide_index=True)
        st.download_button(
            "Download follow-ups as CSV",
            data=followups.to_csv(index=False).encode("utf-8"),
            file_name="telco_followup_records.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# VOICE AI ASSISTANT
# ------------------------------------------------------------
elif page == "Voice AI Assistant":

    st.markdown("### 💬 Churn Conversational Voice Assistant")
    st.caption(
        "Type your question or use the microphone to speak. "
        "The assistant will reply in text and voice."
    )

    # Initialize session variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_topic" not in st.session_state:
        st.session_state.last_topic = ""

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""

    if "auto_play_latest" not in st.session_state:
        st.session_state.auto_play_latest = False

    # Show latest customer prediction
    prediction = st.session_state.get("last_prediction")

    if prediction:
        st.info(
            f"Latest customer prediction: **{prediction['prediction']}** · "
            f"Churn probability: **{prediction['probability']:.1%}**"
        )
    else:
        st.caption(
            "Tip: Make a prediction in 'Predict a Customer' "
            "first to ask questions about that result."
        )

    # Display previous conversation
    for i, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Display audio for assistant replies
            if message["role"] == "assistant" and message.get("audio"):

                is_latest = (
                    i == len(st.session_state.messages) - 1
                    and st.session_state.auto_play_latest
                )

                st.audio(
                    message["audio"],
                    format="audio/wav",
                    autoplay=is_latest
                )

                if is_latest:
                    st.session_state.auto_play_latest = False

    # ChatGPT-style input: text OR microphone
    prompt = st.chat_input(
        "Type a question or record your voice...",
        accept_audio=True,
        audio_sample_rate=16000,
        key="voice_chat_input"
    )

    if prompt:

        try:
            user_question = ""

            # Get typed message
            if prompt.text and prompt.text.strip():
                user_question = prompt.text.strip()

            # Or transcribe voice message
            elif prompt.audio:
                with st.spinner("Listening to your voice..."):
                    user_question = transcribe_audio(prompt.audio)

            if user_question:

                # Save user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_question
                })

                with st.spinner("AI Assistant is thinking..."):

                    # Generate written answer
                    answer = create_answer(user_question)

                    # Generate spoken answer automatically
                    audio_bytes = None

                    try:
                        audio_bytes = speak_text(answer)
                    except Exception as audio_error:
                        st.warning(
                            f"Could not generate voice reply: {audio_error}"
                        )

                # Save assistant reply and audio
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "audio": audio_bytes
                })

                st.session_state.last_answer = answer
                st.session_state.auto_play_latest = bool(audio_bytes)

                st.rerun()

        except Exception as exc:
            st.error(f"Message processing error: {exc}")

    # Replay latest assistant voice response
    latest_audio = next(
        (
            message.get("audio")
            for message in reversed(st.session_state.messages)
            if message["role"] == "assistant" and message.get("audio")
        ),
        None
    )

    if latest_audio:
        if st.button(
            "🔊 Speak latest answer",
            use_container_width=True
        ):
            st.audio(
                latest_audio,
                format="audio/wav",
                autoplay=True
            )

    # Clear conversation
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_topic = ""
            st.session_state.last_answer = ""
            st.session_state.auto_play_latest = False
            st.rerun()

    st.caption(
        "Voice recognition requires an internet connection. "
    )
