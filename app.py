import os
import io
import tempfile
import joblib
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Telco Churn Conversational Assistant", page_icon="☎️", layout="centered")
st.title("☎️ Telco Churn Conversational Voice Assistant")
st.write("Predict churn risk and ask the Voice AI Assistant follow-up questions by typing or speaking for more customer-retention insights.")

MODEL_PATH = "models/model.pkl"
THRESHOLD_PATH = "models/threshold.pkl"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
CATEGORICAL = ["Contract", "InternetService", "TechSupport", "TenureGroup", "ChargeLevel"]

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH) if os.path.exists(THRESHOLD_PATH) else 0.40
    return model, float(threshold)

def clean_data(df):
    df = df.copy()
    if "TotalCharges" not in df.columns or "Churn" not in df.columns:
        raise ValueError("CSV must contain TotalCharges and Churn columns.")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().drop_duplicates().copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df

def add_training_features(X):
    X = X.copy()
    X["AvgMonthlySpend"] = X["TotalCharges"] / (X["tenure"] + 1)
    X["TenureGroup"] = pd.cut(X["tenure"], [0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
    X["ChargeLevel"] = pd.qcut(X["MonthlyCharges"], q=3,
        labels=["Low", "Medium", "High"], duplicates="drop")
    return X

def add_customer_features(customer, charge_edges):
    X = pd.DataFrame([customer])
    X["AvgMonthlySpend"] = X["TotalCharges"] / (X["tenure"] + 1)
    X["TenureGroup"] = pd.cut(X["tenure"], [0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
    X["ChargeLevel"] = pd.cut(X["MonthlyCharges"], bins=charge_edges,
        labels=["Low", "Medium", "High"], include_lowest=True)
    return X

@st.cache_resource
def build_preprocessor(csv_bytes):
    df = clean_data(pd.read_csv(io.BytesIO(csv_bytes)))
    X, y = df.drop(columns=["Churn"]), df["Churn"].map({"No": 0, "Yes": 1})
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2,
        random_state=42, stratify=y)
    edges = pd.qcut(X_train["MonthlyCharges"], q=3, retbins=True,
        duplicates="drop")[1]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL)
    ])
    preprocessor.fit(add_training_features(X_train))
    return preprocessor, edges

def create_answer(question):
    q = question.lower().strip()
    prediction = st.session_state.get("last_prediction")
    previous_topic = st.session_state.get("last_topic", "")

    # Resolve short follow-up questions using conversation context.
    if q in ["why?", "why", "explain", "explain it", "tell me more", "how?", "what does that mean?"]:
        if prediction and previous_topic == "prediction":
            if prediction["probability"] >= 0.40:
                return (f"This customer has a churn probability of {prediction['probability']:.1%}. "
                        f"The prediction is {prediction['prediction']}. A higher probability means the model sees a greater likelihood of leaving. "
                        f"The customer's contract is {prediction['Contract']}, tenure is {prediction['tenure']} months, "
                        f"and monthly charge is {prediction['MonthlyCharges']:.2f}. These are useful factors to review, "
                        "but this simple app cannot explain individual feature contributions.")
            return (f"The model estimates a churn probability of {prediction['probability']:.1%}, "
                    f"so the result is {prediction['prediction']}. This is an estimate, not a guarantee.")
        return "Could you tell me which part you want me to explain?"

    if any(x in q for x in ["my prediction", "this customer", "customer result", "risk", "will they churn", "churn probability", "what is the result"]):
        if prediction:
            st.session_state["last_topic"] = "prediction"
            return (f"The churn probability is {prediction['probability']:.1%}. "
                    f"The model result is: {prediction['prediction']}. "
                    "This is a model estimate, not a guarantee.")
        return "Please make a customer churn prediction first. Then I can discuss the result."

    if any(x in q for x in ["what is churn", "meaning of churn", "define churn"]):
        st.session_state["last_topic"] = "churn"
        return "Customer churn means a customer stops using a company's service or cancels their subscription. In telecom, it usually means disconnecting or switching providers."

    if any(x in q for x in ["reduce", "prevent", "stop churn", "retain customers"]):
        st.session_state["last_topic"] = "retention"
        return "A telecom company can reduce churn by improving customer support, resolving network issues quickly, offering suitable plans, rewarding loyalty, and contacting customers who show signs of leaving. Would you like examples of retention offers?"

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
            "and your latest prediction. You can also ask follow-up questions such as 'Why?' or 'How can we reduce it?'.")

def speak_text(text):
    import pyttsx3
    engine = pyttsx3.init()
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        engine.stop()
        if os.path.exists(wav_path):
            os.remove(wav_path)

def handle_message(question, source="text"):
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    answer = create_answer(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_answer = answer
    st.session_state.last_source = source

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if not os.path.exists(MODEL_PATH):
    st.error("Place your existing model.pkl inside the models folder.")
    st.stop()
try:
    model, threshold = load_model()
except Exception as e:
    st.error(f"Could not load saved model: {e}")
    st.stop()

csv_file = st.sidebar.file_uploader("Upload original Telco CSV", type=["csv"])
if csv_file is None:
    st.info("Upload the same Telco CSV used in your notebook. The app rebuilds preprocessing only; it does not retrain XGBoost.")
    st.stop()
try:
    preprocessor, charge_edges = build_preprocessor(csv_file.getvalue())
except Exception as e:
    st.error(f"Could not prepare CSV: {e}")
    st.stop()

st.subheader("1. Predict a customer")
with st.form("customer_form"):
    tenure = st.number_input("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0)
    total = st.number_input("Total charges", min_value=0.0, value=840.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    predict = st.form_submit_button("Predict churn")
if predict:
    customer = {"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
                "Contract": contract, "InternetService": internet, "TechSupport": support}
    try:
        matrix = preprocessor.transform(add_customer_features(customer, charge_edges))
        probability = float(model.predict_proba(matrix)[0][1])
        result = "Likely to Churn" if probability >= threshold else "Not Likely to Churn"
        st.metric("Churn probability", f"{probability:.1%}")
        st.warning(result) if probability >= threshold else st.success(result)
        st.session_state.last_prediction = {"probability": probability, "prediction": result, **customer}
        st.session_state.last_topic = "prediction"
    except Exception as e:
        st.error(f"Prediction failed. Check that this is the same CSV used in your notebook. Details: {e}")

st.divider()
st.subheader("2. Have a conversation")
st.caption("Conversation history is kept while this Streamlit session is open. You can type or record each turn.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

typed_question = st.chat_input("Type a question or follow-up...")
if typed_question:
    handle_message(typed_question, "text")
    st.rerun()

st.markdown("**Or ask by voice**")
audio = st.audio_input("Record your next message")
if audio is not None:
    if st.button("Send voice message"):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio.getvalue())) as source:
                audio_data = recognizer.record(source)
            question = recognizer.recognize_google(audio_data)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.last_source = "voice"
            answer = create_answer(question)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_answer = answer
            st.rerun()
        except sr.UnknownValueError:
            st.warning("I couldn't understand that recording. Please try again.")
        except Exception as e:
            st.error(f"Voice recognition error: {e}")

if st.session_state.get("last_answer"):
    if st.button("🔊 Speak latest answer"):
        try:
            audio_bytes = speak_text(st.session_state.last_answer)
            st.audio(audio_bytes, format="audio/wav")
        except Exception as e:
            st.error(f"Local text-to-speech error: {e}")

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_topic = ""
        st.session_state.last_answer = ""
        st.rerun()

st.caption("Free conversational demo: rule-based responses with session conversation context. Speech recognition requires internet; speech output is generated locally. No OpenAI key or API credits are needed.")
