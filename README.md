# AI-Powered Telecom Customer Churn Prediction & Retention System

An end-to-end machine learning and conversational AI solution designed to help telecommunications businesses identify customers at risk of churn, understand customer behaviour, and develop proactive retention strategies.

This project combines **predictive analytics, XGBoost machine learning, interactive business intelligence, and a voice-enabled AI assistant** to transform customer data into actionable insights. It aims to support data-driven decision-making, improve customer retention, and reduce potential revenue loss.

## Project Overview

Customer churn is a major challenge in the telecommunications industry. Identifying customers who are likely to discontinue their services allows businesses to take preventive action before losing valuable customers.

This solution analyses customer demographics, service subscriptions, tenure, billing information, and contract details to estimate churn probability and classify customer risk. It also provides an interactive analytics environment and a conversational AI assistant that helps users interpret predictions and explore possible retention strategies.

## Key Capabilities

- **Machine Learning-Based Churn Prediction:** Uses an XGBoost classification model to estimate the probability of customer churn.
- **Customer Risk Segmentation:** Categorises customers into risk levels to help prioritise retention efforts.
- **Interactive Business Intelligence Dashboard:** Presents churn trends, customer insights, service patterns, and analytical visualisations.
- **Individual Customer Assessment:** Allows users to enter customer details and generate personalised churn predictions.
- **AI-Powered Voice Assistant:** Enables users to ask questions through text or voice and receive conversational responses in written and spoken formats.
- **Retention Strategy Recommendations:** Provides actionable suggestions to help reduce customer churn.
- **Customer Follow-up Management:** Supports tracking of retention actions and follow-up activities.
- **Dataset Management:** Supports a permanent customer dataset for consistent analysis, with the option to upload updated data.

## Technology Stack

Python | XGBoost | Scikit-learn | Pandas | NumPy | Streamlit | Plotly | Speech Recognition | Text-to-Speech | Conversational AI

## Project Objective

The primary objective is to build an intelligent, data-driven customer retention system that goes beyond predicting churn by helping users understand potential churn risks and take informed preventive actions.

## Project structure

```text
telco_churn_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── telco_churn.csv
├── models/
│   ├── model.pkl
│   └── threshold.pkl
└── followups.db              # created automatically when the app runs
```

Keep the dataset and model files in the locations shown above. If your files have different names, update the corresponding paths in `app.py`.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- The Telco customer churn CSV used by the project
- The existing trained model file: `models/model.pkl`
- The decision threshold file: `models/threshold.pkl` (the app uses 0.40 as a fallback if configured that way)
- A working microphone for voice input
- Internet access for Google speech recognition used by `SpeechRecognition`

## Setup on Windows

Open Command Prompt or Anaconda Prompt and navigate to the project folder:

```bat
cd path\to\telco_churn_dashboard
```

Create and activate a virtual environment (recommended):

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the files in the expected folders:

- Copy your full Telco CSV to `data/telco_churn.csv`.
- Copy the existing XGBoost estimator to `models/model.pkl`.
- Copy the decision threshold to `models/threshold.pkl`.

Run the application:

```bat
streamlit run app.py
```

Streamlit will display a local address, usually `http://localhost:8501`. Open it in your browser.

## Dataset

The app expects the full Telco Customer Churn dataset, not a small preview. The individual-customer prediction and preprocessing logic uses these fields from the dataset:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `TechSupport`

The dataset also commonly includes customer and service attributes such as `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`, `PaymentMethod`, and `Churn`.

The permanent CSV is read locally by the app. To use a newer dataset, replace `data/telco_churn.csv` with a compatible CSV and restart or refresh the app. Make sure the replacement dataset is compatible with the preprocessing used for the saved model.

## Model and prediction notes

- The application loads the existing XGBoost estimator from `models/model.pkl`.
- The decision threshold is loaded from `models/threshold.pkl` when available; the project default is 0.40.
- The app reconstructs the preprocessing objects from the permanent dataset because the original scaler and one-hot encoder were not exported separately.
- For consistent results, use the same source dataset and compatible preprocessing setup that were used with the original model.
- A churn probability is a model estimate, not a guarantee that a customer will leave. It should support, not replace, human review.

**Do not retrain or overwrite the original model as part of normal app startup.** Model training and dashboard inference are separate tasks.

## Voice AI Assistant

The assistant supports both typed questions and recorded voice messages in the chat input. It can answer general questions about churn, customer risk, tenure, contracts, and retention ideas, and can discuss the latest customer prediction when one is available.

- Speech-to-text uses the `SpeechRecognition` package with Google's web speech recognition service, so voice transcription requires internet access.
- Text-to-speech uses `pyttsx3` locally. The available voice and audio playback depend on the operating system and installed speech engine.
- The assistant's conversational answers are rule-based in this project; an external LLM API key is not required.
- Browser autoplay policies may prevent audio from starting automatically. If that happens, use the audio player's Play control or the **Speak latest answer** button.

## Follow-up records and privacy

Follow-up entries are stored in a local SQLite database (`followups.db`) created by the app. Keep this file backed up if you need to preserve follow-up history. Do not commit real customer records, the SQLite database, private model artifacts, API keys, or other confidential information to a public repository.

For a public GitHub repository, use an anonymized or permitted sample dataset, or keep the real dataset outside GitHub and provide it locally in `data/telco_churn.csv`.

## Troubleshooting

- **Dataset not found:** confirm the file is named `telco_churn.csv` and is inside the `data` folder.
- **Model not found:** confirm `model.pkl` and `threshold.pkl` are inside `models`.
- **Voice transcription fails:** check your microphone, internet connection, and microphone permissions in your browser/operating system.
- **No spoken reply:** check that `pyttsx3` is installed and that your computer has a working speech engine. Use the audio player's Play button if autoplay is blocked.
- **Dependency errors:** activate the project's virtual environment and run `pip install -r requirements.txt` again.

## License and data

Add a license before redistributing this project. Ensure you have permission to share the dataset and any model artifacts included with the repository. Remove or anonymize personal/customer information before publishing.
