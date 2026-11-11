# 📞 Telco Churn Conversational Voice Assistant
#### *Telco Customer Churn Prediction & Voice AI Assistant*

An interactive machine learning application that predicts customer churn in the telecommunications industry and provides a conversational Voice AI Assistant to help users understand churn-related insights.

## Project Overview

Customer churn is a major challenge for telecommunications companies because losing existing customers can negatively impact business revenue and growth.

This project uses **Machine Learning (XGBoost)** to predict whether a customer is likely to leave a telecom service. It also includes a **Streamlit-based Voice AI Assistant** that allows users to interact with the application through text and voice.

The goal is to help businesses identify customers at risk of churn and support better customer retention decisions.

## Key Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to churn using a trained XGBoost model.
- **Interactive Streamlit Dashboard:** Provides a user-friendly interface for entering customer details and viewing predictions.
- **Voice AI Assistant:** Enables users to ask churn-related questions using voice or text.
- **Speech-to-Text:** Converts spoken questions into text using SpeechRecognition.
- **Text-to-Speech:** Provides spoken responses using pyttsx3.
- **Conversational Interaction:** Supports multi-turn conversations and contextual follow-up questions.
- **Data Preprocessing:** Handles numerical scaling and categorical feature encoding.
- **Feature Engineering:** Creates additional features such as average monthly spend, tenure groups, and charge levels.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and preprocessing |
| Scikit-learn | Feature engineering and preprocessing |
| XGBoost | Customer churn prediction |
| Streamlit | Interactive web application |
| SpeechRecognition | Speech-to-text conversion |
| pyttsx3 | Text-to-speech conversion |
| Jupyter Notebook | Model development and experimentation |

##  Machine Learning Workflow

1. **Data Collection:** Used a Telco Customer Churn dataset containing customer demographics, services, account information, and churn status.
2. **Data Preprocessing:** Handled missing values, converted data types, and prepared numerical and categorical features.
3. **Feature Engineering:** Created additional features to capture customer spending patterns and tenure categories.
4. **Feature Transformation:** Applied StandardScaler to numerical features and OneHotEncoder to categorical features.
5. **Model Development:** Trained an XGBoost classifier to identify customers likely to churn.
6. **Threshold Optimization:** Used a classification threshold of 0.40 for churn prediction.
7. **Deployment:** Integrated the trained model into a Streamlit application.

##  Voice AI Assistant

The project includes a conversational assistant designed to make churn-related information easier to access.

Users can:
- Ask questions about customer churn.
- Interact through text or voice.
- Receive spoken responses.
- Ask follow-up questions during a conversation.
- Understand churn predictions and possible retention strategies.

The current assistant uses rule-based conversational responses, SpeechRecognition, and pyttsx3 for voice interaction.

##  Project Structure

```text
telco-churn-voice-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── model.pkl
│   └── threshold.pkl
│
└── .gitignore
```

**Note:** The trained model files and original dataset are not included in this repository by default. Add the required model files locally before running the prediction application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd telco-churn-voice-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the required files

Place the following files inside the `models/` directory:

- `model.pkl`
- `threshold.pkl`

Upload the original Telco Customer Churn CSV through the application when prompted.

Use the same dataset and preprocessing-compatible data used during model development.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Expected Outcome

The application provides:
- Customer churn predictions.
- Interactive customer-level analysis.
- Conversational access to churn-related information.
- Voice-based interaction for a more accessible user experience.

## Future Enhancements

- Integrate an LLM API for more intelligent and flexible conversations.
- Improve the assistant's ability to explain individual customer predictions.
- Add visual analytics and customer churn dashboards.
- Deploy the application on Streamlit Community Cloud.
- Develop personalized customer retention recommendations.
- Integrate additional speech recognition and voice synthesis options.

## Author

**Rupsa Roy\**
---
