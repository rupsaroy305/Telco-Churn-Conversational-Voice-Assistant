# Telco Churn Conversational Voice Assistant

1. Copy your existing `model.pkl` and `threshold.pkl` into `models/`.
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Upload the exact original Telco CSV used in the notebook.

The app does not retrain XGBoost. It rebuilds preprocessing only.
The conversation is maintained in Streamlit session state and supports multiple typed or spoken turns, including simple follow-ups such as "why?".
Speech-to-text uses SpeechRecognition's Google web recognizer (internet required). Text-to-speech uses local pyttsx3. No OpenAI key or paid API is needed.

This is a small rule-based conversational demo, not a full LLM. Conversation history lasts for the current Streamlit session.
