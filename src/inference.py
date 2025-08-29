import joblib
from src.preprocessing import clean_text

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_category(resume_text: str):
    text = clean_text(resume_text)
    vectorized = vectorizer.transform([text])
    return model.predict(vectorized)[0]
