import joblib
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
from pydantic import BaseModel
from src.monitoring import REQUEST_COUNT, REQUEST_LATENCY, logger

app = FastAPI()

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

class ResumeText(BaseModel):
    text: str

@app.post("/predict/")
def predict(request: ResumeText):
    # Convert text into TF-IDF features
    vec = vectorizer.transform([request.text])
    
    # Predict
    prediction = model.predict(vec)[0]
    
    return {"predicted_category": prediction}


# Middleware for metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        response = await call_next(request)
    logger.info(f"Request: {request.url.path} - Status: {response.status_code}")
    return response

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

# Example route
@app.get("/")
def root():
    logger.info("Root endpoint hit")
    return {"message": "Resume Classification API is running!"}

# Mount metrics endpoint
app.mount("/metrics", make_asgi_app())

