import joblib
import logging
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from src.monitoring import REQUEST_COUNT, REQUEST_LATENCY, logger

app = FastAPI()

# Load models
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

class ResumeText(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict/")
def predict(request: ResumeText):
    vec = vectorizer.transform([request.text])
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

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Root endpoint
@app.get("/")
def root():
    logger.info("Root endpoint hit")
    return {"message": "Resume Classification API is running!"}

# Mount metrics endpoint
app.mount("/metrics", make_asgi_app())

# Debug: Print registered routes
print("\n--- Registered Routes ---")
for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"{route.path} -> {route.methods}")
print("-------------------------\n")
