# src/monitoring.py
import logging
from prometheus_client import Counter, Histogram

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),   # Save logs to file
        logging.StreamHandler()                # Show logs in console
    ]
)

logger = logging.getLogger(__name__)

# ---------------- Metrics Setup ----------------
REQUEST_COUNT = Counter("request_count", "Total requests received")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")
