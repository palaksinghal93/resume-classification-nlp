import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app


client = TestClient(app)

def test_predict():
    response = client.post("/predict/", json={"text": "Python developer with ML experience"})
    assert response.status_code == 200
    assert "predicted_category" in response.json()

