from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 7,
        "sepal_width": 3,
        "petal_length": 6,
        "petal_width": 2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Virginica"
        assert "timestamp" in response.json()


# test to check if Iris Versicolour is classified correctly
def test_pred_versicolour():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 6,
        "sepal_width": 3,
        "petal_length": 4,
        "petal_width": 1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Versicolour"
        assert "timestamp" in response.json()


# test to check if Iris Setoda is classified correctly
def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5,
        "sepal_width": 3,
        "petal_length": 1,
        "petal_width": 0,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Setosa"
        assert "timestamp" in response.json()
