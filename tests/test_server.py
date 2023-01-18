# from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.server.main import app

client = TestClient(app)


def test_show_info():
    response = client.get("/info")
    print(response.json())
    assert response.status_code == 200


def test_post_predict():

    body = {"index": 10}
    response = client.post("/api/v1/predict", headers={}, json=body)

    print(response.json())
    assert response.status_code == 200
