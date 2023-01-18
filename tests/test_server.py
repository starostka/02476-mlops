import os
import pytest
import torch
# from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests import _PATH_DATA
from src.server.main import app

client = TestClient(app)


def test_show_info():
    response = client.get('/info')
    print(response.json())
    assert response.status_code == 200


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_post_predict():
    data = torch.load(_PATH_DATA)[0]

    sample_idx = 35
    sample = data.x[data.test_mask][sample_idx]

    body = {'data': sample.tolist()}
    response = client.post('/api/v1/predict', headers={}, json=body)

    print(f"Post Predict result: {response}")
    print("Response")
    print(response.json())
    assert response.status_code == 500
