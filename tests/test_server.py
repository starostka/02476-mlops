import os
import pytest
import torch
# from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests import _PATH_DATA, _PATH_CKPT
from src.server.main import app

client = TestClient(app)


def test_show_info():
    response = client.get('/info')
    print(response.json())
    assert response.status_code == 200


@pytest.mark.skipif(not os.path.exists(_PATH_CKPT), reason="Checkpoint not found")
def test_post_predict():

    body = {'index': 10}
    response = client.post('/api/v1/predict', headers={}, json=body)

    print(response.json())
    assert response.status_code == 200
