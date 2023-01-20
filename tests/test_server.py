import os

import pytest
from fastapi.testclient import TestClient

from src.server.main import app
from tests import _PATH_CKPT

client = TestClient(app)


def test_show_info():
    response = client.get("/info")
    print(response.json())
    assert response.status_code == 200


@pytest.mark.skipif(not os.path.exists(_PATH_CKPT), reason="Checkpoint not found")
def test_post_predict():

    body = {"index": 10}
    response = client.post("/api/v1/predict", headers={}, json=body)

    print(response.json())
    assert response.status_code == 200
