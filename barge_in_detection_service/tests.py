# To test run: python3 tests.py
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch
from barge_in_model.barge_in_inference_pipeline import EmbeddingModelError, BargeInModelError

client = TestClient(app)

def test_invalid_data_format():
    # empty object
    response = client.post("/query-interrupt", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid data format."
    # missing 'id'
    response = client.post("/query-interrupt", json={"audio": {"data": "1234"}})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid data format."
    # audio data not base64
    response = client.post("/query-interrupt", json={"audio": {"data": "@"}})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid data format."
    print("test_invalid_data_format passed!")

def test_none_or_empty_audio():
    response = client.post("/query-interrupt", json={"audio": {"data": ""}, "id": "1234"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Audio data cannot be None or of length 0."
    print("test_none_or_empty_audio passed!")

def test_invalid_base64_data():
    response = client.post("/query-interrupt", json={"audio": {"data": "invalidbase64data"}, "id": "1234"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid base64 audio data."
    print("test_invalid_base64_data passed!")

def test_valid_base64_data():
    with open('./testing_resources/bargein_base64_audio.txt', 'r') as file:
        base64_audio_string = file.read()
    response = client.post("/query-interrupt", json={"audio": {"data": base64_audio_string}, "id": "1234"})
    assert response.status_code == 200
    print("test_valid_base64_data passed!")

def raise_barge_in_model_error(*args, **kwargs):
    print('Mock function called')
    e = Exception("Barge-in classification model error")
    raise BargeInModelError(f"Error during barge-in classifier model processing: {str(e)}") from e

@patch("app.check_barge_in", side_effect=raise_barge_in_model_error)
def test_internal_barge_in_model_error(mock_check_barge_in):  
    with open('./testing_resources/bargein_base64_audio.txt', 'r') as file:
        base64_audio_string = file.read()
    response = client.post("/query-interrupt", json={"audio": {"data": base64_audio_string}, "id": "1234"})
    assert response.status_code == 500
    print("test_check_barge_in_error passed!")

def raise_embedding_model_error(*args, **kwargs):
    e = Exception("Embedding generation error")
    raise EmbeddingModelError(f"Error during embedding model processing: {str(e)}") from e

@patch("app.check_barge_in", side_effect=raise_embedding_model_error)
def test_internal_embedding_model_error(mock_check_barge_in):  
    with open('./testing_resources/bargein_base64_audio.txt', 'r') as file:
        base64_audio_string = file.read()
    response = client.post("/query-interrupt", json={"audio": {"data": base64_audio_string}, "id": "1234"})
    assert response.status_code == 500
    print("test_check_barge_in_error passed!")

if __name__ == "__main__":
    test_invalid_data_format()
    test_none_or_empty_audio()
    test_invalid_base64_data()
    test_valid_base64_data()
    test_internal_barge_in_model_error()
    test_internal_embedding_model_error()