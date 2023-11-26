import shutil, time, os, tempfile, requests
from fastapi.testclient import TestClient
from app.main import app, BASE_DIR, UPLOAD_DIR, get_settings

ENDPOINT = "https://cma-microservice-nha72.ondigitalocean.app/"
ENDPOINT_ECHO = "https://cma-microservice-nha72.ondigitalocean.app/file-echo/"

def test_get_home():
    response = requests.get(ENDPOINT)
    assert response.status_code == 200
    assert "text/html" in response.headers['content-type']

def test_post_file_analysis():
    settings = get_settings()
    file_saved_path = BASE_DIR / "testFiles"

    for path in file_saved_path.glob("*"):
        with open(path, 'rb') as file:
            fext = file.name.split('/')[-1]

            files = {'file': (fext, file, 'multipart/form-data')}
            response = requests.post(ENDPOINT,
                files=files,
                headers={"Authorization": f"JWT {settings.app_auth_token_prod}"}
            )
            assert response.status_code == 200
            assert "application/json" in response.headers['content-type']

def test_post_file_analysis_bad_auth():
    settings = get_settings()
    file_saved_path = BASE_DIR / "testFiles"

    for path in file_saved_path.glob("*"):
        with open(path, 'rb') as file:
            fext = file.name.split('/')[-1]

            files = {'file': (fext, file, 'multipart/form-data')}
            response = requests.post(ENDPOINT,
                files=files,
            )
            assert response.status_code == 401
            assert "application/json" in response.headers['content-type']

