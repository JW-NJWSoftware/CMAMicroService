import shutil, time, os, tempfile
from fastapi.testclient import TestClient
from app.main import app, BASE_DIR, UPLOAD_DIR, get_settings

client = TestClient(app)

def test_get_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers['content-type']

def test_post_file_analysis():
    settings = get_settings()
    file_saved_path = BASE_DIR / "testFiles"
    for path in file_saved_path.glob("*"):
        with open(path, 'rb') as file:
            fext = file.name.split('/')[-1]

            files = {'file': (fext, file, 'multipart/form-data')}
            response = client.post("/",
                files=files,
                headers={"Authorization": f"JWT {settings.app_auth_token}"}
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
            response = client.post("/",
                files=files,
            )
            assert response.status_code == 401
            assert "application/json" in response.headers['content-type']

def test_echo_upload():
    file_saved_path = BASE_DIR / "testFiles"

    for path in file_saved_path.glob("*"):
        with open(path, 'rb') as file:
            fext = file.name.split('/')[-1]

            files = {'file': (fext, file, 'multipart/form-data')}
            response = client.post("/file-echo/", files=files)

            # Check if the request was successful (status code 200)
            assert response.status_code == 200

            # Get the uploaded file path returned in the response
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            # Check if the uploaded file exists and its contents match the original file
            assert os.path.exists(temp_file_path)

            # Compare the content of the uploaded file and the original file
            with open(path, 'rb') as original_file, open(temp_file_path, 'rb') as uploaded_file:
                assert original_file.read() == uploaded_file.read()

    # time.sleep(10)
    shutil.rmtree(UPLOAD_DIR)