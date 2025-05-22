import pytest
import pytest_asyncio # For async fixtures
from httpx import AsyncClient
from pathlib import Path
import os
import shutil
from unittest.mock import patch, MagicMock, AsyncMock

# Import constants and app components from the main app
# Adjust the import path if your app structure is different
from projects.web_api.app import app, DOWNLOAD_DIR, OUTPUT_DIR_BASE

# --- Test Setup and Teardown ---

@pytest.fixture(scope="module")
def setup_test_environment():
    """
    Creates DOWNLOAD_DIR and OUTPUT_DIR_BASE before tests
    and cleans them up after tests.
    Also creates a dummy PDF for /get_pdf/ tests.
    """
    # Create directories
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

    # Create a dummy PDF file in DOWNLOAD_DIR
    dummy_pdf_content = b"%PDF-1.0\n%EOF" # Minimal valid PDF
    dummy_pdf_path = DOWNLOAD_DIR / "dummy.pdf"
    with open(dummy_pdf_path, "wb") as f:
        f.write(dummy_pdf_content)
    
    # Create another dummy file for a specific test case
    dummy_for_download_test_path = DOWNLOAD_DIR / "test_pdf_123.pdf"
    with open(dummy_for_download_test_path, "wb") as f:
        f.write(dummy_pdf_content)


    yield # This is where the testing happens

    # Teardown: Remove directories and their contents
    shutil.rmtree(DOWNLOAD_DIR)
    shutil.rmtree(OUTPUT_DIR_BASE)

# --- Mock Objects ---

# Mock for InferenceResult and PipeResult
# These need to be defined at the module level for @patch to find them easily if used directly
# However, it's often cleaner to create them inside the test or a fixture if they are stateful.

mock_infer_result_global = MagicMock()
mock_pipe_result_global = MagicMock()

def mock_dump_md_func(writer, base_path, image_folder_name):
    # Simulate writing MD content to the writer
    writer.write_string(base_path, "## Mocked Markdown Content")

mock_pipe_result_global.dump_md = MagicMock(side_effect=mock_dump_md_func)


# --- Test Cases ---

@pytest.mark.usefixtures("setup_test_environment") # Apply setup/teardown to all tests in this module
class TestPdfApiEndpoints:

    # Tests for POST /parse_pdf_from_url

    @patch('projects.web_api.app.process_file', return_value=(mock_infer_result_global, mock_pipe_result_global))
    @patch('projects.web_api.app.requests.get') # Patch where requests.get is used
    async def test_parse_pdf_from_url_success(self, mock_requests_get, mock_process_file_func, client: AsyncClient):
        # Configure the mock for requests.get
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"%PDF-1.4...", b"some more content"] # Dummy PDF bytes
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_requests_get.return_value = mock_response

        pdf_url_to_test = "http://example.com/test.pdf"
        response = await client.post("/parse_pdf_from_url", json={"pdf_url": pdf_url_to_test, "parse_method": "auto"})

        assert response.status_code == 200
        json_response = response.json()
        assert "downloaded_pdf_path" in json_response
        assert "markdown_file_path" in json_response
        assert "md_content" in json_response
        assert json_response["md_content"] == "## Mocked Markdown Content"

        # Verify process_file was called
        mock_process_file_func.assert_called_once()

        # Verify requests.get was called
        mock_requests_get.assert_called_once_with(pdf_url_to_test, stream=True, timeout=30)

        # Verify downloaded file exists (name derived from URL)
        expected_download_filename = "test.pdf" # Based on pdf_url_to_test
        assert (DOWNLOAD_DIR / expected_download_filename).exists()
        assert (DOWNLOAD_DIR / expected_download_filename).is_file()


        # Verify Markdown file was created
        # The filename_without_ext logic in app.py is "test"
        # Output path is OUTPUT_DIR_BASE / "test" / "test.md"
        expected_md_filename = "test.md"
        expected_md_dir = OUTPUT_DIR_BASE / "test"
        assert (expected_md_dir / expected_md_filename).exists()
        
        with open(expected_md_dir / expected_md_filename, "r") as f:
            content = f.read()
            assert content == "## Mocked Markdown Content"


    async def test_parse_pdf_from_url_invalid_url_scheme(self, client: AsyncClient):
        response = await client.post("/parse_pdf_from_url", json={"pdf_url": "ftp://example.com/test.pdf"})
        assert response.status_code == 400
        assert "Invalid URL scheme" in response.json()["detail"]

    async def test_parse_pdf_from_url_no_filename_in_url(self, client: AsyncClient):
        # This test assumes that URLs ending with / or having no clear filename part after the last /
        # and no Content-Disposition header will result in a default filename or an error.
        # The current implementation defaults to "downloaded_file.pdf" or similar if it can't derive a name.
        # Let's test a URL that might cause issues with Path(url_path.name)
        # The code was updated to add .pdf if no extension, this test might need specific mocking
        # for the filename generation part if we want to test that edge case precisely.
        # For now, let's assume it appends .pdf and tries to download.
        # If requests.get is not mocked here, it would try to actually download.
        # We need to mock requests.get to avoid real network call.
        with patch('projects.web_api.app.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"%PDF-1.0...", b"trailer"]
            mock_response.headers = {'Content-Type': 'application/pdf'}
            mock_get.return_value = mock_response
            
            response = await client.post("/parse_pdf_from_url", json={"pdf_url": "http://example.com/nodirectfilename/"})
            # Assuming the default filename logic kicks in (e.g., "nodirectfilename.pdf" or similar if it takes last part)
            # Or if it's truly empty, it should error earlier, but the code tries to append .pdf to an empty string.
            # The current code `filename = url_path.name` if url_path is `http://example.com/nodirectfilename/`
            # `url_path.name` will be 'nodirectfilename'. Then it adds .pdf. So it becomes 'nodirectfilename.pdf'.
            # This should pass if process_file is also mocked.
            with patch('projects.web_api.app.process_file', return_value=(mock_infer_result_global, mock_pipe_result_global)):
                 response = await client.post("/parse_pdf_from_url", json={"pdf_url": "http://example.com/nodirectfilename/"})
                 assert response.status_code == 200 # Expecting success as filename becomes "nodirectfilename.pdf"
                 assert (DOWNLOAD_DIR / "nodirectfilename.pdf").exists()


    @patch('projects.web_api.app.requests.get')
    async def test_parse_pdf_from_url_download_failure(self, mock_requests_get, client: AsyncClient):
        mock_requests_get.side_effect = Exception("Simulated download error") # Simulate requests.exceptions.RequestException

        response = await client.post("/parse_pdf_from_url", json={"pdf_url": "http://example.com/failing.pdf"})
        assert response.status_code == 500
        assert "Failed to download PDF" in response.json()["detail"]
        assert not (DOWNLOAD_DIR / "failing.pdf").exists() # Ensure no partial file is left if download fails early

    @patch('projects.web_api.app.requests.get')
    async def test_parse_pdf_from_url_unsupported_content_type(self, mock_requests_get, client: AsyncClient):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"some plain text content"]
        mock_response.headers = {'Content-Type': 'text/plain'} # Unsupported type
        mock_requests_get.return_value = mock_response

        response = await client.post("/parse_pdf_from_url", json={"pdf_url": "http://example.com/document.txt"})
        
        assert response.status_code == 400
        json_response = response.json()
        assert "content type ('text/plain') is not supported" in json_response["detail"]
        # Check that the downloaded file (even if temporary) is cleaned up
        assert not (DOWNLOAD_DIR / "document.txt.pdf").exists() # name might get .pdf appended by default logic
        assert not (DOWNLOAD_DIR / "document.txt").exists()


    # Tests for GET /get_pdf/{filename}

    async def test_get_pdf_success(self, client: AsyncClient):
        # This relies on "dummy.pdf" created in setup_test_environment
        response = await client.get("/get_pdf/dummy.pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        # Check if content is roughly what we wrote (optional, but good for sanity)
        assert b"%PDF-1.0" in response.content
        assert b"%EOF" in response.content

    async def test_get_pdf_not_found(self, client: AsyncClient):
        response = await client.get("/get_pdf/non_existent_dummy.pdf")
        assert response.status_code == 404
        assert "PDF not found" in response.json()["detail"]

    @patch('projects.web_api.app.Path.is_file', return_value=False) # Mock Path.is_file to simulate file disappearing after check
    async def test_get_pdf_file_disappears_after_check(self, mock_is_file, client: AsyncClient):
        # This is an edge case, less critical but good for robustness
        # To make this test work, we need a file that exists for the initial check
        # but then disappears. The fixture creates "dummy.pdf".
        # Path.is_file will be called by the endpoint.
        # We make it return False to simulate file gone missing.
        # The current implementation of get_pdf in app.py checks if pdf_path.is_file().
        # If we mock this to return False, it will directly go to 404.
        # A more complex scenario would be if it's True, then file gets deleted before FileResponse.
        # However, FileResponse itself might error. For now, this covers the check.
        
        # To ensure the file "exists" for the path construction part, but is_file() check fails
        # We can rely on dummy.pdf existing from the fixture.
        # The mock_is_file will make the `if not pdf_path.is_file():` condition true.
        response = await client.get("/get_pdf/dummy.pdf") # dummy.pdf does exist
        assert response.status_code == 404 # Because is_file is mocked to False
        assert "PDF not found" in response.json()["detail"]

    @patch('projects.web_api.app.process_file', return_value=(mock_infer_result_global, mock_pipe_result_global))
    @patch('projects.web_api.app.requests.get')
    async def test_parse_pdf_from_url_filename_sanitization(self, mock_requests_get, mock_process_file, client: AsyncClient):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"%PDF-1.0..."]
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_requests_get.return_value = mock_response

        # Test with a URL that has characters needing sanitization
        # Example: "http://example.com/file with spaces & special_chars!.pdf"
        # Expected sanitized name: "file_with_spaces___special_chars_.pdf" (based on current app.py logic)
        pdf_url = "http://example.com/file with spaces & special_chars!.pdf"
        expected_sanitized_filename = "file_with_spaces___special_chars_.pdf"

        response = await client.post("/parse_pdf_from_url", json={"pdf_url": pdf_url})
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["downloaded_pdf_path"] == str(DOWNLOAD_DIR / expected_sanitized_filename)
        
        # Verify the file was created with the sanitized name
        assert (DOWNLOAD_DIR / expected_sanitized_filename).exists()
        
        # Verify MD file structure (filename_without_ext should be "file_with_spaces___special_chars_")
        filename_without_ext = Path(expected_sanitized_filename).stem
        expected_md_dir = OUTPUT_DIR_BASE / filename_without_ext
        expected_md_file = expected_md_dir / f"{filename_without_ext}.md"
        assert expected_md_file.exists()
        assert json_response["markdown_file_path"] == str(expected_md_file)

    @patch('projects.web_api.app.requests.get')
    async def test_parse_pdf_from_url_office_document_content_type(self, mock_requests_get, client: AsyncClient):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # Simulate .docx content type
        mock_response.headers = {'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}
        mock_response.iter_content.return_value = [b"PK...", b"word/document.xml..."] # Dummy docx bytes
        mock_requests_get.return_value = mock_response

        # Mock process_file to ensure it's called with the correct extension
        mock_infer = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.dump_md = MagicMock(side_effect=lambda w, p, i: w.write_string(p, "## Office Content"))

        with patch('projects.web_api.app.process_file', return_value=(mock_infer, mock_pipe)) as mock_pf:
            # URL has no extension, so content-type will be primary
            response = await client.post("/parse_pdf_from_url", json={"pdf_url": "http://example.com/mydoc"})
            assert response.status_code == 200
            json_response = response.json()
            
            # Check that the downloaded file has .docx extension appended
            # Original filename "mydoc" + derived extension ".docx"
            downloaded_file_path = Path(json_response["downloaded_pdf_path"])
            assert downloaded_file_path.name == "mydoc.docx" # Endpoint logic appends .docx
            assert downloaded_file_path.exists()

            # Check that process_file was called with .docx
            args, kwargs = mock_pf.call_args
            assert kwargs['file_extension'] == '.docx'
            assert json_response["md_content"] == "## Office Content"
            
            # Check MD file path
            md_file_path = Path(json_response["markdown_file_path"])
            assert md_file_path.name == "mydoc.md" # Based on filename_without_ext "mydoc"


    @patch('projects.web_api.app.requests.get')
    async def test_parse_pdf_from_url_image_content_type(self, mock_requests_get, client: AsyncClient):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {'Content-Type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b"\xff\xd8\xff\xe0...", b"some image data"] # Dummy jpg bytes
        mock_requests_get.return_value = mock_response

        mock_infer = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.dump_md = MagicMock(side_effect=lambda w, p, i: w.write_string(p, "## Image Content"))

        with patch('projects.web_api.app.process_file', return_value=(mock_infer, mock_pipe)) as mock_pf:
            response = await client.post("/parse_pdf_from_url", json={"pdf_url": "http://example.com/myimage"})
            assert response.status_code == 200
            json_response = response.json()

            downloaded_file_path = Path(json_response["downloaded_pdf_path"])
            assert downloaded_file_path.name == "myimage.jpg" # Endpoint appends .jpg
            assert downloaded_file_path.exists()

            args, kwargs = mock_pf.call_args
            assert kwargs['file_extension'] == '.jpg'
            assert json_response["md_content"] == "## Image Content"

            md_file_path = Path(json_response["markdown_file_path"])
            assert md_file_path.name == "myimage.md"

# To run these tests:
# Ensure pytest, httpx, pytest-asyncio are installed.
# Navigate to the directory containing `projects`
# Run `pytest projects/web_api/tests/test_app.py`
# Or from `projects/web_api`: `pytest tests/test_app.py`
# The conftest.py should be automatically picked up if it's in the tests directory or a parent.
# The setup_test_environment fixture handles directory creation/cleanup.
# The client fixture is provided by conftest.py.

# Note on file cleanup:
# The `setup_test_environment` fixture should handle cleaning DOWNLOAD_DIR and OUTPUT_DIR_BASE.
# Individual tests might create files like "test.pdf" or "failing.pdf" inside DOWNLOAD_DIR.
# These are cleaned as part of the directory removal.
# If a download fails mid-way, the current app logic doesn't guarantee cleanup of partial files
# *before* raising the HTTPException, but the test for download failure checks this.
# The unsupported content type test also verifies cleanup of the temporarily downloaded file.

```
