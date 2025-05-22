import pytest
import pytest_asyncio # Import this to use the new fixture type
from httpx import AsyncClient
from projects.web_api.app import app # Adjust import path as necessary

@pytest_asyncio.fixture(scope="session") # Use pytest_asyncio.fixture for async fixtures
async def client():
    # Set a base URL that doesn't conflict with a running server,
    # as httpx.AsyncClient will handle the app directly.
    async with AsyncClient(app=app, base_url="http://127.0.0.1:8001") as ac: # Port can be arbitrary for testing when app is passed
        yield ac
