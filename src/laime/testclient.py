from httpx import ASGITransport, AsyncClient

from laime import app


def async_test_client() -> AsyncClient:
	return AsyncClient(transport=ASGITransport(app=app), base_url="http://laime_test")
