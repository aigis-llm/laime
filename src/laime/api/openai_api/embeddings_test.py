import hashlib
import struct

import pytest
from openai import AsyncOpenAI, UnprocessableEntityError

from laime.backends.none import NoneBackend, NoneBackendConfig
from laime.testclient import async_test_client


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_create():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	embedding = (
		(await openai.embeddings.create(model="mxbai-embed-large-v1", input="a"))
		.data[0]
		.embedding
	)
	assert (
		hashlib.sha3_256(struct.pack("%sf" % len(embedding), *embedding)).hexdigest()
		== "ab888bfc28bf023fa9ded6d8d6e0bed761d17ddb862b223bd64624198ca7a54f"
	)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_no_model():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = await openai.embeddings.create(model="no_model", input="a")
	assert excinfo.value.body == {"detail": {"message": "No model no_model."}}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_not_embedding_model():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = await openai.embeddings.create(model="not_embedding_model", input="a")
	assert excinfo.value.body == {
		"detail": {"message": "not_embedding_model is not an embedding model."}
	}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_not_embedding_model_existing_backend(
	monkeypatch: pytest.MonkeyPatch,
):
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	monkeypatch.setattr(
		"laime.api.openai_api.embeddings.model_backends",
		{"not_embedding_model": NoneBackend(NoneBackendConfig())},
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = await openai.embeddings.create(model="not_embedding_model", input="a")
	assert excinfo.value.body == {
		"detail": {"message": "not_embedding_model is not an embedding model."}
	}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_multiple_inputs():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	embeddings = await openai.embeddings.create(
		model="mxbai-embed-large-v1", input=["a", "b"]
	)
	assert (
		hashlib.sha3_256(
			struct.pack(
				"%sf" % len(embeddings.data[0].embedding), *embeddings.data[0].embedding
			)
		).hexdigest()
		== "ab888bfc28bf023fa9ded6d8d6e0bed761d17ddb862b223bd64624198ca7a54f"
	)
	assert (
		hashlib.sha3_256(
			struct.pack(
				"%sf" % len(embeddings.data[1].embedding), *embeddings.data[1].embedding
			)
		).hexdigest()
		== "f43b9ba424d7790e266bf11b2706204b390abfe4640bf74581b1ff6b8efa7aa9"
	)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_unable_to_handle_inputs():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = await openai.embeddings.create(
			model="mxbai-embed-large-v1",
			input=None,  # pyright: ignore [reportArgumentType]
		)
	assert excinfo.value.body == {
		"detail": {"message": "Unable to handle your inputs."}
	}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_unable_to_handle_integers():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = await openai.embeddings.create(
			model="mxbai-embed-large-v1",
			input=[1],
		)
	assert excinfo.value.body == {
		"detail": {"message": "laime is unable to handle non-string inputs."}
	}
