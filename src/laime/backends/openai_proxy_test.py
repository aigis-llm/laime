import pytest
from openai import APIError, AsyncOpenAI, UnprocessableEntityError
from openai.types import CompletionChoice
from openai.types.completion import Completion

from laime.api.openai_api.text_generation_test import (
	TextGenTestBackend,
	TextGenTestBackendConfig,
)
from laime.backends.openai_proxy import OpenAIProxyBackend, OpenAIProxyBackendConfig
from laime.backends.sentence_transformers import (
	SentenceTransformersBackend,
	SentenceTransformersBackendConfig,
)
from laime.testclient import async_test_client


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_create(
	monkeypatch: pytest.MonkeyPatch,
):
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)

	openai_proxy = OpenAIProxyBackend(
		OpenAIProxyBackendConfig(
			endpoint="http://laime_test/openai/v1",
			api_key_env_var="",
			model_name="embedding_test",
			supported_apis=["embeddings"],
		)
	)
	openai_proxy.openai_client = openai

	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"embedding_test": SentenceTransformersBackend(
				SentenceTransformersBackendConfig(
					model="mixedbread-ai/mxbai-embed-large-v1",
					device="cpu",
				)
			),
			"openai_proxy": openai_proxy,
		},
	)

	embedding = (
		(await openai.embeddings.create(model="openai_proxy", input="a"))
		.data[0]
		.embedding
	)
	assert len(embedding) == 1024


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_fail(
	monkeypatch: pytest.MonkeyPatch,
):
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)

	openai_proxy = OpenAIProxyBackend(
		OpenAIProxyBackendConfig(
			endpoint="http://laime_test/openai/v1",
			api_key_env_var="",
			model_name="no_model",
			supported_apis=[],
		)
	)
	openai_proxy.openai_client = openai

	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"openai_proxy": openai_proxy,
		},
	)

	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = (
			(await openai.embeddings.create(model="openai_proxy", input="a"))
			.data[0]
			.embedding
		)
	assert excinfo.value.body == {
		"detail": {"message": "no_model is not an embedding model."}
	}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_completion_create(
	monkeypatch: pytest.MonkeyPatch,
):
	async def next_completion_func():
		return Completion(
			id="test0",
			choices=[CompletionChoice(finish_reason="length", text="World!", index=0)],
			created=0,
			model="",
			object="text_completion",
		)

	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)

	openai_proxy = OpenAIProxyBackend(
		OpenAIProxyBackendConfig(
			endpoint="http://laime_test/openai/v1",
			api_key_env_var="",
			model_name="textgen_test",
			supported_apis=["completion"],
		)
	)
	openai_proxy.openai_client = openai

	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"textgen_test": TextGenTestBackend(
				TextGenTestBackendConfig(next_completion_func=next_completion_func)
			),
			"openai_proxy": openai_proxy,
		},
	)

	async for chunk in await openai.completions.create(
		model="openai_proxy", prompt="Hello ", stream=True
	):
		assert chunk == Completion(
			id="test0",
			choices=[
				CompletionChoice(
					finish_reason="length", index=0, logprobs=None, text="World!"
				)
			],
			created=0,
			model="",
			object="text_completion",
			system_fingerprint=None,
			usage=None,
		)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_completion_fail(
	monkeypatch: pytest.MonkeyPatch,
):
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)

	openai_proxy = OpenAIProxyBackend(
		OpenAIProxyBackendConfig(
			endpoint="http://laime_test/openai/v1",
			api_key_env_var="",
			model_name="no_model",
			supported_apis=[],
		)
	)
	openai_proxy.openai_client = openai

	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"openai_proxy": openai_proxy,
		},
	)

	with pytest.raises(APIError) as excinfo:
		async for chunk in await openai.completions.create(
			model="openai_proxy", prompt="Hello ", stream=True
		):
			pass  # pragma: no cover
	assert excinfo.value.body == {
		"type": "server_error",
		"code": 422,
		"message": "no_model is not a completion model.",
	}
