from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from typing import override

import pytest
from fastapi import HTTPException
from openai import APIError, AsyncOpenAI, UnprocessableEntityError
from openai.types import CompletionChoice
from openai.types.completion import Completion
from pydantic import BaseModel

from laime.backends.abc import Backend, TextGenerationBackend
from laime.testclient import async_test_client


class TextGenTestBackendConfig(BaseModel):
	next_completion_func: Callable[[], Awaitable[Completion]]


class TextGenTestBackend(Backend[TextGenTestBackendConfig], TextGenerationBackend):
	backend_name: str = "textgen_test"
	backend_config_model = TextGenTestBackendConfig  # pyright: ignore [reportUnannotatedClassAttribute]
	next_completion_func: Callable[[], Awaitable[Completion]]

	@override
	def __init__(self, config: TextGenTestBackendConfig) -> None:
		self.next_completion_func = config.next_completion_func

	@override
	async def completion(
		self,
		prompt: str | list[str] | Iterable[int] | Iterable[Iterable[int]] | None,
		frequency_penalty: float | None,
		logit_bias: dict[str, int] | None,
		presence_penalty: float | None,
		seed: int | None,
		stop: str | list[str] | None,
		temperature: float | None,
		top_p: float | None,
	) -> AsyncGenerator[Completion]:
		while True:
			completion: Completion = await self.next_completion_func()
			yield completion
			if completion.choices[0].finish_reason:
				break


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
	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"textgen_test": TextGenTestBackend(
				TextGenTestBackendConfig(next_completion_func=next_completion_func)
			)
		},
	)
	async for chunk in await openai.completions.create(
		model="textgen_test", prompt="Hello ", stream=True
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
async def test__unable_to_handle_inputs(monkeypatch: pytest.MonkeyPatch):
	async def next_completion_func():
		pass  # pragma: no cover

	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"textgen_test": TextGenTestBackend(
				TextGenTestBackendConfig(next_completion_func=next_completion_func)  # pyright: ignore [reportArgumentType]
			)
		},
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		async for chunk in await openai.completions.create(  # pyright: ignore [reportCallIssue, reportUnknownVariableType]
			model=0,  # pyright: ignore [reportArgumentType]
			prompt="Hello ",
			stream=True,
		):
			pass  # pragma: no cover
	assert excinfo.value.body == {
		"detail": {"message": "Unable to handle your inputs."}
	}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_completion_stream_error(monkeypatch: pytest.MonkeyPatch):
	async def next_completion_func():
		raise HTTPException(
			status_code=500,
			detail={"message": "error handling test"},
		)

	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"textgen_test": TextGenTestBackend(
				TextGenTestBackendConfig(next_completion_func=next_completion_func)
			)
		},
	)
	with pytest.raises(APIError) as excinfo:
		async for chunk in await openai.completions.create(
			model="textgen_test",
			prompt="Hello ",
			stream=True,
		):
			pass  # pragma: no cover
	assert excinfo.value.body == {
		"type": "server_error",
		"code": 500,
		"message": "error handling test",
	}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_completion_non_streaming_error(monkeypatch: pytest.MonkeyPatch):
	async def next_completion_func():
		pass  # pragma: no cover

	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	monkeypatch.setattr(
		"laime.models.model_backends",
		{
			"textgen_test": TextGenTestBackend(
				TextGenTestBackendConfig(next_completion_func=next_completion_func)  # pyright: ignore [reportArgumentType]
			)
		},
	)
	with pytest.raises(UnprocessableEntityError) as excinfo:
		_ = await openai.completions.create(
			model="textgen_test",
			prompt="Hello ",
			stream=False,
		)
	assert excinfo.value.body == {
		"detail": {
			"message": "laime currently does not support non-streaming responses.",
		}
	}
