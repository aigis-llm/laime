import atexit
import random
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from subprocess import PIPE, Popen
from typing import override

import httpx
from openai.types.completion import Completion
from pydantic import BaseModel

from laime.backends.abc import Backend, EmbeddingsBackend, TextGenerationBackend
from laime.backends.openai_proxy import (
	OpenAIProxyBackend,
	OpenAIProxyBackendConfig,
	SupportedAPIs,
)
from laime.settings import get_settings


class LlamaServerBackendConfig(BaseModel):
	model_path: Path
	server_flags: list[str] = []
	supported_apis: SupportedAPIs
	embedding_post_token: str = ""


class LlamaServerBackend(
	Backend[LlamaServerBackendConfig], EmbeddingsBackend, TextGenerationBackend
):
	backend_name: str = "llama_server"
	backend_config_model = LlamaServerBackendConfig  # pyright: ignore [reportUnannotatedClassAttribute]
	openai_proxy: OpenAIProxyBackend
	port: int
	process: Popen[bytes]
	model_name: str
	supported_apis: SupportedAPIs

	def __init__(self, config: LlamaServerBackendConfig) -> None:
		self.port = random.randrange(1024, 65536)
		self.process = Popen(
			[
				"llama-server",
				"--host",
				"127.0.0.1",
				"--port",
				f"{self.port}",
				"-m",
				config.model_path,
			]
			+ config.server_flags,
			cwd=get_settings().models_dir,
			stdout=PIPE,
		)
		self.openai_proxy = OpenAIProxyBackend(
			OpenAIProxyBackendConfig(
				endpoint=f"http://localhost:{self.port}/v1/",
				api_key_env_var="NO_API_KEY",
				model_name="model",
				supported_apis=config.supported_apis,
				embedding_post_token=config.embedding_post_token,
			)
		)
		_ = atexit.register(self.__del__)
		while True:
			try:
				if httpx.get(f"http://localhost:{self.port}/health").status_code == 200:
					break
			except httpx.ConnectError:  # pragma: no cover
				pass

	@override
	async def embed(self, input: str) -> list[float]:
		return await self.openai_proxy.embed(input)

	@override
	async def count_tokens(self, input: str) -> int:
		return await self.openai_proxy.count_tokens(input)

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
		async for chunk in self.openai_proxy.completion(
			prompt,
			frequency_penalty,
			logit_bias,
			presence_penalty,
			seed,
			stop,
			temperature,
			top_p,
		):
			yield chunk

	def __del__(self):
		self.process.kill()  # pragma: no cover
