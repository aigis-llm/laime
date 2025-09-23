import os
from collections.abc import AsyncGenerator, Iterable
from typing import Literal, override

from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.completion import Completion
from pydantic import BaseModel

from laime.backends.abc import Backend, EmbeddingsBackend, TextGenerationBackend

type SupportedAPIs = list[Literal["embeddings", "completion"]]


class OpenAIProxyBackendConfig(BaseModel):
	endpoint: str
	api_key_env_var: str
	model_name: str
	supported_apis: SupportedAPIs
	embedding_post_token: str = ""


class OpenAIProxyBackend(
	Backend[OpenAIProxyBackendConfig], EmbeddingsBackend, TextGenerationBackend
):
	backend_name: str = "openai_proxy"
	backend_config_model = OpenAIProxyBackendConfig  # pyright: ignore [reportUnannotatedClassAttribute]
	openai_client: AsyncOpenAI
	model_name: str
	supported_apis: SupportedAPIs
	embedding_post_token: str = ""

	def __init__(self, config: OpenAIProxyBackendConfig) -> None:
		self.model_name = config.model_name
		self.supported_apis = config.supported_apis
		self.openai_client = AsyncOpenAI(
			base_url=config.endpoint,
			api_key=os.environ.get(config.api_key_env_var, "none"),
		)

	@override
	async def embed(self, input: str) -> list[float]:
		if "embeddings" in self.supported_apis:
			return (
				(
					await self.openai_client.embeddings.create(
						input=input + self.embedding_post_token, model=self.model_name
					)
				)
				.data[0]
				.embedding
			)
		else:
			raise HTTPException(
				status_code=422,
				detail={"message": f"{self.model_name} is not an embedding model."},
			)

	@override
	async def count_tokens(self, input: str) -> int:
		return 0  # TODO: implement this right

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
		if "completion" in self.supported_apis:
			stream = await self.openai_client.completions.create(
				model=self.model_name,
				stream=True,
				prompt=prompt,
				frequency_penalty=frequency_penalty,
				logit_bias=logit_bias,
				presence_penalty=presence_penalty,
				seed=seed,
				stop=stop,
				temperature=temperature,
				top_p=top_p,
			)
			async for chunk in stream:
				yield chunk
		else:
			raise HTTPException(
				status_code=422,
				detail={"message": f"{self.model_name} is not a completion model."},
			)
