from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from typing import TypeVar

from openai.types.completion import Completion
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Backend[T](ABC):
	backend_name: str
	backend_config_model: type[T]

	@abstractmethod
	def __init__(self, config: T) -> None:
		pass  # pragma: no cover


class EmbeddingsBackend(ABC):
	@abstractmethod
	async def embed(self, input: str) -> list[float]:
		pass  # pragma: no cover

	@abstractmethod
	async def count_tokens(self, input: str) -> int:
		pass  # pragma: no cover


class TextGenerationBackend(ABC):
	@abstractmethod
	def completion(
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
		pass  # pragma: no cover
