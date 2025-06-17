from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Backend[T](ABC):
	backend_name: str
	backend_config_model: type[T]

	@abstractmethod
	def __init__(self, config: T) -> None:
		pass


class EmbeddingsBackend(Backend[T], ABC):
	@abstractmethod
	async def embed(self, input: str) -> list[float]:
		pass

	@abstractmethod
	async def count_tokens(self, input: str) -> int:
		pass
