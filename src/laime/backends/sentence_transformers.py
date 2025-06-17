from typing import override

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from laime.backends.abc import EmbeddingsBackend


class SentenceTransformersBackendConfig(BaseModel):
	model: str
	device: str


class SentenceTransformersBackend(EmbeddingsBackend[SentenceTransformersBackendConfig]):
	backend_name: str = "sentence_transformers"
	backend_config_model = SentenceTransformersBackendConfig  # pyright: ignore [reportUnannotatedClassAttribute]
	model: SentenceTransformer
	tokenizer: PreTrainedTokenizer

	@override
	def __init__(self, config: SentenceTransformersBackendConfig) -> None:
		self.model = SentenceTransformer(config.model, device="cpu")
		self.tokenizer = AutoTokenizer.from_pretrained(  # pyright: ignore [reportUnknownMemberType]
			config.model, device=config.device
		)

	@override
	async def embed(self, input: str) -> list[float]:
		return self.model.encode(input, device="cpu").tolist()  # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]

	@override
	async def count_tokens(self, input: str) -> int:
		return len(self.tokenizer.tokenize(input))  # pyright: ignore [reportUnknownMemberType]
