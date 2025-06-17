from typing import override

from pydantic import BaseModel

from laime.backends.abc import Backend


class NoneBackendConfig(BaseModel):
	pass


class NoneBackend(Backend[NoneBackendConfig]):
	backend_name: str = "none"
	backend_config_model = NoneBackendConfig  # pyright: ignore [reportUnannotatedClassAttribute]

	@override
	def __init__(self, config: NoneBackendConfig) -> None:
		pass
