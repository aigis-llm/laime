from fastapi import APIRouter, HTTPException, Request
from openai.types import CreateEmbeddingResponse, Embedding, EmbeddingCreateParams
from openai.types.create_embedding_response import Usage
from pydantic import TypeAdapter, ValidationError

from laime.backends.abc import EmbeddingsBackend
from laime.models import get_models, model_backends
from laime.typeguards import is_str_list

openai_embeddings_router = APIRouter()

embedding_create_params_ta = TypeAdapter(EmbeddingCreateParams)


@openai_embeddings_router.post("/v1/embeddings")
async def create_embeddings(req: Request):
	models = get_models()
	try:
		body = embedding_create_params_ta.validate_python(await req.json())
	except ValidationError:
		raise HTTPException(
			status_code=422, detail={"message": "Unable to handle your inputs."}
		)
	backend = None

	if not model_backends.get(body["model"]):
		model = models.get(body["model"])
		if model is None:
			raise HTTPException(
				status_code=422, detail={"message": f"No model {body['model']}."}
			)
		if not issubclass(model[0], EmbeddingsBackend):
			raise HTTPException(
				status_code=422,
				detail={"message": f"{body['model']} is not an embedding model."},
			)
		backend = model[0](model[1])
		model_backends[body["model"]] = backend
	else:
		backend = model_backends[body["model"]]
		if not isinstance(backend, EmbeddingsBackend):
			raise HTTPException(
				status_code=422,
				detail={"message": f"{body['model']} is not an embedding model."},
			)

	embedding_data: list[Embedding] = []
	usage = 0

	if isinstance(body["input"], str):
		usage += await backend.count_tokens(body["input"])
		embedding_data.append(
			Embedding(
				index=0,
				embedding=(await backend.embed(body["input"])),
				object="embedding",
			)
		)
	elif isinstance(body["input"], list) and is_str_list(body["input"], False):
		for index, input in enumerate(body["input"]):
			usage += await backend.count_tokens(input)
			embedding_data.append(
				Embedding(
					index=index,
					embedding=(await backend.embed(input)),
					object="embedding",
				)
			)
	else:
		raise HTTPException(
			status_code=422,
			detail={"message": "laime is unable to handle non-string inputs."},
		)

	return CreateEmbeddingResponse(
		model=body["model"],
		data=embedding_data,
		usage=Usage(prompt_tokens=usage, total_tokens=usage),
		object="list",
	)
