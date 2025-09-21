import json
from typing import cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai.types.completion_create_params import CompletionCreateParams
from pydantic import TypeAdapter, ValidationError

from laime.backends.abc import TextGenerationBackend
from laime.models import get_backend

openai_text_generation_router = APIRouter()

completion_create_params_ta: TypeAdapter[CompletionCreateParams] = TypeAdapter(
	CompletionCreateParams
)


@openai_text_generation_router.post("/v1/completions")
async def create_completion(req: Request):
	try:
		body: CompletionCreateParams = completion_create_params_ta.validate_python(
			await req.json()
		)
	except ValidationError:
		raise HTTPException(
			status_code=422, detail={"message": "Unable to handle your inputs."}
		)

	backend = get_backend(body["model"], "text generation", TextGenerationBackend)

	async def stream():
		try:
			async for completion in backend.completion(
				prompt=body.get("prompt"),
				frequency_penalty=body.get("frequency_penalty"),
				logit_bias=body.get("logit_bias"),
				presence_penalty=body.get("presence_penalty"),
				seed=body.get("seed"),
				stop=body.get("stop"),
				temperature=body.get("temperature"),
				top_p=body.get("top_p"),
			):
				yield f"data: {completion.to_json(indent=None)}\n\n"
		except HTTPException as ex:
			yield f"data: {
				json.dumps(
					{
						'error': {
							'type': 'server_error',
							'code': ex.status_code,
							'message': cast(dict[str, object], cast(object, ex.detail))[
								'message'
							],
						}
					},
					indent=None,
				)
			}\n\n"
		finally:
			yield "data: [DONE]\n\n"

	if not body.get("stream", False):
		raise HTTPException(
			status_code=422,
			detail={
				"message": "laime currently does not support non-streaming responses."
			},
		)
	else:
		return StreamingResponse(stream(), media_type="text/event-stream")
