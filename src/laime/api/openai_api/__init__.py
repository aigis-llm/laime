from fastapi import APIRouter

from laime.api.openai_api.embeddings import openai_embeddings_router
from laime.api.openai_api.text_generation import openai_text_generation_router

openai_router = APIRouter(prefix="/openai")

openai_router.include_router(openai_embeddings_router)
openai_router.include_router(openai_text_generation_router)
