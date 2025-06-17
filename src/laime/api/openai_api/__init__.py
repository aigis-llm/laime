from fastapi import APIRouter

from laime.api.openai_api.embeddings import openai_embeddings_router

openai_router = APIRouter(prefix="/openai")

openai_router.include_router(openai_embeddings_router)
