import pytest
from openai import AsyncOpenAI

from laime.testclient import async_test_client


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_completion_create():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	token_count = 0

	async for chunk in await openai.completions.create(
		model="qwen3-0.6b",
		prompt="Say this is a test",
		temperature=0,
		max_tokens=1,
		stream=True,
	):
		assert chunk.choices[0].text == " question"
		token_count = token_count + 1
		if token_count == 1:
			break


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_embedding_create():
	openai = AsyncOpenAI(
		base_url="http://laime_test/openai/v1",
		api_key="test",
		http_client=async_test_client(),
	)
	embedding = (
		(await openai.embeddings.create(model="qwen3-0.6b-embedding", input="a"))
		.data[0]
		.embedding
	)
	assert len(embedding) == 1024
