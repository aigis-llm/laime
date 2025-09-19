from laime.backends.none import NoneBackend
from laime.backends.openai_proxy import OpenAIProxyBackend
from laime.backends.sentence_transformers import SentenceTransformersBackend

backends = {
	NoneBackend.backend_name: NoneBackend,
	SentenceTransformersBackend.backend_name: SentenceTransformersBackend,
	OpenAIProxyBackend.backend_name: OpenAIProxyBackend,
}
