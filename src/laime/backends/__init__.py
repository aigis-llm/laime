from laime.backends.none import NoneBackend
from laime.backends.sentence_transformers import SentenceTransformersBackend

backends = {
	NoneBackend.backend_name: NoneBackend,
	SentenceTransformersBackend.backend_name: SentenceTransformersBackend,
}
