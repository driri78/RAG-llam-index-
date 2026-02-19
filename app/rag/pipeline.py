from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from app.core.config import get_vector_store


Settings.embed_model = HuggingFaceEmbedding(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)


def init_rag():
    global retriever

    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever()


def retrieve(question):

    results = retriever.retrieve(question)

    for r in results:
        print(r.node.text)
