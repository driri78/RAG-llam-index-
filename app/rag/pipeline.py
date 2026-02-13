from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from app.core.config import get_vector_store


Settings.embed_model = HuggingFaceEmbedding(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)


def load_rag_index():

    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)


def retrieve(question):
    index = load_rag_index()

    retriever = index.as_retriever()
    results = retriever.retrieve(question)

    for r in results:
        print(r.node.text)
