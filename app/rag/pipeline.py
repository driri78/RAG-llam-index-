from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from app.core.config import get_vector_store


def load_rag_index():

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    )

    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)


index = load_rag_index()

q = "사업관리자 초기 설정 방법"
retriever = index.as_retriever()
results = retriever.retrieve(q)

for r in results:
    print(r.node.text)
