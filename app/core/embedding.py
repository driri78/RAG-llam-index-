from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext
from app.loaders.pdf_loader import pdf_loader
from config import get_vector_store


# model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
# 페이지 단위로 청킹하기 때문에 필요x
# splitter = SentenceSplitter(
#     chunk_size=800,
#     chunk_overlap=100
# )
Settings.embed_model = HuggingFaceEmbedding(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)
vector_store = get_vector_store()

storage_context = StorageContext.from_defaults(vector_store=vector_store)


def build_rag_index():

    docs = pdf_loader()
    print("문서 개수: ", len(docs))
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    return index


# 1회용 => 벡터db 저장
# build_rag_index()
