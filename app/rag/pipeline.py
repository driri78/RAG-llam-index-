from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from app.core.config import get_vector_store
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

# 임베딩 모델 세팅
Settings.embed_model = HuggingFaceEmbedding(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

# llm 모델 세팅
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.2},
)


class RAGPipeline:
    def __init__(self):
        vector_store = get_vector_store()
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        self.retriever = self.index.as_retriever()
        self.query_engine = self.index.as_query_engine()

    def query(self, question):
        return self.query_engine.query(question)

    def retrieve(self, question):
        return self.retriever.retrieve(question)


rag_pipeline = None


def init_rag():
    global rag_pipeline
    rag_pipeline = RAGPipeline()
