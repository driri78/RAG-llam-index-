from app.rag.pipeline import load_rag_index
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.2},
)
Settings.llm = llm


def query(question: str):

    index = load_rag_index()

    query_engine = index.as_query_engine()

    response = query_engine.query(question)

    return response
