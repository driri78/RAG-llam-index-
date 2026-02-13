from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5433/postgres"
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1;"))
    print("DB 연결 성공:", result.scalar())

# %%writefile app/core/config.py
# from llama_index.vector_stores.postgres import PGVectorStore


# def get_vector_store():
#     vector_store = PGVectorStore.from_params(
#    
#     )

#     return vector_store

# 세션이 유지될 동안에만 존재하므로 세션 종료시 다시 실행, 설치 필요
# 런타임 재시작 / GPU 재할당 / 세션 끊김 / Colab 새 접속 => /content 초기화
# !git clone https://github.com/driri78/RAG-llam-index-.git
# %cd /content/RAG-llam-index-
# !pip install llama-index
# !pip install llama-index-llms-huggingface
# !pip install llama-index-embeddings-huggingface
# !pip install llama-index-vector-stores-postgres
# !pip install transformers accelerate torch


# ip 확인 supabase에 등록하기 위함(등록 안하면 db접속 안됨)
# ip는 colab 세션마다 달라짐 => 개발시 Allow all access 설정(나중에 해제)
# !curl ifconfig.me



# gpu 확인
# 메모리 상태
# !nvidia-smi
# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("GPU 개수:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     print("GPU 이름:", torch.cuda.get_device_name(0))

!ls
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

from app.rag.pipeline import load_rag_index
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.2}
)
Settings.llm = llm

index = load_rag_index()

query_engine = index.as_query_engine()

while True:
    question = input("질문을 입력하세요. (stop 입력시 멈춤)")
    if(question == "stop"): break
    
    response = query_engine.query(question)
    print(str(response))
