pip list : 설치 패키지 리스트

-docker postgrepsql, pgvector
docker run -d \
  --name pgvector14 \
  -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg14


RAG langchain 패키지(조립형 프레임 워크)
pip install \
  langchain \
  langchain-community \
  langchain-openai \
  psycopg2-binary \
  pypdf \
  openai

llamaIndex 패키지(통합 프레임 워크)
pip install llama-index \
  llama-index-embeddings-openai \
  -- openai 대신 무료 임베딩 라이브러리
  llama-index-embeddings-huggingface \
  llama-index-vector-stores-postgres \
  llama-index-llms-huggingface
  psycopg2-binary \
  pypdf


무료 임베딩 모델(삭제 예정) => open ai로 변경 예정
langchain 시 사용
llamaindex에 경우 embeddings-huggingface로 설정
pip install -U sentence-transformers
https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS

pdf 죄표 정보 패키지
pip install pdfplumber

환경변수 조회 패키지
pip install python-dotenv

word 문서 load시 필요
pip install docx2txt

fastAPI
  1. python -m venv venv: 가상환경
  1-1. source venv/Scripts/activate: 가상환경 활성화 <=> deactivate: 바활성화
  2. pip install fastapi uvicorn: 필수 패키지 설차
  3. uvicorn app.main:app --reload: 서버 실행

파이썬 패키지 모드
  1. __init__.py 폴더를 하나의 패키지로 인식하기 위해선 
     python -m 패키지.모듈 로 실행해줘야함

프로젝트 의존성 패키지 목록 관리 파일 생성 => 최신화 하려면 다시 실행하면됨
pip freeze > requirements.txt

포에트리 사용시 requirements 생성
poetry export -f requirements.txt --output requirements.txt

git clone 시
pip install -r requirements.txt로 모듈 한번에 설치

.ipynb에서 파이썬 가상환경을 실행하기 위한 패키지 => colab으로 실행할경우 필요x
pip install ipykernel
