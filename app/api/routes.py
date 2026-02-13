from fastapi import APIRouter
from app.api.schemas import QueryRequest
from app.service.query_service import query

router = APIRouter()


@router.get("/query")
def query_api(question: str):
    answer = query(question)
    return {"answer": answer}
