from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5433/postgres"
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1;"))
    print("DB 연결 성공:", result.scalar())
