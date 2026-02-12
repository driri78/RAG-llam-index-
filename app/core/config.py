from llama_index.vector_stores.postgres import PGVectorStore


def get_vector_store():
    vector_store = PGVectorStore.from_params(
        database="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5433,
        table_name="chunks",
        embed_dim=768,
    )

    return vector_store
