from app.rag import pipeline


def query(question: str):

    return pipeline.rag_pipeline.query(question)


def retrieve(question):

    results = pipeline.rag_pipeline.retrieve(question)

    for r in results:
        print(r.node.text)
