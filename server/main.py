from fastapi import FastAPI

app = FastAPI(title="RAG")

@app.get("/test")
async def test():
    return {"message": "RAG server is up and running!"}