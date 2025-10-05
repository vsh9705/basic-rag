import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from logger import logger

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


MODERATION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI content moderation system. Your task is to determine "
        "if the given text violates any of the company’s policy documents.\n\n"
        "=== POLICY CONTEXT ===\n"
        "{context}\n\n"
        "=== TEXT TO CHECK ===\n"
        "{question}\n\n"
        "Please respond STRICTLY in one of the following two formats:\n"
        "1. 'VIOLATION: <brief explanation of which policy it violates and why>'\n"
        "2. 'REVIEW: <brief explanation of why it needs human review>'\n"
        "3. 'OK: <brief reason why it is compliant>'\n\n"
        "Be concise but explicit in your reasoning."
    )
)

def get_retrieval_qa_chain(vectorstore, k: int = 3, chain_type: str = "stuff"):
    """
    Return a RetrievalQA chain using Groq LLM and the provided vectorstore retriever.
    The LLM will follow the custom moderation prompt above.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=512
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        chain_type_kwargs={"prompt": MODERATION_PROMPT},
        return_source_documents=True
    )

    logger.info("RetrievalQA moderation chain initialized with custom prompt.")
    return chain


def query_chain(chain, user_input: str):
    """
    Run a moderation query against the RetrievalQA chain and return a formatted response.
    """
    try:
        logger.debug(f"Running moderation chain for input: {user_input[:200]}...")
        result = chain({"query": user_input})
        response = {
            "response": result["result"],
            "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
        }
        logger.debug(f"Chain response: {response}")
        return response
    except Exception as e:
        logger.exception("Error in query_chain")
        raise
