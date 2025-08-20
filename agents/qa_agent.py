from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Custom QA prompt (context-limited)
QA_PROMPT_TEMPLATE = """You are a helpful AI assistant answering questions about a specific YouTube video.
Your ONLY source of information is the provided "Context from video" below.
ABSOLUTELY DO NOT use any of your prior knowledge or information outside of this provided context.
If the answer cannot be found within the "Context from video", you MUST explicitly state "Based on the provided video segments, the information to answer that question is not available."
Do not attempt to infer or guess.

Context from video:
---------------------
{context}
---------------------

Question: {question}

Answer (based ONLY on the "Context from video" above):"""
QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Prompt for question rephrasing (for chat continuity)
CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the follow up question is already a standalone question, just return it as is.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

def get_qa_chain(vector_store):
    if vector_store is None:
        print("Error (get_qa_chain): Vector store is None. Cannot create retriever.")
        return None

    # Initialize Groq LLM
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.5,
        max_tokens=1024
    )
    
    # MMR retriever for better relevance diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 7, 'fetch_k': 50}
    )
    
    # Full QA pipeline with chat support
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    
    print("QA chain created.")
    return qa_chain
