# agents/chunk_embed_agent.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import torch
import gradio as gr


def embed_transcript(transcript_utterances: list[str]):
    print("--- embed_transcript (for Q&A) CALLED ---")
    if not transcript_utterances or not any(s.strip() for s in transcript_utterances):
        gr.Warning("Embedder: Received empty raw transcript utterances.")
        return None

    # Combine utterances into a single text blob for coherent splitting
    full_text = "\n".join([u.strip() for u in transcript_utterances if u.strip()])

    # Character-based splitting: ~1000 chars per chunk with 200-char overlap
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(full_text)

    docs_to_embed = []
    for idx, chunk in enumerate(chunks):
        docs_to_embed.append(Document(page_content=chunk, metadata={"chunk_id": idx}))
    print(f"Embedder: Prepared {len(docs_to_embed)} chunks for embedding.")

    # Initialize the embedder
    model_name = "BAAI/bge-base-en-v1.5"
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={}
    )
    print(f"Embedder: Embedding {len(docs_to_embed)} chunks...")
    vectorstore = FAISS.from_documents(docs_to_embed, embedder)
    print("Embedder: Q&A Vector store created successfully.")
    return vectorstore