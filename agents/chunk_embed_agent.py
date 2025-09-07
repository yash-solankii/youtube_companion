from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional, Any

from config import config
from utils.logger import get_logger, timer
from utils.cache import cache_embeddings, get_cached_embeddings

logger = get_logger("chunk_embed_agent")

class ChunkEmbedAgent:
    """Simple chunk embedding agent for processing video transcripts"""
    
    def __init__(self):
        # Text splitter for breaking up long content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
        
        # Embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_transcript_intelligently(self, transcript: List[str], video_url: str = None):
        """Create embeddings for video transcript"""
        timer.start("embed_transcript")
        logger.info("Creating embeddings for transcript")
        
        if not transcript:
            logger.warning("Empty transcript provided")
            return None
        
        try:
            # Check cache first
            if video_url:
                cached = get_cached_embeddings(video_url)
                if cached:
                    logger.info("Retrieved embeddings from cache")
                    timer.end("embed_transcript", {"source": "cache"})
                    return cached
            
            # Join transcript and create chunks
            full_text = "\n".join(transcript)
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Convert to documents
            documents = []
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={'chunk_id': i, 'source': 'youtube_transcript'}
                )
                documents.append(doc)
            
            # Create vector store
            logger.info(f"Creating embeddings for {len(documents)} chunks")
            vector_store = FAISS.from_documents(documents, self.embeddings_model)
            
            # Save to cache
            if video_url:
                cache_embeddings(video_url, vector_store)
                logger.info("Embeddings cached")
            
            logger.info(f"Embeddings created: {len(documents)} chunks")
            timer.end("embed_transcript", {"chunks": len(documents)})
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            timer.end("embed_transcript", {"error": str(e)})
            return None

# Create global instance
chunk_embed_agent = ChunkEmbedAgent()