from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from typing import Optional, Dict, Any, List
import logging

from config import config
from utils.security import security
from utils.logger import get_logger, timer
from utils.model_fallback import create_llm_with_fallback, get_model_for_task

logger = get_logger("qa_agent")

# Comprehensive QA prompt that handles all cases
QA_PROMPT_TEMPLATE = """You are an expert AI assistant helping users understand video content. Analyze the user's question and the provided video content.

INSTRUCTIONS:
1. If the question IS related to the video content:
   - Use the video information to provide a comprehensive, detailed answer
   - Reference specific points from the video when possible
   - Be conversational and helpful in your response
   - Supplement with general knowledge only when it adds value

2. If the question is NOT related to the video content:
   - Answer using your general knowledge
   - Be direct, accurate, and thorough
   - Do NOT mention the video or try to force connections

3. For follow-up questions or clarifications:
   - Maintain context from previous interactions
   - Build upon earlier answers when relevant
   - Be concise but informative

4. General guidelines:
   - Be conversational and natural in your responses
   - Avoid overly technical jargon unless the user is asking for it
   - If something is unclear, ask for clarification
   - Stay focused on helping the user understand the video content

Video content:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Prompt for handling follow-up questions
CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation and a follow-up input, rephrase the follow-up input to be a standalone question, in its original language.

INSTRUCTIONS:
- If the follow-up input is already a complete, standalone question, return it as is
- If it's a follow-up or clarification, rephrase it to be self-contained while maintaining the context
- If it's a conversational acknowledgment (like "cool", "thanks", "got it"), return it as is
- Keep the original language and tone of the user's input
- Focus on making the question clear and complete for answering

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)


def calculate_relevance(question: str, chunk: str) -> float:
    """Simple relevance score between question and chunk"""
    q_words = set(question.lower().split())
    c_words = set(chunk.lower().split())

    if not q_words:
        return 0.0

    # Simple overlap calculation
    overlap = len(q_words.intersection(c_words))
    return min(1.0, overlap / len(q_words))


def get_qa_chain(vector_store):
    """Create a QA chain for answering questions about the video"""
    if vector_store is None:
        return None

    try:
        # Set up the language model with fallback support
        llm = create_llm_with_fallback(
            model_name=get_model_for_task("qa"),
            temperature=0.0,
            max_tokens=4096,
        )
        
        # Create retriever for finding relevant text chunks
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 8, 'fetch_k': 50}  # Reduced for token limit compliance
        )
        
        # Build the QA chain with token-aware settings
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            max_tokens_limit=4000,  # Ensure we stay within Groq limits
        )
        
        logger.info(f"QA chain created successfully with model: {get_model_for_task('qa')}")
        return qa_chain
        
    except Exception as e:
        print(f"Failed to create QA chain: {e}")
        return None

def assess_answer_quality(question: str, answer: str, source_documents: list) -> Dict[str, Any]:
    """Check if answer needs fallback"""
    if not source_documents:
        return {
            "needs_fallback": True,
            "quality_score": 0,
            "indicators": {
                "content_relevance_score": 0,
                "avg_source_relevance": 0
            }
        }

    # Simple relevance check
    all_content = " ".join([doc.page_content for doc in source_documents])
    relevance = calculate_relevance(question, all_content)

    return {
        "needs_fallback": relevance < 0.1,
        "quality_score": relevance,
        "indicators": {
            "content_relevance_score": relevance,
            "avg_source_relevance": relevance
        }
    }

def is_conversational_message(message: str) -> tuple[bool, str]:
    """Check if message is just conversational (not a question)"""
    msg = message.lower().strip()

    # Simple check for common acknowledgments
    if msg in ["cool", "thanks", "thank you", "got it", "ok", "okay", "nice", "great", "good"]:
        return True, "Glad I could help! Ask me anything else about the video."

    # Very short messages without question words
    if len(msg.split()) <= 2 and not any(word in msg for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'should', 'would', 'tell']):
        return True, "Happy to help! Feel free to ask anything else about the video."

    return False, ""

def process_question(question: str, qa_chain, chat_history: list, vector_store) -> Dict[str, Any]:
    """Process a user question"""
    timer.start("process_question")
    
    if not qa_chain or not vector_store:
        timer.end("process_question", {"error": "system_not_ready"})
        return {
            "answer": "Q&A system not ready. Please load a video first.",
            "source_documents": [],
            "error": "system_not_ready"
        }

    try:
        # Check if this is a conversational acknowledgment
        is_conversational, conversational_response = is_conversational_message(question)
        if is_conversational:
            logger.info(f"Detected conversational message: '{question}'")
            timer.end("process_question", {
                "question_length": len(question),
                "answer_length": len(conversational_response),
                "conversational": True
            })
            return {
                "answer": conversational_response,
                "source_documents": [],
                "conversational": True,
                "quality_score": 1.0,
                "content_relevance": 1.0
            }

        # Basic security check
        is_valid, message = security.validate_question(question)
        if not is_valid:
            timer.end("process_question", {"error": "security_validation_failed"})
            return {
                "answer": "Your question contains inappropriate content.",
                "source_documents": [],
                "error": "security_validation_failed"
            }
        
        cleaned_question = security.clean_input(question)
        
        # Rate limiting
        rate_ok, rate_message = security.check_rate_limit("qa_requests")
        if not rate_ok:
            timer.end("process_question", {"error": "rate_limit_exceeded"})
            return {
                "answer": rate_message,
                "source_documents": [],
                "error": "rate_limit_exceeded"
            }
        
        # Convert chat history to LangChain format
        langchain_history = []
        
        # Handle different chat history formats
        if chat_history:
            user_msg = None
            for entry in chat_history:
                if isinstance(entry, dict) and 'role' in entry:
                    if entry['role'] == 'user':
                        user_msg = entry['content']
                    elif entry['role'] == 'assistant' and user_msg:
                        langchain_history.append((user_msg, entry['content']))
                        user_msg = None
        
        # Get answer from QA chain
        try:
            result = qa_chain.invoke({
                "question": cleaned_question,
                "chat_history": langchain_history
            })
            
            answer = result.get("answer", "Sorry, I couldn't find an answer in the video content.")
            source_documents = result.get("source_documents", [])
            
        except Exception as qa_error:
            error_str = str(qa_error)
            logger.warning(f"QA chain failed: {error_str}")
            
            # Handle token limit errors specifically
            if "413" in error_str or "too large" in error_str.lower() or "token" in error_str.lower():
                logger.info("Token limit exceeded, using fallback approach")
                
                # Try with smaller context
                try:
                    # Get just the most relevant chunk
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': 1, 'fetch_k': 10}
                    )
                    
                    docs = retriever.get_relevant_documents(cleaned_question)
                    if docs:
                        context = docs[0].page_content[:1500]  # Limit context size
                        prompt = f"Based on this context, answer: {cleaned_question}\n\nContext: {context}"
                        
                        if hasattr(qa_chain, 'llm'):
                            llm = qa_chain.llm
                            fallback_result = llm.invoke(prompt)
                            answer = fallback_result.content.strip() if hasattr(fallback_result, "content") else str(fallback_result).strip()
                            source_documents = docs
                        else:
                            answer = "Sorry, I couldn't process your question due to technical limitations."
                            source_documents = []
                    else:
                        answer = "Sorry, I couldn't find relevant information in the video for your question."
                        source_documents = []
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback approach also failed: {fallback_error}")
                    answer = "Sorry, I couldn't process your question. Please try a simpler question or reload the video."
                    source_documents = []
            else:
                # Other errors
                answer = "Sorry, I couldn't find an answer in the video content."
                source_documents = []
        
        # Simple quality check
        quality_assessment = assess_answer_quality(cleaned_question, answer, source_documents)
        
        # Log whether fallback knowledge was needed
        content_relevance = quality_assessment['indicators']['content_relevance_score']
        if quality_assessment["needs_fallback"]:
            logger.info(f"Fallback knowledge needed - content relevance: {content_relevance:.3f} (below threshold)")
        else:
            if content_relevance < 0.2:
                logger.info(f"Video content may be insufficient - content relevance: {content_relevance:.3f} (low)")
            else:
                logger.info(f"Video content sufficient - content relevance: {content_relevance:.3f} (good)")
        
        # Only retry for clear errors
        if quality_assessment["needs_fallback"]:
            try:
                if hasattr(qa_chain, 'llm'):
                    llm = qa_chain.llm
                    retry_result = llm.invoke(f"Answer this question: {cleaned_question}")
                    retry_answer = retry_result.content.strip() if hasattr(retry_result, "content") else str(retry_result).strip()
                    
                    if len(retry_answer) > len(answer):
                        answer = retry_answer
                
            except:
                pass  # Keep original answer
        
        # Clean the answer without truncating
        cleaned_answer = security.clean_output(answer)
        
        timer.end("process_question", {
            "question_length": len(question),
            "answer_length": len(cleaned_answer),
            "has_error": False,
            "needs_fallback": quality_assessment["needs_fallback"],
            "quality_score": quality_assessment["quality_score"],
            "sources": len(source_documents),
            "content_relevance": quality_assessment["indicators"].get("content_relevance_score", 0),
            "avg_source_relevance": quality_assessment["indicators"].get("avg_source_relevance", 0),
            "fallback_used": quality_assessment["needs_fallback"]
        })
        
        return {
            "answer": cleaned_answer,
            "source_documents": source_documents,
            "error": None,
            "quality_assessment": quality_assessment
        }
        
    except Exception as e:
        timer.end("process_question", {"error": str(e)})
        return {
            "answer": "Sorry, there was an error processing your question. Please try again.",
            "source_documents": [],
            "error": str(e)
        }
