from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from typing import Tuple, List

from config import config
from utils.logger import get_logger, timer
from utils.cache import cache_summary, get_cached_summary
from utils.rate_limiter import get_rate_limiter
from utils.model_fallback import create_llm_with_fallback, get_model_for_task

logger = get_logger("summarizer_agent")

# LLM setup with fallback support - increased tokens for detailed summaries
llm = create_llm_with_fallback(
    model_name=get_model_for_task("summary"),
    temperature=0.3,
    max_tokens=3000,  # Increased for comprehensive summaries
)

# Text splitter for breaking up long content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len
)

# Enhanced prompts for comprehensive in-depth analysis
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Create a comprehensive, detailed summary of this video content. Your summary should be thorough and informative, covering:

1. **Main Topic & Purpose**: What is the video about and why was it created?
2. **Key Concepts & Ideas**: What are the main concepts, theories, or ideas discussed?
3. **Detailed Explanations**: How are these concepts explained? What examples or analogies are used?
4. **Important Details**: What specific facts, statistics, or details are shared?
5. **Step-by-Step Processes**: If applicable, what processes or methods are described?
6. **Insights & Analysis**: What insights, analysis, or unique perspectives are offered?
7. **Practical Applications**: How can this information be applied in real life?
8. **Key Takeaways**: What are the most important points viewers should remember?
9. **Context & Background**: What background information or context is provided?
10. **Value Proposition**: What value does this video provide to viewers?

Write a comprehensive summary that thoroughly covers the video content. Aim for 8-12 detailed sentences that provide a complete understanding of the video's content, insights, and value. Be specific and include important details, examples, and explanations.

Video content:
{text}

Comprehensive Summary:"""
)

BULLET_PROMPT = PromptTemplate(
    input_variables=["summary"],
    template="""Create 6-8 comprehensive key points from this summary. Each bullet point should:

1. Be specific and detailed, not generic
2. Capture important concepts, insights, or lessons
3. Include relevant details, examples, or statistics when mentioned
4. Be actionable or informative
5. Cover different aspects of the content (concepts, processes, insights, applications)
6. Start with '•' and be 2-3 sentences long for thorough coverage

Make sure to extract the most valuable and comprehensive points from the summary.

Summary:
{summary}

Key Points:
• """
)

def create_chunks(text: str) -> List[Document]:
    """Break text into smaller pieces for processing"""
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

def generate_summary_and_bullets(transcript: List[str], video_url: str = None) -> Tuple[str, str]:
    """Create summary and key points from video transcript"""
    timer.start("generate_summary")
    logger.info("Starting summary generation")

    # Check rate limiter status
    rate_limiter = get_rate_limiter()
    stats = rate_limiter.get_stats()
    logger.info(f"Rate limiter: {stats['current_requests']}/{stats['max_requests']} rpm ({stats['request_utilization']:.1f}% utilization)")

    if not transcript:
        logger.warning("Empty transcript provided")
        return "Transcript was empty.", "No key points available."

    # Check cache first
    if video_url:
        cached = get_cached_summary(video_url)
        if cached and is_valid_content(cached["summary"], cached["bullets"]):
            logger.info("Using cached summary")
            timer.end("generate_summary", {"source": "cache"})
            return cached["summary"], cached["bullets"]

    try:
        # Join transcript and create chunks
        full_text = "\n".join(transcript)
        text_length = len(full_text)
        logger.info(f"Processing {len(transcript)} segments, {text_length} characters")
        
        # Simple processing logic
        summary = create_summary(full_text)

        # Final safety check - ensure we always have a valid summary
        if not is_valid_summary(summary):
            logger.warning(f"Generated summary failed validation, using fallback: '{summary[:100]}'")
            summary = extract_meaningful_fallback(full_text)
        
        # Generate bullet points
        bullets = generate_bullets(summary, full_text)
        
        # Cache results
        if video_url:
            cache_summary(video_url, summary, bullets)
        
        logger.info("Summary generation completed successfully")
        timer.end("generate_summary", {
            "source": "llm",
            "summary_length": len(summary),
            "bullets_length": len(bullets),
            "total_length": text_length
        })
        
        return summary, bullets
        
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        timer.end("generate_summary", {"error": str(e)})
        
        # Production-ready fallback
        full_text = "\n".join(transcript)
        fallback_summary = extract_meaningful_fallback(full_text)
        fallback_bullets = create_manual_bullets(fallback_summary)
        
        return fallback_summary, fallback_bullets

def create_summary(text: str) -> str:
    """Create summary with smart text selection"""
    try:
        # For long text, take beginning + middle + end
        if len(text) > 6000:
            # Take 1500 chars from beginning, middle, and end
            chunk_size = 1500
            beginning = text[:chunk_size]
            middle = text[len(text)//2 - chunk_size//2:len(text)//2 + chunk_size//2]
            end = text[-chunk_size:]
            content = f"{beginning}\n\n{middle}\n\n{end}"
        else:
            content = text

        # Create summary
        docs = [Document(page_content=content)]
        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff", 
            prompt=SUMMARY_PROMPT,
            verbose=False
        )

        result = get_rate_limiter().execute_with_rate_limit(
            chain.invoke, {"input_documents": docs}
        )
        summary = result.get("output_text", "").strip()

        if is_valid_summary(summary):
            return summary

        # Fallback: direct prompt
        fallback_prompt = f"Summarize this video: {content[:2000]}"
        fallback_result = get_rate_limiter().execute_with_rate_limit(llm.invoke, fallback_prompt)
        fallback_summary = fallback_result.content.strip() if hasattr(fallback_result, 'content') else str(fallback_result).strip()
        
        return fallback_summary if is_valid_summary(fallback_summary) else extract_meaningful_fallback(text)

    except Exception as e:
        logger.warning(f"Summary creation failed: {e}")
        return extract_meaningful_fallback(text)


def extract_meaningful_fallback(text: str) -> str:
    """Simple fallback when AI fails"""
    char_count = len(text)
    if char_count > 10000:
        return f"This video covers multiple topics in detail with over {char_count//1000}k characters of content."
    else:
        return f"This video discusses important topics with {char_count} characters of transcript."

def extract_bullet_points(text: str) -> str:
    """Clean up bullet points from AI response"""
    if not text:
        return ""

    lines = text.split('\n')
    bullets = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Keep lines that look like bullets
        if line.startswith(('•', '-', '*')) or line[0].isdigit() and line[1:3] == '. ':
            # Convert numbered to bullets
            if line[0].isdigit() and line[1:3] == '. ':
                line = '• ' + line[2:].strip()
            bullets.append(line)

    return '\n'.join(bullets[:8])  # Max 8 bullets

def generate_bullets(summary: str, full_text: str = None) -> str:
    """Create bullet points from the summary"""
    try:
        # Use summary, or full text if summary is too short
        content = summary if len(summary) > 100 else (full_text[:2000] if full_text else summary)

        prompt = BULLET_PROMPT.format(summary=content[:1500])

        result = get_rate_limiter().execute_with_rate_limit(llm.invoke, prompt)
        bullets = result.content.strip()

        cleaned_bullets = extract_bullet_points(bullets)

        if is_valid_bullets(cleaned_bullets):
            return cleaned_bullets

        # Try simpler prompt
        simple_prompt = f"Create 6-8 bullet points:\n\n{content[:1000]}\n\n• "
        result = get_rate_limiter().execute_with_rate_limit(llm.invoke, simple_prompt)
        bullets = result.content.strip()
        cleaned_bullets = extract_bullet_points(bullets)

        if is_valid_bullets(cleaned_bullets):
            return cleaned_bullets

    except Exception as e:
        logger.warning(f"Bullet generation failed: {e}")

    # Simple fallback
    return create_manual_bullets(content if len(content) > 50 else summary)

def create_manual_bullets(summary: str) -> str:
    """Create simple bullets when AI fails"""
    if not summary or len(summary) < 50:
        return "• Video processed\n• Content available\n• Ready for questions"

    # Extract sentences as bullets
    sentences = summary.split('. ')[:8]
    bullets = []

    for sentence in sentences:
        if sentence.strip() and len(sentence) > 10:
            bullets.append(f"• {sentence.strip().rstrip('.')}")

    if len(bullets) < 2:
        return "• Video content processed\n• Key points extracted\n• Ready for questions"

    return '\n'.join(bullets)

def is_valid_content(summary: str, bullets: str) -> bool:
    """Check if cached content looks good"""
    return is_valid_summary(summary) and is_valid_bullets(bullets)

def is_valid_summary(summary: str) -> bool:
    """Check if summary is good enough"""
    if not summary or len(summary.strip()) < 30:
        return False
    return True

def is_valid_bullets(bullets: str) -> bool:
    """Check if bullets are good enough"""
    if not bullets or len(bullets.strip()) < 50:
        return False
    lines = [line.strip() for line in bullets.split('\n') if line.strip()]
    bullet_lines = [line for line in lines if line.startswith(('•', '-', '*'))]
    return len(bullet_lines) >= 4  # Require at least 4 bullet points