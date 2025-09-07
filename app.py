import gradio as gr
from agents.transcript_agent import get_transcript, extract_video_id
from agents.chunk_embed_agent import chunk_embed_agent
from agents.qa_agent import get_qa_chain, process_question
from agents.summarizer_agent import generate_summary_and_bullets
from typing import Tuple, Any

from utils.security import security
from utils.logger import get_logger, timer
from utils.cache import cache, clear_invalid_cache
from utils.rate_limiter import get_rate_limiter
from config import config

# Set up logging
logger = get_logger("main_app")

# Clear invalid cache on startup
try:
    clear_invalid_cache()
    logger.info("Invalid cache entries cleared on startup")
except Exception as e:
    logger.warning(f"Failed to clear invalid cache on startup: {e}")

def force_cache_refresh(video_url: str) -> str:
    """Force refresh of cached data for a specific video"""
    try:
        # Clear the specific video's cache
        key = cache._get_key(video_url)
        cache_file = cache._get_path("summaries", key)
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"Cleared cache for video: {video_url}")
        
        return "Cache cleared. Please reload the video to regenerate summary and bullets."
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return f"Failed to clear cache: {e}"

def clear_all_cache() -> str:
    """Clear all cache data to resolve rate limiting issues"""
    try:
        cleared_count = cache.clear_all()
        logger.info("All cache cleared")
        return f"All cache cleared successfully. {cleared_count} files removed."
    except Exception as e:
        logger.error(f"Failed to clear all cache: {e}")
        return f"Failed to clear all cache: {e}"

def load_video(video_url: str) -> Tuple[str, str, Any, Any, Any]:
    """Load and process a YouTube video"""
    timer.start("load_video")
    logger.info(f"Processing video: {video_url[:100]}...")
    
    if not video_url:
        gr.Warning("Please enter a YouTube URL.")
        return "Enter a URL to begin.", "", [], None, None

    # Reset chatbot - initialize with empty messages format
    chatbot_out = [] 
    
    try:
        # Security check
        is_valid, message = security.validate_youtube_url(video_url)
        if not is_valid:
            logger.warning(f"Security check failed: {video_url}")
            gr.Error(f"Security check failed: {message}")
            return f"Error: {message}", "Please provide a valid YouTube URL.", [], None, None
        
        # Rate limiting
        rate_ok, rate_message = security.check_rate_limit("video_loading")
        if not rate_ok:
            logger.warning("Rate limit exceeded for video loading")
            gr.Error(rate_message)
            return "Rate limit exceeded.", "Please wait before processing another video.", [], None, None
        
        gr.Info("1/5: Checking URL and getting transcript...")
        
        # Extract video ID
        try:
            video_id = extract_video_id(video_url)
            logger.info(f"Video ID: {video_id}")
        except Exception as e:
            logger.error(f"Failed to extract video ID: {e}")
            gr.Error("Invalid YouTube URL format.")
            return "Error: Invalid YouTube URL format.", "Please check the URL and try again.", [], None, None
        
        # Get transcript
        transcript_list = get_transcript(video_url)
        if not transcript_list or not any(s.strip() for s in transcript_list):
            gr.Error("Transcript is empty or unavailable for this video.")
            logger.warning("Empty transcript received")
            return "Error: Transcript is empty or unavailable.", "Try a different video.", [], None, None

        logger.info(f"Transcript ready: {len(transcript_list)} segments")
        
        gr.Info("2/5: Creating intelligent summary...")
        summary, bullets = generate_summary_and_bullets(transcript_list, video_url)
        
        # Ensure we have valid summary and bullets
        if not summary or summary.strip() == "":
            logger.warning("Summary generation returned empty result, using fallback")
            summary = "Summary could not be generated. Please try again or check the transcript."
        
        if not bullets or bullets.strip() == "":
            logger.warning("Bullet generation returned empty result, using fallback")
            bullets = "• Summary generated\n• Key points could not be extracted\n• Please review the summary above"
        
        logger.info(f"Intelligent summary created: {len(summary)} chars, bullets: {len(bullets)} chars")
        logger.debug(f"Summary preview: {summary[:200]}...")
        logger.debug(f"Bullets preview: {bullets[:200]}...")
        
        gr.Info("3/5: Preparing intelligent embeddings...")
        vector_index = chunk_embed_agent.embed_transcript_intelligently(transcript_list, video_url)

        if vector_index is None:
            logger.error("Failed to create vector index")
            gr.Warning("Failed to prepare Q&A system, but summary is available.")
            # Return summary and bullets even if Q&A fails
            return summary, bullets, chatbot_out, None, None
        
        gr.Info("4/5: Setting up Q&A...")
        qa_chain = get_qa_chain(vector_index)

        if qa_chain is None:
            logger.error("Failed to initialize QA chain")
            gr.Warning("Failed to set up Q&A system, but summary is available.")
            # Return summary and bullets even if Q&A fails
            return summary, bullets, chatbot_out, vector_index, None

        gr.Info("5/5: Ready to go!")
        logger.info("Video processing completed")
        logger.info(f"Final results - Summary: {len(summary)} chars, Bullets: {len(bullets)} chars, Q&A: {'Ready' if qa_chain else 'Failed'}")
        
        timer.end("load_video", {
            "transcript_segments": len(transcript_list),
            "summary_length": len(summary),
            "bullets_length": len(bullets)
        })
        
        return summary, bullets, chatbot_out, vector_index, qa_chain

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error during video loading: {error_message}")
        
        # Handle specific errors
        if "no element found" in error_message or "Could not retrieve transcript" in error_message:
            gr.Error("This video doesn't have a transcript available. Please try another one.")
            return "Error: No transcript found for this video.", "Please try a different YouTube video that has captions enabled.", [], None, None
        elif "Security check failed" in error_message:
            gr.Error(f"Security check failed: {error_message}")
            return f"Error: {error_message}", "Please provide a valid YouTube URL.", [], None, None
        elif "Rate limit exceeded" in error_message:
            gr.Error(error_message)
            return "Rate limit exceeded.", "Please wait before processing another video.", [], None, None
        else:
            gr.Error(f"An error occurred: {error_message}")
            return "An error occurred.", "Please check the console for details.", [], None, None

def chat_with_video(user_message: str, chat_history: list, vector_index: Any, qa_chain: Any) -> Tuple[list, Any, Any]:
    """Handle chat messages with the video"""
    timer.start("chat_with_video")
    logger.info(f"Processing message: {user_message[:100]}...")
    
    if not qa_chain or not vector_index:
        error_msg = "Q&A system not ready. Please load a video first."
        # Convert to messages format for Gradio
        messages = []
        for entry in chat_history:
            if isinstance(entry, tuple) and len(entry) == 2:
                # Old tuple format: (user_message, bot_message)
                user_msg, bot_msg = entry
                messages.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": bot_msg}
                ])
            elif isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                # Already in messages format: {"role": "...", "content": "..."}
                messages.append(entry)
            else:
                logger.warning(f"Skipping invalid chat entry format: {type(entry)}")
                continue
        # Add current error message
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": error_msg}
        ])
        logger.warning("QA system not available")
        return messages, vector_index, qa_chain

    try:
        # Process the question
        result = process_question(user_message, qa_chain, chat_history, vector_index)
        
        if result.get("error"):
            # Handle specific errors
            if result["error"] == "security_validation_failed":
                answer = "Your question contains inappropriate content and cannot be processed."
                logger.warning("Security validation failed for question")
            elif result["error"] == "rate_limit_exceeded":
                answer = result["answer"]
                logger.warning("Rate limit exceeded for QA")
            else:
                answer = result["answer"]
                logger.error(f"QA processing error: {result['error']}")
        else:
            answer = result["answer"]
            logger.info("Question processed successfully")
        
        # Convert chat history to messages format for Gradio
        messages = []
        for entry in chat_history:
            if isinstance(entry, tuple) and len(entry) == 2:
                # Old tuple format: (user_message, bot_message)
                user_msg, bot_msg = entry
                messages.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": bot_msg}
                ])
            elif isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                # Already in messages format: {"role": "...", "content": "..."}
                messages.append(entry)
            else:
                logger.warning(f"Skipping invalid chat entry format: {type(entry)}")
                continue
        # Add current message and answer
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}
        ])
        
        timer.end("chat_with_video", {
            "question_length": len(user_message),
            "answer_length": len(answer),
            "has_error": bool(result.get("error"))
        })
        
        return messages, vector_index, qa_chain

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        error_answer = "Sorry, there was an error processing your question. Please try again."
        
        # Convert chat history to messages format for Gradio
        messages = []
        for entry in chat_history:
            if isinstance(entry, tuple) and len(entry) == 2:
                # Old tuple format: (user_message, bot_message)
                user_msg, bot_msg = entry
                messages.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": bot_msg}
                ])
            elif isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                # Already in messages format: {"role": "...", "content": "..."}
                messages.append(entry)
            else:
                logger.warning(f"Skipping invalid chat entry format: {type(entry)}")
                continue
        # Add current error message
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": error_answer}
        ])
        
        timer.end("chat_with_video", {"error": str(e)})
        return messages, vector_index, qa_chain


# Build the UI
with gr.Blocks(theme=gr.themes.Soft(), title="YouTube AI Companion") as demo:
    gr.Markdown("""
    # 🎥 YouTube AI Companion
    
    Transform any YouTube video into actionable insights with AI-powered summaries, key points, and intelligent Q&A.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            url_input = gr.Textbox(
                label="YouTube URL", 
                placeholder="Paste your YouTube video URL here...", 
                elem_id="url_input",
                container=True
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("🚀 Load & Analyze", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            refresh_cache_btn = gr.Button("🔄 Refresh Cache", variant="secondary", size="sm")
        with gr.Column(scale=1):
            clear_all_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary", size="sm")

    with gr.Tab("📋 Summary & Key Points"):
        gr.Markdown("### Comprehensive Analysis")
        with gr.Row():
            with gr.Column():
                summary_box = gr.Textbox(
                    label="📝 Detailed Summary", 
                    lines=10, 
                    interactive=False, 
                    elem_classes="output_box",
                    placeholder="Your comprehensive video summary will appear here..."
                )
            with gr.Column():
                bullets_box = gr.Textbox(
                    label="🎯 Key Points", 
                    lines=10, 
                    interactive=False, 
                    elem_classes="output_box",
                    placeholder="Key insights and takeaways will appear here..."
                )
    
    with gr.Tab("💬 Q&A Chat"):
        gr.Markdown("### Ask Questions About the Video")
        chatbot = gr.Chatbot(
            label="Chat with Video",
            height=500,
            type="messages",
            placeholder="Start a conversation about the video content..."
        )
        with gr.Row():
            user_msg = gr.Textbox(
                placeholder="Ask anything about the video content, concepts, or details...", 
                show_label=False,
                elem_id="user_msg_box",
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)


    # State objects
    state_index = gr.State(None)
    state_chain = gr.State(None)

    # Event handlers
    load_btn.click(
        fn=load_video,
        inputs=[url_input],
        outputs=[summary_box, bullets_box, chatbot, state_index, state_chain],
        show_progress="full"
    )
    
    refresh_cache_btn.click(
        fn=force_cache_refresh,
        inputs=[url_input],
        outputs=[summary_box]
    )

    clear_all_cache_btn.click(
        fn=clear_all_cache,
        outputs=[summary_box]
    )
    
    user_msg.submit(
        fn=chat_with_video,
        inputs=[user_msg, chatbot, state_index, state_chain],
        outputs=[chatbot, state_index, state_chain]
    )
    user_msg.submit(lambda: "", None, user_msg, queue=False)
    
    send_btn.click(
        fn=chat_with_video,
        inputs=[user_msg, chatbot, state_index, state_chain],
        outputs=[chatbot, state_index, state_chain]
    )
    send_btn.click(lambda: "", None, user_msg, queue=False)
    

# Launch the app
if __name__ == "__main__":
    logger.info("Starting YouTube AI Companion")
    demo.launch()