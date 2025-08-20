# app.py
import gradio as gr
from agents.transcript_agent import get_transcript, extract_video_id
from agents.chunk_embed_agent import embed_transcript
from agents.qa_agent import get_qa_chain
from agents.summarizer_agent import generate_summary_and_bullets
from langchain.schema import HumanMessage, AIMessage

# --- ERROR HANDLING & UI UPDATES ---
def load_video(video_url):
    print(f"\n--- load_video CALLED with URL: {video_url} ---")
    
    if not video_url:
        gr.Warning("Please enter a YouTube URL.")
        return "Enter a URL to begin.", "", None, None, None

    # Reset chatbot UI to be empty.
    chatbot_out = None 
    
    try:
        gr.Info("1/5: Validating URL and fetching transcript...")
        extract_video_id(video_url) 
        transcript_list = get_transcript(video_url)

        if not transcript_list or not any(s.strip() for s in transcript_list):
            gr.Error("Transcript is empty or unavailable for this video.")
            print("--- load_video EXIT: Empty transcript ---")
            return "Error: Transcript is empty or unavailable.", "Try a different video.", chatbot_out, None, None

        print(f"Transcript fetched. Utterances: {len(transcript_list)}")
        
        gr.Info("2/5: Summarizing video...")
        summary, bullets = generate_summary_and_bullets(transcript_list)
        print(f"Summary generated: {summary[:50]}...")
        
        gr.Info("3/5: Embedding transcript for Q&A...")
        vector_index = embed_transcript(transcript_list)

        if vector_index is None:
            gr.Error("Failed to create vector index for Q&A.")
            return summary, bullets, chatbot_out, None, None
        
        gr.Info("4/5: Initializing Q&A agent...")
        qa_chain = get_qa_chain(vector_index)

        if qa_chain is None:
            gr.Error("Failed to initialize Q&A agent.")
            return summary, bullets, chatbot_out, vector_index, None

        gr.Info("5/5: Video processed successfully!")
        print("--- load_video COMPLETED ---")
        return summary, bullets, chatbot_out, vector_index, qa_chain

    except Exception as e:
        error_message = str(e)
        print(f"An error occurred during video loading: {error_message}")
        # Specifically check for the known transcript error
        if "no element found" in error_message or "Could not retrieve transcript" in error_message:
            gr.Error("Failed: This video does not have a transcript available. Please try another one.")
            return "Error: No transcript found for this video.", "Please try a different YouTube video that has captions enabled.", chatbot_out, None, None
        else:
            gr.Error(f"An unknown error occurred: {error_message}")
            return "An unknown error occurred.", "Please check the console for details.", chatbot_out, None, None


def chat_with_video(user_message, chat_history, vector_index, qa_chain):
    print(f"\n--- chat_with_video CALLED ---")
    
    if not qa_chain or not vector_index:
        chat_history.append((user_message, "The Q&A system is not available. Please load a video successfully first."))
        return chat_history, vector_index, qa_chain

    langchain_chat_history = []
    # Convert Gradio's history to LangChain's format
    for user_q, ai_a in chat_history:
        langchain_chat_history.append(HumanMessage(content=user_q))
        langchain_chat_history.append(AIMessage(content=ai_a))

    answer = "Sorry, I couldn't find an answer in the video content for that."
    try:
        print(f"User question for chat: {user_message}")
        
        result = qa_chain.invoke({
            "question": user_message,
            "chat_history": langchain_chat_history
        })
        answer = result.get("answer", "Sorry, I couldn't find an answer in the video content.")
        
        source_documents = result.get("source_documents", [])
        if source_documents:
            print("\n--- Retrieved Source Documents for Q&A ---")
            for i, doc in enumerate(source_documents):
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                print(f"  Chunk {chunk_id} (Doc {i+1}): '{doc.page_content[:250]}...'")
        else:
            print("\n--- No source documents were retrieved for this question. ---")

        print(f"LLM Answer (Chat): {answer}")

    except Exception as e:
        print(f"Error during QA chain invocation (Chat): {e}")
        answer = f"An error occurred: {e}"
            
    chat_history.append((user_message, answer))
    print("--- chat_with_video COMPLETED ---")
    return chat_history, vector_index, qa_chain


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 YouTube AI Companion")
    gr.Markdown("Enter a YouTube URL to get a summary, key takeaways, and ask questions about the video.")

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL", 
            placeholder="e.g., https://www.youtube.com/watch?v=kCc8FmEb1nY", 
            elem_id="url_input"
        )
        load_btn  = gr.Button("Load & Analyze", variant="primary")

    with gr.Tab("Summary & Key Points"):
        with gr.Row():
            summary_box = gr.Textbox(label="Summary", lines=8, interactive=False, elem_classes="output_box")
            bullets_box = gr.Textbox(label="Key Points", lines=8, interactive=False, elem_classes="output_box")
    
    with gr.Tab("Q&A Chat"):
        chatbot = gr.Chatbot(
            label="Chat with Video",
            height=450,
            bubble_full_width=False
        )
        user_msg = gr.Textbox(
            placeholder="Ask a question about the video…", 
            show_label=False,
            elem_id="user_msg_box"
        )

    # State objects to hold the vector index and QA chain
    state_index = gr.State(None)
    state_chain = gr.State(None)

    # Event handlers
    load_btn.click(
        fn=load_video,
        inputs=[url_input],
        outputs=[summary_box, bullets_box, chatbot, state_index, state_chain],
        show_progress="full"
    )
    user_msg.submit(
        fn=chat_with_video,
        inputs=[user_msg, chatbot, state_index, state_chain],
        outputs=[chatbot, state_index, state_chain]
    )
    user_msg.submit(lambda: "", None, user_msg, queue=False)

demo.launch()
