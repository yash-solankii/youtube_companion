# agents/summarizer_agent.py
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

load_dotenv()

# LLM SETUP
llm = ChatGroq(
    model="llama3-70b-8192", 
    temperature=0.2,
    max_tokens=4096
)

# SPLITTING STRATEGY
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# PROMPTS
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following excerpt into a single coherent paragraph "
        "that captures all the key ideas and technical points:\n\n"
        "{text}"
    )
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You have several detailed paragraph summaries. "
        "Merge them into one concise, well‑structured overview of the full transcript, "
        "preserving important concepts and examples:\n\n"
        "{text}"
    )
)

BULLET_PROMPT = PromptTemplate(
    input_variables=["summary"],
    template=(
        "From the summary below, list the 3–5 most important takeaways as bullet points:\n\n"
        "{summary}"
    )
)

# CORE PROCESSING FUNCTION
def generate_summary_and_bullets(transcript: list[str]) -> tuple[str, str]:
    """
    Generates a summary and bullet points from a list of transcript utterances.
    """
    print("--- generate_summary_and_bullets CALLED ---")
    if not transcript:
        return "Transcript was empty.", "No key points could be generated."

    # Join the list of utterances into a single string
    full_text = "\n".join(transcript)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]
    
    # Summarize (map_reduce chain)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        # add verbose=True here for more detailed console logs
    )
    summary_input = {"input_documents": docs}
    summary = chain.invoke(summary_input).get("output_text", "").strip()
    
    # Bullet extraction
    bullet_prompt_formatted = BULLET_PROMPT.format(summary=summary)
    raw = llm.invoke(bullet_prompt_formatted)
    bullets = raw.content.strip() if hasattr(raw, "content") else str(raw).strip()
    
    print("Summarizer: Summary and bullets generated successfully.")
    return summary, bullets