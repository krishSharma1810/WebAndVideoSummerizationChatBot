import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# page header
st.set_page_config(page_title='Langchain: summarize text from Youtube and Website', page_icon='')
st.title("Langchain: summarize text from Youtube and Website")
st.subheader('Summarize URL')

# to get the api key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value='', type='password')

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'youtu.be' in url:
        return url.split('/')[-1]
    elif 'youtube.com' in url:
        return url.split('v=')[1].split('&')[0]
    return None

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_youtube_content(url):
    """Get YouTube video transcript"""
    try:
        video_id = get_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript pieces
        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
        
        # Split into chunks
        chunks = chunk_text(transcript_text)
        
        # Convert chunks to documents
        return [type('Document', (), {'page_content': chunk, 'metadata': {'source': url}}) 
                for chunk in chunks]
    
    except Exception as e:
        st.error(f"Error getting YouTube content: {str(e)}")
        st.info("Make sure the video exists and has English captions available.")
        return None

# defining the llm model
if groq_api_key:
    llm = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=groq_api_key)

# defining the system prompt templates
chunk_template = """
Summarize the following transcript chunk in 100 words or less:

{text}

Summary:
"""
chunk_prompt = PromptTemplate(template=chunk_template, input_variables=["text"])

final_template = """
Create a coherent 300-word summary from these chunk summaries:

{text}

Final Summary:
"""
final_prompt = PromptTemplate(template=final_template, input_variables=["text"])

generic_url = st.text_input('URL', label_visibility='collapsed')

if st.button('Summarize the content from Youtube or the Website'):
    # validating the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and URL")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Loading content and generating summary..."):
                # loading the url's data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = get_youtube_content(generic_url)
                    if not docs:  # If YouTube content extraction failed
                        st.stop()
                    
                    # Process chunks individually
                    chunk_summaries = []
                    for doc in docs:
                        chain = load_summarize_chain(llm, chain_type='stuff', prompt=chunk_prompt)
                        response = chain.invoke([doc])
                        summary = response['output_text'] if isinstance(response, dict) else response
                        chunk_summaries.append(summary)
                    
                    # Combine chunk summaries
                    combined_summary = " ".join(chunk_summaries)
                    final_doc = [type('Document', (), {'page_content': combined_summary, 'metadata': {'source': generic_url}})]
                    
                    # Create final summary
                    final_chain = load_summarize_chain(llm, chain_type='stuff', prompt=final_prompt)
                    final_response = final_chain.invoke(final_doc)
                    output_summary = final_response['output_text'] if isinstance(final_response, dict) else final_response
                    
                else:
                    # Handle non-YouTube URLs
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()
                    if docs and docs[0].page_content.strip():
                        chain = load_summarize_chain(llm, chain_type='stuff', prompt=final_prompt)
                        response = chain.invoke(docs)
                        output_summary = response['output_text'] if isinstance(response, dict) else response

                if output_summary:
                    st.success(output_summary)
                else:
                    st.error("No content could be extracted from the provided URL")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if hasattr(e, '__traceback__'):
                st.exception(e)