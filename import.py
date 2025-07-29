# Import required libraries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os # For checking environment variables

# LangChain imports for embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings # Primary embedding model

from langchain_community.vectorstores import FAISS # For FAISS vector store
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template # Your local HTML templates

# *** NEW IMPORTS FOR HUGGINGFACE PIPELINE ***
# Corrected: Use AutoModelForSeq2SeqLM for T5 models (encoder-decoder)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline # Use HuggingFacePipeline
# *****************************************

# Keep OpenAI imports if you plan to use them, otherwise they can be removed
from langchain_openai import ChatOpenAI


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vectorstore(text_chunks, embedding_model):
    embeddings = None
    if embedding_model == "OpenAI":
        try:
            embeddings = OpenAIEmbeddings()
            st.info("Using OpenAIEmbeddings (ensure OPENAI_API_KEY is set).")
        except Exception as e:
            st.warning(f"Could not initialize OpenAIEmbeddings (Error: {e}). Falling back to HuggingFaceInstructEmbeddings.")
            try:
                # Ensure InstructorEmbedding is correctly initialized
                embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            except Exception as e_hf:
                st.error(f"Failed to initialize HuggingFaceInstructEmbeddings as well (Error: {e_hf}). Check its dependencies.")
                st.stop()
    elif embedding_model == "HuggingFace":
        st.info("Using HuggingFaceInstructEmbeddings.")
        try:
            # Ensure InstructorEmbedding is correctly initialized
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        except Exception as e_hf:
            st.error(f"Failed to initialize HuggingFaceInstructEmbeddings (Error: {e_hf}). Check its dependencies like sentence-transformers, torch, and transformers.")
            st.stop()
    else:
        st.warning("Invalid embedding model choice. Defaulting to HuggingFaceInstructEmbeddings.")
        try:
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        except Exception as e_hf:
            st.error(f"Failed to initialize HuggingFaceInstructEmbeddings (Error: {e_hf}). Check its dependencies.")
            st.stop()

    if embeddings is None:
        st.error("Failed to initialize any embedding model. This should not happen if previous checks pass.")
        st.stop() # Should ideally be caught by specific error handling above

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create conversation chain
def get_conversation_chain(vectorstore, conversational_model, model_temperature=0.5):
    llm = None
    if conversational_model == "OpenAI":
        try:
            llm = ChatOpenAI(temperature=model_temperature)
            st.info("Using OpenAI Chat Model (ensure OPENAI_API_KEY is set).")
        except Exception as e:
            st.warning(f"Could not initialize OpenAI Chat Model (Error: {e}). Falling back to HuggingFace conversational model.")
            # FALLBACK TO THE SMALLEST FLAN-T5 MODEL
            model_id = "google/flan-t5-base" # Using the smallest model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

                pipe = pipeline(
                    "text2text-generation", # Correct task for T5 models
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256, # Reduced for smaller model and efficiency
                    temperature=model_temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
                llm = HuggingFacePipeline(pipeline=pipe)
                st.success(f"Successfully initialized HuggingFace model: {model_id}")
            except Exception as e_hf:
                st.error(f"Failed to initialize HuggingFace model via pipeline (Error: {e_hf}). Check model name, GPU, and internet connection.")
                st.stop()

    elif conversational_model == "HuggingFace":
        st.info("Using HuggingFace conversational model (via pipeline).")
        # USING THE SMALLEST FLAN-T5 MODEL
        model_id = "google/flan-t5-xxl"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

            pipe = pipeline(
                "text2text-generation", # Correct task for T5 models
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256, # Reduced for smaller model and efficiency
                temperature=model_temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            st.success(f"Successfully initialized HuggingFace model: {model_id}")
        except Exception as e_hf:
            st.error(f"Failed to initialize HuggingFace model via pipeline (Error: {e_hf}). Check model name, GPU, and internet connection.")
            st.stop()
    else: # Fallback if invalid choice (shouldn't happen with radio buttons)
        st.warning("Invalid conversational model choice. Defaulting to HuggingFace conversational model.")
        # USING THE SMALLEST FLAN-T5 MODEL
        model_id = "google/flan-t5-small"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=model_temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            st.success(f"Successfully initialized HuggingFace model: {model_id}")
        except Exception as e_hf:
            st.error(f"Failed to initialize HuggingFace model via pipeline (Error: {e_hf}). Check model name, GPU, and internet connection.")
            st.stop()

    if llm is None:
        st.error("Failed to initialize any conversational model. This should not happen if previous checks pass.")
        st.stop() # Should ideally be caught by specific error handling above

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

# Function to handle user input and display chat history
def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0: # User message
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else: # Bot message
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please process documents first to start the conversation.")


# Main function for the Streamlit app
def main():
    # Load environment variables from .env file (for API keys)
    load_dotenv()

    # Configure Streamlit page
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True) # Apply custom CSS

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main content area
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    # Sidebar for document upload and settings
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'", accept_multiple_files=True)

        # Model selection for embeddings
        embedding_model_choice = st.radio(
            "Choose Embedding Model:",
            ("HuggingFace", "OpenAI")
        )

        # Model selection for conversational LLM
        conversational_model_choice = st.radio(
            "Choose Conversational Model:",
            ("HuggingFace", "OpenAI")
        )

        # Temperature slider for LLM (optional)
        model_temperature = st.slider(
            "Model Temperature (Creativity):",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
                return

            with st.spinner("Processing"):
                # 1. Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # 2. Get text chunks
                text_chunks = get_text_chunks(raw_text)

                # 3. Create vector store
                vectorstore = get_vectorstore(text_chunks, embedding_model_choice)

                # 4. Create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, conversational_model_choice, model_temperature
                )
                st.success("Documents processed and chatbot ready! You can now ask questions.")

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()
