import streamlit as st 
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env and configure API keys
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY", "")
def get_vectorstore(website_url):
    # Load and index documents from the provided website URL
    loader = WebBaseLoader(website_url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    document_chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(
        document_chunks,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )
    return vectorstore


def get_response(user_input):
    # Retrieve relevant context from the vector store
    retriever = st.session_state.vector_store.as_retriever()
    retrieved_docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Build a conversational RAG prompt and query the LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "chat_history": st.session_state.chat_history,
            "input": user_input,
        }
    )

    # ChatGroq returns an AIMessage-like object; use its content
    return response.content

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(website_url)    

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


