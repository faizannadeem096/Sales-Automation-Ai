import os
from getpass import getpass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

def chunk_data(data, chunk_size=256):
    """
    Split the input data into chunks.

    Args:
        data (str): The input data to be chunked.
        chunk_size (int, optional): The size of each chunk. Defaults to 256.

    Returns:
        list: A list of chunked data.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    
    # Split the data into chunks
    chunks = text_splitter.split_documents(data)
    
    return chunks

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    """
    Create Chroma embeddings from text chunks.

    Args:
        chunks (list): List of text chunks.
        persist_directory (str, optional): Directory to persist the vector store. Defaults to './chroma_db'.

    Returns:
        Chroma: The created Chroma vector store.
    """
    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    embeddings = HuggingFaceEmbeddings() 
    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store

if __name__ == "__main__":
    #embeddings = HuggingFaceEmbeddings()
    loader = CSVLoader('sales_data.csv')
    data = loader.load()
    # Split the documents into chunks
    chunks = chunk_data(data, chunk_size=256)
        
    # Create Chroma embeddings from the chunks
    vector_store = create_embeddings_chroma(chunks)
    print("done")