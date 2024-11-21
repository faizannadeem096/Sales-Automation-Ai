from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import OpenAI
#from langchain.cache import InMemoryCache

#Chunking data
def chunk_data(data, chunk_size=256):
    """
    Chunk the input data into smaller pieces.

    Args:
        data (str): The input data to be chunked.
        chunk_size (int, optional): The size of each chunk. Defaults to 256.

    Returns:
        list: A list of chunks.
    """
    # Initialize a RecursiveCharacterTextSplitter object
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    
    # Use the text splitter to split the data into chunks
    chunks = text_splitter.split_documents(data)
    
    # Return the list of chunks
    return chunks


#Load embedding chroma
def load_embeddings_chroma(persist_directory='./chroma_db'):
    """
    Load the embeddings and Chroma vector store from the specified directory.

    Args:
        persist_directory (str, optional): The directory from which to load the Chroma vector store.
            Defaults to './chroma_db'.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    # Instantiate the same embedding model used during creation
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Return the loaded vector store
    return vector_store

# create a function to ask questions
def ask_questions(q):
    """
    Ask a question to the given chain and retrieve the result.

    Args:
        q (str): The question to ask.
        chain: The chain object to invoke the question on.

    Returns:
        dict: The result of invoking the question on the chain.
    """
    # Invoke the question on the chain and retrieve the result
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
    set_llm_cache(InMemoryCache())
    prompt = q
    answer = llm.invoke(prompt)
    # Return the result
    return answer
def ask_question(q, chain):
    """
    Ask a question to the given chain and retrieve the result.

    Args:
        q (str): The question to ask.
        chain: The chain object to invoke the question on.

    Returns:
        dict: The result of invoking the question on the chain.
    """
    # Invoke the question on the chain and retrieve the result
    result = chain.invoke({'question': q})

    # Return the result
    return result

def main():
    """
    Main function containing the core logic.

    Returns:
        ConversationalRetrievalChain: The configured ConversationalRetrievalChain object.
    """
    #print("This is the main function.")
    
    # Load embeddings and Chroma vector store
    db = load_embeddings_chroma()
    
    # Initialize LLM, retriever, and memory
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Define system and user message templates
    system_template = r'''
    you are a helpful assistant and friendly answer the user query .

    
    ---------------
    Context: ```{context}```
    '''

    user_template = '''
    Question: ```{question}```
    '''

    messages= [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)
    
    # Create and configure the ConversationalRetrievalChain
    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type='stuff',
        combine_docs_chain_kwargs={'prompt': qa_prompt },
        verbose=False
    )
    
    return crc