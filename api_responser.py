# Importing requred libraries of python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import shutil
import os



CHROMA_PATH = "Vector_Database/"             # Directory for saving the vector database


# OPEN AI API key in use
Open_ai = "sk"

custom_separators = ["}, {\'url\':"]

def split_data():
    from api_text import api_data
    api_data_str = str(api_data)
    text_splitter = RecursiveCharacterTextSplitter(
        separators = custom_separators,
        chunk_size = 1700,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.create_documents([api_data_str])
    print(f"Split {len([api_data_str])} data sets into {len(chunks)} chunks.")
    print(chunks[1])
    return chunks
split_data()

# Saving the chunks to the vector database
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vector_store = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=Open_ai), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def prompt():
    # Ask the question.
    query_text = input("Ask any question from the testing_data:  \n \t ")

    # Prepare the vector Database to answer the question.
    embedding_function = OpenAIEmbeddings(openai_api_key=Open_ai)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Searching the Database.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # prompting the openai model
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    model = ChatOpenAI(openai_api_key=Open_ai)
    response_text = model.invoke(prompt)
    print(response_text)

    ask_again()

def ask_again():
    again = input("Would you like to ask another question? (yes/no) \n \t")
    if again.lower() == "yes":
        prompt()
    else:
        print("Thanks for using our service. Goodbye!")

# main function that executes all other functions
def main():
    chunks = split_data()
    save_to_chroma(chunks)
    prompt()      


# calling the main function    
if __name__ == "__main__":
    main()   