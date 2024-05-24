#imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
import chromadb
import argparse
import shutil
import os
import csv

# Directory paths
DATA_PATH = "data/Books"
CHROMA_PATH = "data/Vector_Database/"

Open_ai = "sk-LTPToGpHWlPp74IFFWpeT3BlbkFJH3FUaEjjpRVS8CyFeiG"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_data()
    chunks = split_data(documents)
    save_to_chroma(chunks)


# #For loading CSV files
# def load_data():
#     data_loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
#     loaded_data = data_loader.load()
#     return loaded_data
 
#For loading PDF files
def load_data():
    data_loader = PyPDFDirectoryLoader(DATA_PATH)
    loaded_data = data_loader.load()
    return loaded_data
 
def split_data(loaded_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 300,
        length_function = len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(loaded_data)
    print(f"Split {len(loaded_data)} Data Sets into {len(chunks)} chunks.")

    return chunks



def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vector_store = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=Open_ai), persist_directory=CHROMA_PATH
    )
    # vector_store.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()   
