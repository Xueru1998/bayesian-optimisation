import time
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datasets import load_dataset

current_dir_path = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir_path, "data-ingestion-local")

## checking if the directory already exists
if not os.path.exists(persistent_directory):
    print("[INFO] Initiating the build of Vector Database from SQuAD dataset..", end="\n\n")

    # Load SQuAD dataset
    squad_dataset = load_dataset("squad", split="train[:10000]")
    print(f"[INFO] Dataset loaded with {len(squad_dataset)} entries", end="\n")
    
    print("[INFO] Sample entry from dataset:", end="\n")
    print(squad_dataset[0])
    
    # Convert dataset entries to Document objects
    doc_container = []
    
    for entry in squad_dataset:
        context = entry['context']
        question = entry['question']
        
        # Get the answer text
        if entry['answers'] and len(entry['answers']['text']) > 0:
            answer = entry['answers']['text'][0]  # Take the first answer
            
            # Create a document with the context, question and answer
            content = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
            metadata = {
                "source": "squad",
                "question": question,
                "title": entry.get('title', 'Untitled')
            }
            doc = Document(page_content=content, metadata=metadata)
            doc_container.append(doc)
    
    ## splitting the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs_split = splitter.split_documents(documents=doc_container)

    ## displaying information about the split documents
    print("\n--- Document Chunks Information ---", end="\n")
    print(f"Number of document chunks: {len(docs_split)}", end="\n\n")

    ## embedding and vector store
    embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    print("[INFO] Started embedding", end="\n")
    start = time.time()

    """
    creating the embeddings for the documents and
    then storing to a vector database
    """
    vectorDB = Chroma.from_documents(documents=docs_split,
                                     embedding=embedF,
                                     persist_directory=persistent_directory)
    
    end = time.time()
    print("[INFO] Finished embedding", end="\n")
    print(f"[ADD. INFO] Time taken: {end - start}")

else:
    print("[ALERT] Vector Database already exist. ️⚠️")