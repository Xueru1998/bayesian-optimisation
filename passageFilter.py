import time
from typing import List, Dict, Any, Optional
import pandas as pd

from langchain_core.documents import Document
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter


class PassageFilterComponent:
    def __init__(self, filter_type: str, embeddings=None, threshold: float = 0.7, percentile: float = 75):
        self.filter_type = filter_type
        self.threshold = threshold
        self.percentile = percentile
        self.embeddings = embeddings
        
        if self.embeddings is None and (filter_type == "threshold_cutoff" or filter_type == "percentile_cutoff"):
            # Default embeddings if none provided
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def filter(self, documents: List[Document], query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Ensure we're working with Document objects
        if documents and not isinstance(documents[0], Document):
            print("Warning: Converting string documents to Document objects")
            documents = [Document(page_content=doc) if isinstance(doc, str) else doc for doc in documents]
        
        if self.filter_type == "threshold_cutoff":
            try:
                filter_obj = EmbeddingsFilter(
                    embeddings=self.embeddings,
                    similarity_threshold=self.threshold,
                    k=None  # Use threshold instead of top-k
                )
                filtered_docs = filter_obj.compress_documents(documents, query)
            except Exception as e:
                print(f"Warning: EmbeddingsFilter error ({str(e)}). Using all documents.")
                filtered_docs = documents
                
        elif self.filter_type == "percentile_cutoff":
            try:
                # for percentile cutoff, calculate how many documents to keep
                k = max(1, int(len(documents) * (self.percentile / 100.0)))
                
                filter_obj = EmbeddingsFilter(
                    embeddings=self.embeddings,
                    k=k  # Keep top k documents
                )
                filtered_docs = filter_obj.compress_documents(documents, query)
            except Exception as e:
                print(f"Warning: EmbeddingsFilter error ({str(e)}). Using all documents.")
                filtered_docs = documents
        else:
            # No filtering, return all documents
            filtered_docs = documents
        
        end_time = time.time()
        
        return {
            "documents": filtered_docs,  # This is a list of Document objects
            "time_taken": end_time - start_time
        }