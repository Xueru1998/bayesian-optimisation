import time
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset

class RetrievalComponent:
    def __init__(self, retrieval_type: str, vector_store_path: str = None):
        self.retrieval_type = retrieval_type
        self.vector_store_path = vector_store_path
        self.retriever = None
        self.setup()
        
        # Cache for SQuAD question-answer pairs
        self.squad_qa_pairs = self._load_squad_qa_pairs()
    
    def setup(self):
        if self.retrieval_type == "chroma":
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.retriever = Chroma(persist_directory=self.vector_store_path, 
                                  embedding_function=embeddings).as_retriever(search_kwargs={"k": 5})
        elif self.retrieval_type == "bm25":
            docs = self._load_documents_from_vectorstore()
            self.retriever = BM25Retriever.from_documents(docs)
        else:
            raise ValueError(f"Unsupported retrieval type: {self.retrieval_type}")
    
    def _load_documents_from_vectorstore(self) -> List[Document]:
        # Load documents from Chroma to use for BM25
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=self.vector_store_path, 
                            embedding_function=embeddings)
        
        try:
            docs = vectorstore.similarity_search("", k=1000)  
            
            if not docs:
                print("Warning: No documents found in vector store. Creating sample documents.")

        except Exception as e:
            print(f"Error retrieving documents from vector store: {str(e)}")
        return docs
    
    def _load_squad_qa_pairs(self):
        """Load SQuAD questions and answers for ground truth evaluation"""
        try:
            # Load a subset of SQuAD to match what was ingested
            squad_dataset = load_dataset("squad", split="train[:10000]")
            
            # Create a mapping of questions to contexts
            qa_pairs = {}
            for item in squad_dataset:
                question = item['question']
                context = item['context']
                qa_pairs[question] = context
                
            return qa_pairs
        except Exception as e:
            print(f"Error loading SQuAD dataset: {e}")
            return {}
        
    def retrieve(self, query: str, ground_truth_context: str = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents and evaluate the results
        
        Args:
            query: The query string
            ground_truth_context: The ground truth context for evaluation
            
        Returns:
            Dictionary with retrieved documents and evaluation metrics
        """
        start_time = time.time()
        
        try:
            # Get results from the retriever
            results = self.retriever.invoke(query)
        except (AttributeError, TypeError):
            # Fall back to old API
            results = self.retriever.get_relevant_documents(query)
        
        end_time = time.time()
        
        # Extract document contents and generate IDs
        doc_contents = [doc.page_content for doc in results]
        doc_ids = [f"doc_{i}" for i, _ in enumerate(results)]  # Generate synthetic IDs
        
        # Get scores based on retrieval type
        scores = self._get_scores(query, results)
        
        # Evaluation metrics
        evaluation = {}
        
        # If we have ground truth, evaluate the results
        if ground_truth_context:
            # For SQuAD, we consider a document relevant if it contains the ground truth context
            relevance = [1.0 if ground_truth_context in content else 0.0 for content in doc_contents]
            
            if sum(relevance) > 0:
                # Calculate basic metrics
                evaluation["recall"] = sum(relevance) / 1.0  # Only one ground truth document
                evaluation["precision"] = sum(relevance) / len(relevance)
                
                if evaluation["precision"] + evaluation["recall"] > 0:
                    evaluation["f1"] = (2 * evaluation["precision"] * evaluation["recall"]) / \
                                     (evaluation["precision"] + evaluation["recall"])
                else:
                    evaluation["f1"] = 0.0
                
                # Calculate MRR
                for i, rel in enumerate(relevance):
                    if rel > 0:
                        evaluation["mrr"] = 1.0 / (i + 1)
                        break
                else:
                    evaluation["mrr"] = 0.0
                
                # Calculate DCG and nDCG
                dcg = self._calculate_dcg(relevance)
                ideal_dcg = self._calculate_dcg([1.0] + [0.0] * (len(relevance) - 1))
                evaluation["ndcg"] = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                
                # Set average precision (for use in MAP calculation)
                relevant_count = 0
                sum_precision = 0.0
                for i, rel in enumerate(relevance):
                    if rel > 0:
                        relevant_count += 1
                        precision_at_i = relevant_count / (i + 1)
                        sum_precision += precision_at_i
                
                evaluation["average_precision"] = sum_precision
            else:
                # No relevant documents found
                evaluation = {
                    "recall": 0.0,
                    "precision": 0.0,
                    "f1": 0.0,
                    "mrr": 0.0,
                    "ndcg": 0.0,
                    "average_precision": 0.0
                }
        
        # Add score statistics to evaluation
        evaluation["mean_score"] = sum(scores) / len(scores) if scores else 0
        evaluation["max_score"] = max(scores) if scores else 0
        evaluation["min_score"] = min(scores) if scores else 0
        
        return {
            "documents": results,
            "document_ids": doc_ids,
            "document_contents": doc_contents,
            "scores": scores,
            "time_taken": end_time - start_time,
            "evaluation": evaluation
        }
    
    def retrieve_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of queries and calculate MAP"""
        results = []
        ap_values = []
        
        for query in queries:
            # Get ground truth context if available
            ground_truth_context = self.squad_qa_pairs.get(query, None)
            
            # Retrieve and evaluate
            result = self.retrieve(query, ground_truth_context)
            results.append(result)
            
            if "evaluation" in result and "average_precision" in result["evaluation"]:
                ap_values.append(result["evaluation"]["average_precision"])
        
        # Calculate MAP
        if ap_values:
            map_value = sum(ap_values) / len(ap_values)
            
            # Add MAP to each result
            for result in results:
                if "evaluation" in result:
                    result["evaluation"]["map"] = map_value
        
        return results
        
    def _get_scores(self, query: str, results: List[Document]) -> List[float]:
        """Get relevance scores based on retrieval type"""
        if self.retrieval_type == "chroma":
            # For Chroma, scores are stored in the metadata if available
            scores = []
            for doc in results:
                if 'score' in doc.metadata:
                    scores.append(doc.metadata['score'])
                else:
                    # If scores not in metadata, calculate cosine similarity
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    query_embedding = embeddings.embed_query(query)
                    doc_embedding = embeddings.embed_documents([doc.page_content])[0]
                    score = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                    scores.append(score)
        elif self.retrieval_type == "bm25":
            # For BM25, calculate scores using the BM25 algorithm
            # This is a simplified approach
            if hasattr(self.retriever, 'bm25'):
                # If using BM25 with scores available
                tokenized_query = self.retriever._tokenizer.tokenize(query)
                scores = self.retriever.bm25.get_scores(tokenized_query)[:len(results)]
            else:
                # Fallback for when direct scores aren't available
                # Assign estimated scores based on position (not ideal but workable)
                scores = [1.0 - (i * 0.1) for i in range(len(results))]
        else:
            # Default scoring when retrieval type is not recognized
            scores = [1.0 - (i * 0.1) for i in range(len(results))]
            
        return scores
    
    def _calculate_dcg(self, relevance_scores: List[float]) -> float:
        """Calculate Discounted Cumulative Gain"""
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0