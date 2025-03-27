import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class PromptMakerComponent:
    def __init__(self, prompt_type: str):
        self.prompt_type = prompt_type
    
    def create_prompt(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.prompt_type == "fstring":
            context = "\n\n".join([doc.page_content for doc in documents])
            prompt_template = "Question: {query}\n\nContext: {context}\n\nAnswer:"
            prompt = prompt_template.format(query=query, context=context)
        
        elif self.prompt_type == "long_text_reorder":
            # Reorder documents by relevance and add document markers
            formatted_docs = []
            for i, doc in enumerate(documents):
                formatted_docs.append(f"[Document {i+1}]\n{doc.page_content}\n")
            
            context = "\n".join(formatted_docs)
            prompt_template = (
                "Below are several documents providing context for a question.\n\n"
                "{context}\n\n"
                "Question: {query}\n\n"
                "Using only the information from these documents, provide a concise answer:"
            )
            prompt = prompt_template.format(query=query, context=context)
        
        end_time = time.time()
        
        return {
            "prompt": prompt,
            "time_taken": end_time - start_time
        }
