import time
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

load_dotenv() 
class GeneratorComponent:
    def __init__(self, generator_type: str, openai_api_key=None):
        self.generator_type = generator_type
        self.openai_api_key = openai_api_key
        self.generator = self._setup_generator()
    
    def _setup_generator(self):
        if self.generator_type == "openai":
           return ChatOpenAI(model="gpt-3.5-turbo", api_key=self.openai_api_key)
        elif self.generator_type == "ollama":
            return Ollama(model="llama2")
        else:
            raise ValueError(f"Unsupported generator type: {self.generator_type}")
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        
        response = self.generator.invoke(prompt)
        
        end_time = time.time()
        
        # Handle different response formats depending on the model
        if self.generator_type == "openai":
            # ChatOpenAI returns a message with content property
            answer = response.content
        elif self.generator_type == "ollama":
            # Ollama return different formats depending on the version
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
        
        return {
            "answer": answer,
            "time_taken": end_time - start_time
        }