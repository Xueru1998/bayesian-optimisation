import time
import streamlit as st
import os
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from retrieval import RetrievalComponent
from passageFilter import PassageFilterComponent
from promptMaker import PromptMakerComponent
from generator import GeneratorComponent

# Set page config
st.set_page_config(page_title="RAG-Based Wikipedia Assistant")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Wikipedia Assistant")

# Set file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

# Load the best configuration (if it exists)
best_config_path = "results/best_config.csv"
if os.path.exists(best_config_path):
    best_config = pd.read_csv(best_config_path).iloc[0].to_dict()
else:
    # Default configuration if no evaluation was run
    best_config = {
        'retrieval_type': 'chroma',
        'filter_type': 'threshold_cutoff',
        'prompt_type': 'fstring',
        'generator_type': 'openai'
    }

# Initialize components with the best configuration
retrieval = RetrievalComponent(best_config['retrieval_type'], persistent_directory)
filter_comp = PassageFilterComponent(best_config['filter_type'])
prompt_maker = PromptMakerComponent(best_config['prompt_type'])
generator = GeneratorComponent(best_config['generator_type'])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Reset conversation function
def reset_conversation():
    st.session_state['messages'] = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

user_query = st.chat_input("Ask me anything about Wikipedia articles...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating response...", expanded=True):

            chat_history = []
            for i in range(0, len(st.session_state.messages), 2):
                if i+1 < len(st.session_state.messages):
                    chat_history.append(
                        (st.session_state.messages[i].content, 
                         st.session_state.messages[i+1].content)
                    )
            
            # Run the RAG pipeline
            # Step 1: Retrieve documents
            retrieval_result = retrieval.retrieve(user_query)
            documents = retrieval_result["documents"]
            
            # Step 2: Filter documents
            filter_result = filter_comp.filter(documents, user_query)
            filtered_documents = filter_result["documents"]
            
            # Step 3: Create prompt
            prompt_result = prompt_maker.create_prompt(user_query, filtered_documents)
            prompt = prompt_result["prompt"]
            
            # Step 4: Generate answer
            generate_result = generator.generate(prompt)
            answer = generate_result["answer"]
            
            # Create placeholder for streaming output
            message_placeholder = st.empty()
            
            # Add disclaimer for Wikipedia content
            full_response = (
                "**_This information is based on Trained data and may not be comprehensive. "
                "Please verify important information from reliable sources._** \n\n\n"
            )
            
            # Simulate streaming response
            for chunk in answer.split(". "):
                if chunk:
                    full_response += chunk + ". "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")
            
        st.button('Reset Conversation 🗑️', on_click=reset_conversation)
    
    # Add to conversation history
    st.session_state.messages.extend([
        HumanMessage(content=user_query),
        AIMessage(content=answer)
    ])