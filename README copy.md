# Project Setup and Execution Guide

## Setup Instructions


### 1. Create a Virtual Environment 
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**If python version is incompatible with some packages' version, please try to use python version <= 3.10, and create new venv environment**

### 2. Configure Environment Variables
Create a `.env` file in the root directory of the project with your OpenAI API key:
Open the `.env` file and add the following:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies
Install all required packages by running:
```bash
pip install -r requirements.txt
```

### 4. Data Ingestion
Prepare data by running the data ingestion script:
```bash
python data-ingestion.py
```
This script will process and prepare the necessary data for further analysis.

### 5. Basic RAG pipeline
To run iterations across different combinations:

**Exhaustive Grid Search: Tries every possible combination of components**

**Brute Force: No optimization strategy**

**Sequential Evaluation: Processes all combinations one after another**

**Simple Analysis: Simply identifies the best configuration after trying everything**

```bash
python pipelineRunner.py
```

### 6. RAG Pipeline Optimization with BOAH in BOAH.py

The BOAH.py script implements the BOAH (Bayesian Optimization & Analysis of Hyperparameters) framework to optimize RAG pipeline configurations:

### Core BOAH Concepts Applied

1. **ConfigSpace Integration**: 
   - Defines a formal search space for RAG parameters
   - Supports categorical and conditional hyperparameters

2. **Multi-Fidelity Optimization (BOHB)**:
   - Uses BOHB (Bayesian Optimization HyperBand) algorithm
   - "Budget" concept = number of queries to evaluate
   - Efficiently allocates resources by quickly eliminating poor configurations

3. **Worker-Based Evaluation**:
   - `RAGWorker` class handles configuration evaluation
   - Performance metric: average time per query
   - Tracks detailed timing for each pipeline component

4. **Optimization Flow**:
   - Configuration sampling through Bayesian optimization
   - Successive halving to focus on promising configurations
   - Parallel evaluation for faster optimization

This implementation demonstrates efficient RAG pipeline optimization by applying BOAH's multi-fidelity Bayesian approach to find the best combination of retrieval method, filtering strategy, prompt construction, and generation model.

Note: for small number of combinations, BOAH may take more time than basic RAG pipepline. 

The early stopping mechanisms are removed here. 

```bash
python BOAH.py
```



### 7. Web Interface
Launch the web application using Streamlit:
```bash
streamlit run main-app.py
```

