import os
import time
import logging
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import serpent

# Custom serializer for numpy string types(due to python and package version incompatible)
original_serialize = serpent.Serializer._serialize

def patched_serialize(self, obj, out, level):
    if isinstance(obj, np.str_):
        self._serialize(str(obj), out, level)
    else:
        original_serialize(self, obj, out, level)

serpent.Serializer._serialize = patched_serialize
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RAG-BOAH')


class RAGWorker(Worker):
    """Worker class for RAG pipeline evaluation with BOHB"""
    
    def __init__(self, vector_store_path: str, queries: List[str], *args, **kwargs):
        """
        Initialize the RAG Worker for BOHB optimization. 
        The worker will perform actual evaluation of configurations. 
        
        Args:
            vector_store_path (str): Path to the vector store
            queries (List[str]): List of sample queries to evaluate
            *args, **kwargs: Additional arguments for the Worker base class
        """
        super().__init__(*args, **kwargs)
        self.vector_store_path = vector_store_path
        self.queries = queries
        
        
    #The budget parameter determines how many queries to evaluate 
    # lower budgets = faster, less accurate evaluations; 
    # higher budgets = slower, more accurate evaluations.    
    def compute(self, config: Dict[str, Any], budget: float, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a RAG pipeline configuration with the given budget
        
        Args:
            config (Dict[str, Any]): Configuration from ConfigSpace
            budget (float): Budget for this evaluation (number of queries to evaluate)
            
        Returns:
            Dict[str, Any]: Results including loss and additional info
        """
        # Import components dynamically to avoid circular imports
        from retrieval import RetrievalComponent
        from passageFilter import PassageFilterComponent
        from promptMaker import PromptMakerComponent
        from generator import GeneratorComponent
        from datasets import load_dataset
        
        start_time = time.time()
        
        # Convert budget to number of queries to evaluate
        num_queries = max(1, min(len(self.queries), int(budget)))
        subset_queries = self.queries[:num_queries]
        
        # Initialize components based on configuration
        retrieval = RetrievalComponent(config['retrieval_type'], self.vector_store_path)
        filter_comp = PassageFilterComponent(config['filter_type'])
        prompt_maker = PromptMakerComponent(config['prompt_type'])
        generator = GeneratorComponent(config['generator_type'])
        
        # Load SQuAD dataset to get ground truth contexts
        squad_dataset = load_dataset("squad", split="train[:10000]")
        
        # Create a mapping of questions to contexts for ground truth
        qa_pairs = {}
        for item in squad_dataset:
            qa_pairs[item['question']] = item['context']
        
        # Metrics aggregation 
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'mrr': [],
            'ndcg': [],
            'average_precision': []
        }
        
        for query in subset_queries:
            # Get ground truth context if available
            ground_truth_context = qa_pairs.get(query, None)
            
            # Step 1: Retrieve documents with evaluation
            retrieval_result = retrieval.retrieve(query, ground_truth_context)
            documents = retrieval_result["documents"]
            
            # Collect retrieval metrics
            if "evaluation" in retrieval_result:
                eval_metrics = retrieval_result["evaluation"]
                for metric_name in metrics.keys():
                    if metric_name in eval_metrics:
                        metrics[metric_name].append(eval_metrics[metric_name])
            
            # Step 2: Filter documents (continue with pipeline for completeness)
            filtered_documents = filter_comp.filter(documents, query)["documents"]
            
            # Step 3: Create prompt
            prompt = prompt_maker.create_prompt(query, filtered_documents)["prompt"]
            
            # Step 4: Generate answer (optional - not needed for retrieval evaluation)
            generator.generate(prompt)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
            else:
                avg_metrics[metric_name] = 0.0
        
        # Calculate overall score (normalized average of all metrics)
        # This is similar to the overall score in PipelineRunner
        metric_weights = {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.5,  # Give F1 higher weight as it balances precision/recall
            'mrr': 1.0,
            'ndcg': 1.2, # nDCG slightly higher as it accounts for position
            'average_precision': 1.3  # AP is important for ranking quality
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for metric_name, weight in metric_weights.items():
            if metric_name in avg_metrics and avg_metrics[metric_name] > 0:
                weighted_sum += avg_metrics[metric_name] * weight
                total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Return a loss to minimize
        # Since higher metrics are better, we use 1 - score as the loss
        loss = 1 - overall_score
        
        # Return results in the format expected by BOHB
        return {
            'loss': loss,  # primary metric to minimize (1 - overall_score)
            'info': {
                'overall_score': overall_score,
                'metrics': avg_metrics,
                'num_queries': num_queries,
                'total_time': time.time() - start_time,
                'config': config
            }
        }

def get_configspace() -> CS.ConfigurationSpace:
    """
    Define the configuration space for RAG pipeline components
    Includes parameters that affect retrieval quality
    
    Returns:
        CS.ConfigurationSpace: Configuration space object
    """
    config_space = CS.ConfigurationSpace()
    
    # Define hyperparameters
    retrieval_type = CSH.CategoricalHyperparameter(
        'retrieval_type', 
        choices=['chroma', 'bm25'], 
        default_value='chroma'
    )
    
    filter_type = CSH.CategoricalHyperparameter(
        'filter_type', 
        choices=['threshold_cutoff', 'percentile_cutoff', 'none'], 
        default_value='threshold_cutoff'
    )
    
    prompt_type = CSH.CategoricalHyperparameter(
        'prompt_type', 
        choices=['fstring', 'long_text_reorder'], 
        default_value='fstring'
    )
    
    generator_type = CSH.CategoricalHyperparameter(
        'generator_type', 
        choices=['openai', 'ollama'], 
        default_value='openai'
    )
    
    # Additional parameters that might affect retrieval quality
    similarity_threshold = CSH.UniformFloatHyperparameter(
        'similarity_threshold',
        lower=0.5,
        upper=0.9,
        default_value=0.7
    )
    
    percentile_cutoff = CSH.UniformFloatHyperparameter(
        'percentile_cutoff',
        lower=50.0,
        upper=90.0,
        default_value=75.0
    )
    
    # Add hyperparameters to configuration space
    config_space.add_hyperparameters([
        retrieval_type,
        filter_type,
        prompt_type,
        generator_type,
        similarity_threshold,
        percentile_cutoff
    ])
    
    # Add conditions for filter-specific parameters
    similarity_threshold_condition = CS.EqualsCondition(
        similarity_threshold, filter_type, 'threshold_cutoff'
    )
    
    percentile_cutoff_condition = CS.EqualsCondition(
        percentile_cutoff, filter_type, 'percentile_cutoff'
    )
    
    config_space.add_conditions([
        similarity_threshold_condition,
        percentile_cutoff_condition
    ])
    
    return config_space
    
    # ! could add conditional hyperparameters that only apply when certain conditions are met below
    # eg: temperature for LLM models, max/min values for percentile/threshold
        
    return config_space

def run_bohb_optimization(
    vector_store_path: str,
    queries: List[str],
    output_dir: str = 'results/bohb',
    min_budget: int = 1,
    max_budget: Optional[int] = None,
    n_iterations: int = 10,
    n_workers: int = 1,
    eta: int = 2
) -> Tuple[Dict[str, Any], Dict[str, Any], hpres.Result]:
    """
    Run BOHB optimization for RAG pipeline configuration
    
    Args:
        vector_store_path (str): Path to the vector store
        queries (List[str]): List of sample queries to evaluate
        output_dir (str): Directory to store results
        min_budget (int): Minimum budget (number of queries to evaluate)
        max_budget (int, optional): Maximum budget
        n_iterations (int): Number of iterations for BOHB
        n_workers (int): Number of parallel workers
        eta (int): Parameter for BOHB controlling aggressiveness
        
    Returns:
        Tuple containing:
            - Best configuration value (loss)
            - Best configuration
            - BOHB result object
    """
    # Set the maximum budget if not provided
    if max_budget is None:
        max_budget = len(queries)
    
    # Need to ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_space = get_configspace()
    
    # Save configuration space for later analysis
    from ConfigSpace.read_and_write import json as cs_json
    with open(output_dir / 'configspace.json', 'w') as f:
        f.write(cs_json.write(config_space))
    
    # Set up nameserver: distributed system infrastructure
    NS = hpns.NameServer(run_id='rag_bohb', host='127.0.0.1', port=None)
    NS.start()
    
    # Create workers
    workers = []
    for i in range(n_workers):
        worker = RAGWorker(
            vector_store_path=vector_store_path,
            queries=queries,
            nameserver='127.0.0.1',
            run_id='rag_bohb',
            id=i
        )
        worker.run(background=True)
        workers.append(worker)
    
    result_logger = hpres.json_result_logger(directory=output_dir, overwrite=True)
    
    # Set up BOHB optimizer
    bohb = BOHB(
        configspace=config_space,
        run_id='rag_bohb',
        nameserver='127.0.0.1',
        result_logger=result_logger,
        min_budget=min_budget,
        max_budget=max_budget,
        eta=eta
    )
    
    # Run optimization
    logger.info(f"Starting BOHB optimization with {n_iterations} iterations")
    res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)
    
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    
    # Get the incumbent (best configuration)
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    inc_value = res.get_runs_by_id(incumbent)[-1]['loss']
    inc_config = id2config[incumbent]['config']
    
    logger.info(f"Optimization complete. Best configuration: {inc_config}")
    logger.info(f"Best value (avg time per query): {inc_value:.4f} seconds")
    
    # Save incumbent trajectory for analysis
    incumbent_trajectory = res.get_incumbent_trajectory()
    
    return inc_value, inc_config, res

def run_full_boah_optimization(
    vector_store_path: str,
    sample_queries: List[str],
    output_dir: str = 'results/boah',
    min_budget: int = 1,
    max_budget: Optional[int] = None,
    n_iterations: int = 10,
    n_workers: int = 1
) -> Dict[str, Any]:
    """
    Run full BOAH optimization pipeline (BOHB only)
    Evaluates configurations based on retrieval quality metrics
    
    Args:
        vector_store_path (str): Path to the vector store
        sample_queries (List[str]): List of sample queries to evaluate
        output_dir (str): Directory to store results
        min_budget (int): Minimum budget (number of queries)
        max_budget (int, optional): Maximum budget
        n_iterations (int): Number of iterations for BOHB
        n_workers (int): Number of parallel workers
        
    Returns:
        Dict[str, Any]: Best configuration with metadata
    """
    # Run BOHB optimization
    logger.info(f"Starting BOAH optimization at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    inc_value, inc_config, result = run_bohb_optimization(
        vector_store_path=vector_store_path,
        queries=sample_queries,
        output_dir=output_dir,
        min_budget=min_budget,
        max_budget=max_budget,
        n_iterations=n_iterations,
        n_workers=n_workers
    )
    
    optimization_time = time.time() - start_time
    
    # Prepare best configuration result
    best_config_result = inc_config.copy()
    best_config_result['overall_score'] = 1 - inc_value  # Convert loss back to score
    best_config_result['optimization_time'] = optimization_time
    
    # Get detailed metrics from the best run
    all_runs = result.get_all_runs()
    for run in all_runs:
        if run['config'] == inc_config and run['budget'] == max_budget:
            if 'info' in run and 'metrics' in run['info']:
                for metric_name, metric_value in run['info']['metrics'].items():
                    best_config_result[f'metric_{metric_name}'] = metric_value
            break
    
    logger.info("\nBest Configuration:")
    for key, value in best_config_result.items():
        logger.info(f"{key}: {value}")
    
    # Calculate improvement over initial configurations
    if all_runs:
        initial_scores = []
        for run in all_runs:
            if run['budget'] == min_budget and 'info' in run and 'overall_score' in run['info']:
                initial_scores.append(run['info']['overall_score'])
        
        if initial_scores:
            initial_avg = sum(initial_scores) / len(initial_scores)
            current_score = best_config_result['overall_score']
            improvement = (current_score - initial_avg) / initial_avg * 100 if initial_avg > 0 else float('inf')
            logger.info(f"Quality improvement: {improvement:.2f}% better than initial configurations")
            best_config_result['improvement_percentage'] = improvement
    
    logger.info(f"Total optimization time: {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    return best_config_result

if __name__ == "__main__":
    # Sample queries 
    sample_queries = [
        "What are the main provisions of contract law?",
        "Explain the concept of reasonable doubt",
        "What rights do tenants have when facing eviction?",
        "Who was the first president of the United States?",
        "When was the Declaration of Independence signed?",
        "What factors contributed to the fall of the Roman Empire?"
    ]
    
    vector_store_path = "data-ingestion-local"
    
    # Run BOAH optimization
    best_config = run_full_boah_optimization(
        vector_store_path=vector_store_path,
        sample_queries=sample_queries,
        n_iterations=10,
        n_workers=1,
        min_budget=1,
        max_budget=len(sample_queries)
    )
    
    print(f"Best RAG pipeline configuration: {best_config}")