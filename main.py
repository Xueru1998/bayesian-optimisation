from datasets import load_dataset
from retrieval import RetrievalComponent
from pipelineRunner import PipelineRunner
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == "__main__":
    # Path to your vector store
    vector_store_path = "data-ingestion-local"
    
    start_time = time.time()
    
    # Load SQuAD dataset to get questions for testing
    print("[INFO] Loading SQuAD dataset to get test questions...")
    squad_dataset = load_dataset("squad", split="train[:1000]")
    
    # Extract questions and their contexts from the dataset
    questions_with_context = [(entry['question'], entry['context']) for entry in squad_dataset]
    
    # Make a unique set of questions (some questions might be duplicates)
    unique_questions = list(set(q for q, _ in questions_with_context))
    print(f"[INFO] Found {len(unique_questions)} unique questions in the dataset")
    
    # Select random subset for evaluation
    test_size = min(10, len(unique_questions))  # Increased test size for better metrics
    sample_queries = random.sample(unique_questions, test_size)
    
    print(f"[INFO] Selected {len(sample_queries)} questions for evaluation")
    
    # Create a mapping of questions to contexts
    squad_qa_pairs = {}
    for item in squad_dataset:
        question = item['question']
        context = item['context']
        squad_qa_pairs[question] = context
    
    # Initialize the pipeline runner with the squad dataset
    runner = PipelineRunner(vector_store_path, squad_qa_pairs=squad_qa_pairs)
    
    print("\n[INFO] Running all pipeline combinations...")
    print(f"[INFO] This may take some time. Using {len(sample_queries)} queries.")
    
    # Run all combinations
    runner.run_all_combinations(queries=sample_queries)
    
    print("\n[INFO] Finding best configurations...")
    
    # Get best configurations for different metrics
    metrics_to_check = ["overall", "precision", "recall", "f1", "ndcg", "mrr"]
    
    for metric in metrics_to_check:
        best_config = runner.get_best_configuration(prioritize_metric=metric)
        print(f"Best configuration for {metric}: {best_config}")
    
    # Create visualizations of the results
    try:
        os.makedirs("results/visualizations", exist_ok=True)
        
        # Load the metrics summary
        summary_df = pd.read_csv("results/metrics_summary.csv")
        
        # Visualize the performance of different retrieval types
        plt.figure(figsize=(12, 8))
        metrics = [col for col in summary_df.columns if col.startswith("retrieval_") and col != "retrieval_time"]
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            avg_by_retrieval = summary_df.groupby('retrieval_type')[metric].mean().reset_index()
            
            # Create bar plot
            sns.barplot(x='retrieval_type', y=metric, data=avg_by_retrieval)
            plt.title(f'Average {metric} by Retrieval Type')
            plt.ylabel(metric.replace('retrieval_', ''))
            plt.tight_layout()
            plt.savefig(f"results/visualizations/{metric}_by_retrieval.png")
            plt.close()
        
        # Create comparison heatmap of all metrics
        plt.figure(figsize=(14, 10))
        # Create a pivot table with retrieval type and filter type as indices
        heatmap_data = summary_df.pivot_table(
            index=['retrieval_type', 'filter_type'],
            columns=['prompt_type', 'generator_type'],
            values='retrieval_f1'  # Or any other key metric
        )
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title('F1 Score Comparison Across Configurations')
        plt.tight_layout()
        plt.savefig("results/visualizations/f1_heatmap.png")
        
        print("\n[INFO] Visualizations created in results/visualizations/")
    except Exception as e:
        print(f"[WARNING] Could not create visualizations: {str(e)}")
    
    # Report total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\n[INFO] Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")