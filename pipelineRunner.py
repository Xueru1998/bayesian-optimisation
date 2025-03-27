import os
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from retrieval import RetrievalComponent
from passageFilter import PassageFilterComponent
from promptMaker import PromptMakerComponent
from generator import GeneratorComponent


class PipelineRunner:
    def __init__(self, vector_store_path: str, squad_qa_pairs=None):
        self.vector_store_path = vector_store_path
        self.results = []
        
        # Define all possible components
        self.retrieval_options = ["chroma", "bm25"]
        self.filter_options = ["threshold_cutoff", "percentile_cutoff", "none"]
        self.prompt_options = ["fstring", "long_text_reorder"]
        self.generator_options = ["openai", "ollama"]
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Use provided SQuAD dataset or load it if not provided
        if squad_qa_pairs:
            self.squad_qa_pairs = squad_qa_pairs
        else:
            self.squad_qa_pairs = self._load_squad_qa_pairs()
    
    def _load_squad_qa_pairs(self, split="train[:1000]"):
        """Load SQuAD questions and answers for ground truth evaluation"""
        try:
            from datasets import load_dataset
            # Load a subset of SQuAD to match what was ingested
            # Default to 1000 entries to match main.py
            squad_dataset = load_dataset("squad", split=split)
            
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

    def run_combination(self, 
                    retrieval_type: str, 
                    filter_type: str, 
                    prompt_type: str, 
                    generator_type: str,
                    queries: List[str]):
        
        print(f"Running combination: {retrieval_type} + {filter_type} + {prompt_type} + {generator_type}")
        
        retrieval = RetrievalComponent(retrieval_type, self.vector_store_path)
        filter_comp = PassageFilterComponent(filter_type)
        prompt_maker = PromptMakerComponent(prompt_type)
        generator = GeneratorComponent(generator_type)
        
        # Track metrics for this combination
        combination_metrics = []
        
        for query in queries:            
            # Prepare result data
            result_data = {
                "retrieval_type": retrieval_type,
                "filter_type": filter_type,
                "prompt_type": prompt_type,
                "generator_type": generator_type,
                "query": query
            }
            
            # Get ground truth context if available
            ground_truth_context = self.squad_qa_pairs.get(query, None)
            
            # Step 1: Retrieve documents
            retrieval_result = retrieval.retrieve(query, ground_truth_context)
            documents = retrieval_result["documents"]
            result_data["retrieval_time"] = retrieval_result.get("time_taken", 0)
            
            # Add retrieval metrics if available
            if "evaluation" in retrieval_result:
                for metric_name, metric_value in retrieval_result["evaluation"].items():
                    result_data[f"retrieval_{metric_name}"] = metric_value
                    
                # Add this query's metrics to the combination metrics
                if ground_truth_context:
                    combination_metrics.append(retrieval_result["evaluation"])
            
            # Step 2: Filter documents
            filter_result = filter_comp.filter(documents, query)
            filtered_documents = filter_result["documents"]  # This is a list of Document objects
            result_data["filter_time"] = filter_result["time_taken"]
            
            # Step 3: Create prompt (pass Document objects)
            prompt_result = prompt_maker.create_prompt(query, filtered_documents)
            prompt = prompt_result["prompt"]
            result_data["prompt_time"] = prompt_result["time_taken"]
            
            # Step 4: Generate answer
            generate_result = generator.generate(prompt)
            answer = generate_result["answer"]
            result_data["generation_time"] = generate_result["time_taken"]
            
            # Calculate total time
            result_data["total_time"] = (
                result_data["retrieval_time"] + 
                result_data["filter_time"] + 
                result_data["prompt_time"] + 
                result_data["generation_time"]
            )
            
            # Store the answer
            result_data["answer"] = answer
            self.results.append(result_data)
        
        # Calculate and log average metrics for this combination
        if combination_metrics:
            print(f"Average metrics for {retrieval_type}+{filter_type}+{prompt_type}+{generator_type}:")
            avg_metrics = {}
            for metric in combination_metrics[0].keys():
                if metric != "map":  # MAP is calculated across all queries together
                    avg_metrics[metric] = sum(cm[metric] for cm in combination_metrics if metric in cm) / len(combination_metrics)
                    print(f"  - {metric}: {avg_metrics[metric]:.4f}")
            
            # Add a summary row with average metrics
            summary_data = {
                "retrieval_type": retrieval_type,
                "filter_type": filter_type,
                "prompt_type": prompt_type,
                "generator_type": generator_type,
                "query": "AVERAGE_METRICS"
            }
            
            for metric, value in avg_metrics.items():
                summary_data[f"retrieval_{metric}"] = value
                
            self.results.append(summary_data)
            
        self.save_results()
    
    def run_all_combinations(self, queries: List[str]):
        start_time = time.time()
        
        total_combinations = len(self.retrieval_options) * len(self.filter_options) * \
                            len(self.prompt_options) * len(self.generator_options)
        
        print(f"[INFO] Running {total_combinations} combinations with {len(queries)} queries each...")
        print(f"[INFO] Total evaluations: {total_combinations * len(queries)}")
        
        for retrieval_type in self.retrieval_options:
            for filter_type in self.filter_options:
                for prompt_type in self.prompt_options:
                    for generator_type in self.generator_options:
                        self.run_combination(
                            retrieval_type, 
                            filter_type, 
                            prompt_type, 
                            generator_type,
                            queries
                        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"[INFO] All combinations completed in {total_time:.2f} seconds")
        print(f"[INFO] Average time per combination: {total_time/total_combinations:.2f} seconds")
    
    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv("results/pipeline_results.csv", index=False)
        
        # Also save a summary of average metrics per configuration
        self.save_summary()
    
    def save_summary(self):
        """Save a summary of average metrics per configuration"""
        if not self.results:
            return
            
        try:
            df = pd.DataFrame(self.results)
            
            # Get all metric columns
            metric_cols = [col for col in df.columns if col.startswith("retrieval_") and col != "retrieval_time"]
            
            if not metric_cols:
                print("No metric columns found in results")
                return
            
            # Filter out rows with average metrics (those with query == "AVERAGE_METRICS")
            avg_df = df[df["query"] == "AVERAGE_METRICS"]
            
            # If we don't have any summary rows, compute them
            if avg_df.empty:
                # Filter any rows with NaN values that might cause issues
                clean_df = df.dropna(subset=['retrieval_type', 'filter_type', 'prompt_type', 'generator_type'])
                
                # Group by configuration and calculate average metrics
                summary = clean_df.groupby(['retrieval_type', 'filter_type', 'prompt_type', 'generator_type'])[metric_cols].mean().reset_index()
            else:
                # Use the pre-computed summary rows
                summary = avg_df[['retrieval_type', 'filter_type', 'prompt_type', 'generator_type'] + metric_cols]
            
            # Save summary to CSV
            summary.to_csv("results/metrics_summary.csv", index=False)
            
            # Print top configurations for each metric
            print("\nTop configurations by metric:")
            for metric in metric_cols:
                if metric not in summary.columns:
                    print(f"Metric {metric} not found in summary")
                    continue
                    
                if summary[metric].isna().all():
                    print(f"All values for {metric} are NaN")
                    continue
                
                try:
                    # Find the row with the maximum value for this metric
                    best_row = summary[summary[metric] == summary[metric].max()]
                    
                    if best_row.empty:
                        print(f"No valid data for metric: {metric}")
                        continue
                    
                    # Get the first row as a dict
                    top_config = best_row.iloc[0].to_dict()
                    
                    # Print the configuration
                    print(f"{metric}: {top_config['retrieval_type']}+{top_config['filter_type']}+"
                          f"{top_config['prompt_type']}+{top_config['generator_type']} = {top_config[metric]:.4f}")
                except Exception as e:
                    print(f"Error processing metric {metric}: {str(e)}")
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
    
    def get_best_configuration(self, prioritize_metric: str = "overall"):
        """
        Return the best configuration based on evaluation metrics
        
        Args:
            prioritize_metric: Which metric to prioritize 
                            ("overall", "precision", "recall", "f1", "ndcg", "mrr", "map")
        Returns:
            Dictionary with the best configuration
        """
        if len(self.results) == 0:
            # load results from file if no results in memory
            try:
                results_df = pd.read_csv("results/pipeline_results.csv")
            except:
                print("No results found. Please run some combinations first.")
                return None
        else:
            # Use the results already in memory
            results_df = pd.DataFrame(self.results)
        
        # Filter out summary rows
        results_df = results_df[results_df["query"] != "AVERAGE_METRICS"]
        
        # Get all retrieval metrics columns
        retrieval_metric_cols = [col for col in results_df.columns if col.startswith("retrieval_") and col != "retrieval_time"]
        
        if not retrieval_metric_cols:
            print("No retrieval metrics found. Falling back to time-based evaluation.")
            return self._get_best_configuration_by_time()
        
        # Ensure all metric columns are numeric
        for col in retrieval_metric_cols:
            # Convert to numeric, non-numeric values become NaN
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        # Drop rows with NaN values in metric columns
        results_df = results_df.dropna(subset=retrieval_metric_cols)
        
        if results_df.empty:
            print("No valid metric data found. Falling back to time-based evaluation.")
            return self._get_best_configuration_by_time()
        
        # Group by configuration and calculate average metrics
        config_performance = results_df.groupby(
            ['retrieval_type', 'filter_type', 'prompt_type', 'generator_type']
        )[retrieval_metric_cols].mean().reset_index()
        
        # Save the detailed metrics for reference
        config_performance.to_csv("results/config_performance.csv", index=False)
        
        # Calculate an overall score (average of normalized metrics)
        if prioritize_metric == "overall":
            # Normalize all metrics to 0-1 range
            for col in retrieval_metric_cols:
                # Check if we have variation in this metric
                if config_performance[col].max() > config_performance[col].min():
                    config_performance[f"{col}_norm"] = (config_performance[col] - config_performance[col].min()) / \
                                                    (config_performance[col].max() - config_performance[col].min())
                else:
                    config_performance[f"{col}_norm"] = 0
            
            # Get normalized columns
            norm_cols = [col for col in config_performance.columns if col.endswith("_norm")]
            
            if norm_cols:
                # Calculate overall score (average of normalized metrics)
                config_performance["overall_score"] = config_performance[norm_cols].mean(axis=1)
                
                # Get best configuration (highest overall score)
                best_config = config_performance.loc[config_performance["overall_score"].idxmax()]
                score_field = "overall_score"
            else:
                print("No normalized metrics could be calculated. Falling back to time-based evaluation.")
                return self._get_best_configuration_by_time()
        else:
            # Use the specific metric directly
            metric_col = f"retrieval_{prioritize_metric}"
            if metric_col in config_performance.columns:
                best_config = config_performance.loc[config_performance[metric_col].idxmax()]
                score_field = metric_col
            else:
                print(f"Metric {prioritize_metric} not found. Falling back to overall evaluation.")
                return self.get_best_configuration(prioritize_metric="overall")
        
        # Save best configuration with metrics
        best_config_dict = {
            'retrieval_type': best_config['retrieval_type'],
            'filter_type': best_config['filter_type'],
            'prompt_type': best_config['prompt_type'],
            'generator_type': best_config['generator_type'],
        }
        
        # Add all metrics to the dictionary
        for col in retrieval_metric_cols:
            best_config_dict[col] = best_config[col]
        
        if prioritize_metric == "overall":
            best_config_dict["overall_score"] = best_config["overall_score"]
        
        # Save to CSV
        best_config_df = pd.DataFrame([best_config_dict])
        best_config_df.to_csv("results/best_config.csv", index=False)
        
        print(f"Best configuration found: {best_config['retrieval_type']} + "
            f"{best_config['filter_type']} + {best_config['prompt_type']} + "
            f"{best_config['generator_type']}")
        print(f"Score ({prioritize_metric}): {best_config[score_field]:.4f}")
        
        return best_config_dict

    def _get_best_configuration_by_time(self):
        """Return the configuration with the lowest average total time"""
        if len(self.results) == 0:
            # load results from file if no results in memory
            try:
                results_df = pd.read_csv("results/pipeline_results.csv")
            except:
                print("No results found. Please run some combinations first.")
                return None
        else:
            # Use the results already in memory
            results_df = pd.DataFrame(self.results)
        
        # Filter out summary rows
        results_df = results_df[results_df["query"] != "AVERAGE_METRICS"]
        
        # Group by configuration and calculate average times
        config_performance = results_df.groupby(
            ['retrieval_type', 'filter_type', 'prompt_type', 'generator_type']
            )['total_time'].mean().reset_index()
        
        # Get best configuration (lowest average time)
        best_config = config_performance.loc[config_performance['total_time'].idxmin()]
        
        return {
            'retrieval_type': best_config['retrieval_type'],
            'filter_type': best_config['filter_type'],
            'prompt_type': best_config['prompt_type'],
            'generator_type': best_config['generator_type'],
            'total_time': best_config['total_time']
        }