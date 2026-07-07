import time
import argparse
import logging
import polars as pl
from evaluator.rag_pipelines.naive_rag import NaiveRAG
from evaluator.rag_pipelines.agentic_rag import AgenticRAG
from evaluator.utils.metrics import evaluate_faithfulness, evaluate_context_precision

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("BenchmarkHarness")

def run_benchmark(queries: list[str]) -> pl.DataFrame:
    pipelines = {
        "NaiveRAG": NaiveRAG(),
        "AgenticRAG": AgenticRAG()
    }
    
    results = []
    
    for pipe_name, pipeline in pipelines.items():
        logger.info(f"--- Starting Evaluation for {pipe_name} ---")
        for query in queries:
            start_time = time.time()
            
            # Execute Pipeline
            response = pipeline.execute(query)
            latency = time.time() - start_time
            
            # Evaluate using LLM-as-a-Judge
            faithfulness = evaluate_faithfulness(response.query, response.retrieved_contexts, response.generated_answer)
            precision = evaluate_context_precision(response.query, response.retrieved_contexts)
            
            results.append({
                "Pipeline": pipe_name,
                "Query": query,
                "Context Precision": precision,
                "Faithfulness": faithfulness,
                "Latency (s)": round(latency, 2),
                "Tokens": response.metadata.get("token_usage", {}).get("total_tokens", 0)
            })
            
    # Compile and output via Polars
    df = pl.DataFrame(results)
    
    summary = df.group_by("Pipeline").agg([
        pl.col("Context Precision").mean().round(3).alias("Avg Precision"),
        pl.col("Faithfulness").mean().round(3).alias("Avg Faithfulness"),
        pl.col("Latency (s)").mean().round(2).alias("Avg Latency (s)"),
        pl.col("Tokens").mean().round(0).alias("Avg Tokens")
    ])
    
    logger.info(f"\nFINAL BENCHMARK SUMMARY:\n{summary}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Benchmarks")
    parser.add_argument("--queries", nargs="+", default=[
        "What are the strict physiological boundaries for patient eligibility in clinical protocol Alpha?",
        "Explain the transaction state commit constraints when bypassing standard neural generations."
    ])
    args = parser.parse_args()
    
    run_benchmark(args.queries)
