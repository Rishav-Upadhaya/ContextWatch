
import asyncio
import json
from pathlib import Path
from contextwatch.services.pipeline.pipeline import build_default_pipeline
from contextwatch.services.ai.training_data import load_system_training_corpus

async def main():
    print("--- LogBERT Fine-tuning Runner ---")
    pipeline = build_default_pipeline()
    
    # We want to use our new consolidated dataset. 
    # Since load_system_training_corpus is hardcoded to SYSTEM_DATASET_FILES,
    # we will manually load our consolidated file to ensure it's used.
    
    consolidated_path = Path("/home/rishavupadhaya/hermes-sandbox/ContextWatch/contextwatch/data/consolidated_training_data.jsonl")
    
    print(f"Loading data from: {consolidated_path}")
    
    normal_logs = []
    anomaly_logs = []
    
    with consolidated_path.open("r") as f:
        for line in f:
            log = json.loads(line)
            # The logic from training_data.py: label 1 if is_anomaly is True
            if log.get("is_anomaly") is True:
                anomaly_logs.append(log)
            else:
                normal_logs.append(log)
                
    print(f"Loaded {len(normal_logs)} normal logs and {len(anomaly_logs)} anomaly logs.")
    
    if not anomaly_logs or not normal_logs:
        print("ERROR: Insufficient data for balanced training. Aborting.")
        return

    print("Starting fine-tuning (this may take a few minutes)...")
    
    # We call the pipeline's finetune method
    # Note: In a real environment, this would run on a GPU.
    metrics = await pipeline.finetune(
        raw_logs=normal_logs + anomaly_logs, 
        epochs=3, 
        learning_rate=0.001,
        use_system_data=False
    )
    
    print("\n--- Fine-tuning Complete! ---")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print("New weights have been updated in the model.")

if __name__ == '__main__':
    asyncio.run(main())
