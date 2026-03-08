import os
import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
PROMPT_DIR = "prompts"
METRICS_DIR = "metrics"
METRIC_CSV = "metrics_raw/metrics.csv"
INDEX_FILE = "index.json"

os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------------
# LOAD ALL PROMPT JSON FILES
# ---------------------------------------------------------

def load_prompts():
    prompts = {}
    for file in sorted(Path(PROMPT_DIR).glob("prompt_*.json")):
        with open(file, "r") as f:
            data = json.load(f)
            prompts[data["prompt_id"]] = data
    return prompts

prompts = load_prompts()

# ---------------------------------------------------------
# LOAD METRICS CSV
# ---------------------------------------------------------

metric_df = pd.read_csv(METRIC_CSV)

def load_metrics_for(input_id, prompt_id):
    row = metric_df[
        (metric_df["input_id"] == input_id) &
        (metric_df["prompt_id"] == prompt_id)
    ]

    if row.empty:
        return {
            "arcface_cosine_face1": None,
            "arcface_cosine_face2": None,
            "clip_t": None,
            "psnr": None,
            "ssim": None
        }

    row = row.iloc[0]

    return {
        "arcface_cosine_face1": row["arcface_cosine_face1"],
        "arcface_cosine_face2": row["arcface_cosine_face2"],
        "clip_t": row["clip_t"],
        "psnr": row["psnr"],
        "ssim": row["ssim"]
    }

# ---------------------------------------------------------
# BUILD METRICS JSON FOR EACH INPUT IMAGE
# ---------------------------------------------------------

def build_metrics_json(input_id):
    input_path = f"{INPUT_DIR}/{input_id}"

    results = []

    for prompt_id, prompt_data in prompts.items():
        output_path = f"{OUTPUT_DIR}/{input_id}/{prompt_id}.png"

        metric_values = load_metrics_for(input_id, prompt_id)

        entry = {
            "prompt_id": prompt_id,
            "category": prompt_data["category"],
            "prompt_text": prompt_data["text"],
            "output_image": output_path,
            "arcface_cosine_face1": metric_values["arcface_cosine_face1"],
            "arcface_cosine_face2": metric_values["arcface_cosine_face2"],
            "clip_t": metric_values["clip_t"],
            "psnr": metric_values["psnr"],
            "ssim": metric_values["ssim"]
        }

        results.append(entry)

    json_data = {
        "input_id": input_id,
        "input_image": f"{INPUT_DIR}/{input_id}",
        "num_prompts": len(prompts),
        "results": results
    }

    with open(f"{METRICS_DIR}/{input_id}.json", "w") as f:
        json.dump(json_data, f, indent=4)

# ---------------------------------------------------------
# SCAN INPUT IMAGES AND BUILD DATASET
# ---------------------------------------------------------

def build_dataset():
    input_files = sorted(os.listdir(INPUT_DIR))

    index_entries = []

    for filename in input_files:
        input_id = filename  # e.g., img_001.jpg

        print(f"Processing {input_id}...")

        build_metrics_json(input_id)

        index_entries.append({
            "id": input_id,
            "path": f"{INPUT_DIR}/{input_id}",
            "metrics": f"{METRICS_DIR}/{input_id}.json"
        })

    index_json = {
        "num_inputs": len(index_entries),
        "num_prompts": len(prompts),
        "categories": sorted(list({p["category"] for p in prompts.values()})),
        "inputs": index_entries
    }

    with open(INDEX_FILE, "w") as f:
        json.dump(index_json, f, indent=4)

    print("\nDataset build complete!")
    print(f"- Metrics saved in: {METRICS_DIR}/")
    print(f"- Index file: {INDEX_FILE}")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    build_dataset()