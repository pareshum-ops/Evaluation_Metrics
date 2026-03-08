#!/usr/bin/env python3
import json
import os
from PIL import Image
import gradio as gr

INDEX_FILE = "index.json"
OUTPUT_DIR = "outputs"
INPUT_DIR = "inputs"

# ---------------------------------------------------------
# CLEAN INPUT ID (remove extension ONLY)
# ---------------------------------------------------------
def clean_input_id(filename):
    return os.path.splitext(filename)[0]

# ---------------------------------------------------------
# LOAD DATASET INDEX
# ---------------------------------------------------------
with open(INDEX_FILE, "r") as f:
    index_data = json.load(f)

input_map = {}
original_name_map = {}

for item in index_data["inputs"]:
    original = item["id"]
    cleaned = clean_input_id(original)
    input_map[cleaned] = item["metrics"]
    original_name_map[cleaned] = original

# ---------------------------------------------------------
# FIND OUTPUT IMAGE
# ---------------------------------------------------------
def find_output_image(cleaned_id, prompt_id):
    folder = f"{OUTPUT_DIR}/{cleaned_id}"

    if not os.path.exists(folder):
        return None

    for ext in [".webp", ".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(folder, prompt_id + ext)
        if os.path.exists(candidate):
            return candidate

    for file in os.listdir(folder):
        if prompt_id in file:
            return os.path.join(folder, file)

    return None

# ---------------------------------------------------------
# LOAD METRICS JSON
# ---------------------------------------------------------
def load_metrics(cleaned_id):
    with open(input_map[cleaned_id], "r") as f:
        return json.load(f)

# ---------------------------------------------------------
# NAVIGATION HELPERS
# ---------------------------------------------------------
def get_prompt_list(cleaned_id):
    data = load_metrics(cleaned_id)
    return [item["prompt_id"] for item in data["results"]]

def next_prompt(cleaned_id, current_prompt):
    prompts = get_prompt_list(cleaned_id)
    idx = prompts.index(current_prompt)
    return prompts[(idx + 1) % len(prompts)]

def prev_prompt(cleaned_id, current_prompt):
    prompts = get_prompt_list(cleaned_id)
    idx = prompts.index(current_prompt)
    return prompts[(idx - 1) % len(prompts)]

# ---------------------------------------------------------
# LOAD VIEW (single input)
# ---------------------------------------------------------
def load_view(cleaned_id, prompt_id):
    data = load_metrics(cleaned_id)

    original_filename = original_name_map[cleaned_id]
    input_img = Image.open(f"{INPUT_DIR}/{original_filename}")

    output_path = find_output_image(cleaned_id, prompt_id)
    output_img = Image.open(output_path) if output_path else None

    entry = next(item for item in data["results"] if item["prompt_id"] == prompt_id)

    metrics = {
        "ArcFace Face1": entry["arcface_cosine_face1"],
        "ArcFace Face2": entry["arcface_cosine_face2"],
        "CLIP-T": entry["clip_t"],
        "PSNR": entry["psnr"],
        "SSIM": entry["ssim"]
    }

    return (
        input_img,
        output_img,
        entry["prompt_text"],
        entry["category"],
        metrics,
        prompt_id
    )

# ---------------------------------------------------------
# LOAD VIEW FOR COMPARE MODE (two inputs, one prompt)
# ---------------------------------------------------------
def load_compare_view(cleaned_id1, cleaned_id2, prompt_id):
    # Input 1
    data1 = load_metrics(cleaned_id1)
    original_filename1 = original_name_map[cleaned_id1]
    input_img1 = Image.open(f"{INPUT_DIR}/{original_filename1}")
    output_path1 = find_output_image(cleaned_id1, prompt_id)
    output_img1 = Image.open(output_path1) if output_path1 else None
    entry1 = next(item for item in data1["results"] if item["prompt_id"] == prompt_id)

    metrics1 = {
        "ArcFace Face1": entry1["arcface_cosine_face1"],
        "ArcFace Face2": entry1["arcface_cosine_face2"],
        "CLIP-T": entry1["clip_t"],
        "PSNR": entry1["psnr"],
        "SSIM": entry1["ssim"]
    }

    # Input 2
    data2 = load_metrics(cleaned_id2)
    original_filename2 = original_name_map[cleaned_id2]
    input_img2 = Image.open(f"{INPUT_DIR}/{original_filename2}")
    output_path2 = find_output_image(cleaned_id2, prompt_id)
    output_img2 = Image.open(output_path2) if output_path2 else None
    entry2 = next(item for item in data2["results"] if item["prompt_id"] == prompt_id)

    metrics2 = {
        "ArcFace Face1": entry2["arcface_cosine_face1"],
        "ArcFace Face2": entry2["arcface_cosine_face2"],
        "CLIP-T": entry2["clip_t"],
        "PSNR": entry2["psnr"],
        "SSIM": entry2["ssim"]
    }

    # Shared prompt text & category
    prompt_text = entry1["prompt_text"]
    category = entry1["category"]

    # Side-by-side metrics table rows
    metrics_rows = []
    for metric in metrics1.keys():
        metrics_rows.append([metric, metrics1[metric], metrics2[metric]])

    return (
        input_img1,
        output_img1,
        input_img2,
        output_img2,
        prompt_text,
        category,
        metrics_rows
    )

# ---------------------------------------------------------
# BUILD UI
# ---------------------------------------------------------
def build_ui():
    cleaned_ids = list(input_map.keys())

    first_metrics = load_metrics(cleaned_ids[0])
    prompt_ids = [item["prompt_id"] for item in first_metrics["results"]]

    with gr.Blocks(title="Dual Identity Validation Viewer") as demo:

        gr.Markdown("# **Dual Identity Image Validation Viewer**")

        compare_toggle = gr.Checkbox(label="Enable Compare Mode", value=False)

        # -------------------------
        # NORMAL MODE GROUP
        # -------------------------
        with gr.Group(visible=True) as normal_group:
            with gr.Row():
                input_dropdown = gr.Dropdown(
                    choices=cleaned_ids,
                    label="Select Input Image"
                )
                prompt_dropdown = gr.Dropdown(
                    choices=prompt_ids,
                    label="Select Prompt"
                )
                load_button = gr.Button("Load")

            with gr.Row():
                prev_button = gr.Button("← Previous")
                next_button = gr.Button("Next →")

            with gr.Row():
                input_image = gr.Image(label="Input Image")
                output_image = gr.Image(label="Generated Output")

            prompt_text = gr.Textbox(label="Prompt Text")
            category_text = gr.Textbox(label="Category")
            metrics_table = gr.JSON(label="Metrics")

            current_prompt_state = gr.State(prompt_ids[0])

            def update_prompt_list(cleaned_id):
                prompts = get_prompt_list(cleaned_id)
                return gr.Dropdown(choices=prompts), prompts[0]

            input_dropdown.change(
                fn=update_prompt_list,
                inputs=input_dropdown,
                outputs=[prompt_dropdown, current_prompt_state]
            )

            load_button.click(
                fn=load_view,
                inputs=[input_dropdown, prompt_dropdown],
                outputs=[
                    input_image,
                    output_image,
                    prompt_text,
                    category_text,
                    metrics_table,
                    current_prompt_state
                ]
            )

            def go_next(cleaned_id, current_prompt):
                return next_prompt(cleaned_id, current_prompt)

            next_button.click(
                fn=go_next,
                inputs=[input_dropdown, current_prompt_state],
                outputs=prompt_dropdown
            ).then(
                fn=load_view,
                inputs=[input_dropdown, prompt_dropdown],
                outputs=[
                    input_image,
                    output_image,
                    prompt_text,
                    category_text,
                    metrics_table,
                    current_prompt_state
                ]
            )

            def go_prev(cleaned_id, current_prompt):
                return prev_prompt(cleaned_id, current_prompt)

            prev_button.click(
                fn=go_prev,
                inputs=[input_dropdown, current_prompt_state],
                outputs=prompt_dropdown
            ).then(
                fn=load_view,
                inputs=[input_dropdown, prompt_dropdown],
                outputs=[
                    input_image,
                    output_image,
                    prompt_text,
                    category_text,
                    metrics_table,
                    current_prompt_state
                ]
            )

        # -------------------------
        # COMPARE MODE GROUP
        # -------------------------
        with gr.Group(visible=False) as compare_group:
            gr.Markdown("## Compare Mode (Two Inputs, One Prompt)")

            with gr.Row():
                input1_dropdown = gr.Dropdown(
                    choices=cleaned_ids,
                    label="Select Input Image 1"
                )
                input2_dropdown = gr.Dropdown(
                    choices=cleaned_ids,
                    label="Select Input Image 2"
                )
                prompt_compare_dropdown = gr.Dropdown(
                    choices=prompt_ids,
                    label="Select Prompt (common)"
                )

            with gr.Row():
                prev_compare_button = gr.Button("← Previous Prompt")
                next_compare_button = gr.Button("Next Prompt →")

            compare_load_button = gr.Button("Load Comparison")

            # 2×2 grid
            with gr.Row():
                with gr.Column():
                    compare_input1 = gr.Image(label="Input Image 1")
                    compare_output1 = gr.Image(label="Output Image 1")
                with gr.Column():
                    compare_input2 = gr.Image(label="Input Image 2")
                    compare_output2 = gr.Image(label="Output Image 2")

            compare_prompt_text = gr.Textbox(label="Prompt Text (common)")
            compare_category_text = gr.Textbox(label="Category (common)")

            compare_metrics_table = gr.Dataframe(
                headers=["Metric", "Image 1", "Image 2"],
                label="Metrics Comparison",
                interactive=False
            )

            # Update prompt list when Input 1 changes
            def update_compare_prompt_list(cleaned_id):
                prompts = get_prompt_list(cleaned_id)
                return gr.Dropdown(choices=prompts)

            input1_dropdown.change(
                fn=update_compare_prompt_list,
                inputs=input1_dropdown,
                outputs=prompt_compare_dropdown
            )

            # Load comparison
            def load_compare(cleaned_id1, cleaned_id2, prompt_id):
                return load_compare_view(cleaned_id1, cleaned_id2, prompt_id)

            compare_load_button.click(
                fn=load_compare,
                inputs=[input1_dropdown, input2_dropdown, prompt_compare_dropdown],
                outputs=[
                    compare_input1,
                    compare_output1,
                    compare_input2,
                    compare_output2,
                    compare_prompt_text,
                    compare_category_text,
                    compare_metrics_table
                ]
            )

            # NEXT / PREVIOUS prompt navigation
            def compare_next(cleaned_id1, prompt_id):
                return next_prompt(cleaned_id1, prompt_id)

            def compare_prev(cleaned_id1, prompt_id):
                return prev_prompt(cleaned_id1, prompt_id)

            next_compare_button.click(
                fn=compare_next,
                inputs=[input1_dropdown, prompt_compare_dropdown],
                outputs=prompt_compare_dropdown
            ).then(
                fn=load_compare,
                inputs=[input1_dropdown, input2_dropdown, prompt_compare_dropdown],
                outputs=[
                    compare_input1,
                    compare_output1,
                    compare_input2,
                    compare_output2,
                    compare_prompt_text,
                    compare_category_text,
                    compare_metrics_table
                ]
            )

            prev_compare_button.click(
                fn=compare_prev,
                inputs=[input1_dropdown, prompt_compare_dropdown],
                outputs=prompt_compare_dropdown
            ).then(
                fn=load_compare,
                inputs=[input1_dropdown, input2_dropdown, prompt_compare_dropdown],
                outputs=[
                    compare_input1,
                    compare_output1,
                    compare_input2,
                    compare_output2,
                    compare_prompt_text,
                    compare_category_text,
                    compare_metrics_table
                ]
            )

        # -------------------------
        # TOGGLE BETWEEN MODES
        # -------------------------
        def toggle_mode(enabled):
            if enabled:
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)

        compare_toggle.change(
            fn=toggle_mode,
            inputs=compare_toggle,
            outputs=[normal_group, compare_group]
        )

    return demo

demo = build_ui()

if __name__ == "__main__":
    demo.launch(share=True)
