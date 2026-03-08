import os
import json

prompts = [
    ("prompt_01", "General", "a photo of a <class word> and a <class word>"),

    # Clothing
    ("prompt_02", "Clothing", "a <class word> and a <class word> wearing a Superman outfit"),
    ("prompt_03", "Clothing", "a <class word> and a <class word> wearing a spacesuit"),
    ("prompt_04", "Clothing", "a <class word> and a <class word> wearing a red sweater"),
    ("prompt_05", "Clothing", "a <class word> and a <class word> wearing a purple wizard outfit"),
    ("prompt_06", "Clothing", "a <class word> and a <class word> wearing a blue hoodie"),

    # Accessory
    ("prompt_07", "Accessory", "a <class word> and a <class word> wearing headphones"),
    ("prompt_08", "Accessory", "a <class word> and a <class word> with red hair"),
    ("prompt_09", "Accessory", "a <class word> and a <class word> wearing headphones with red hair"),
    ("prompt_10", "Accessory", "a <class word> and a <class word> wearing a Christmas hat"),
    ("prompt_11", "Accessory", "a <class word> and a <class word> wearing sunglasses"),
    ("prompt_12", "Accessory", "a <class word> and a <class word> wearing sunglasses and necklace"),
    ("prompt_13", "Accessory", "a <class word> and a <class word> wearing a blue cap"),
    ("prompt_14", "Accessory", "a <class word> and a <class word> wearing a doctoral cap"),
    ("prompt_15", "Accessory", "a <class word> and a <class word> with white hair, wearing glasses"),

    # Action
    ("prompt_16", "Action", "a <class word> and a <class word> in a helmet and vest riding a motorcycle"),
    ("prompt_17", "Action", "a <class word> and a <class word> holding a bottle of red wine"),
    ("prompt_18", "Action", "a <class word> and a <class word> driving a bus in the desert"),
    ("prompt_19", "Action", "a <class word> and a <class word> playing basketball"),
    ("prompt_20", "Action", "a <class word> and a <class word> playing the violin"),
    ("prompt_21", "Action", "a <class word> and a <class word> piloting a spaceship"),
    ("prompt_22", "Action", "a <class word> and a <class word> riding a horse"),
    ("prompt_23", "Action", "a <class word> and a <class word> coding in front of a computer"),
    ("prompt_24", "Action", "a <class word> and a <class word> playing the guitar"),

    # Expression
    ("prompt_25", "Expression", "a <class word> and a <class word> laughing on the lawn"),
    ("prompt_26", "Expression", "a <class word> and a <class word> frowning at the camera"),
    ("prompt_27", "Expression", "a <class word> and a <class word> happily smiling, looking at the camera"),
    ("prompt_28", "Expression", "a <class word> and a <class word> crying disappointedly, with tears flowing"),
    ("prompt_29", "Expression", "a <class word> and a <class word> wearing sunglasses"),

    # View
    ("prompt_30", "View", "a <class word> and a <class word> playing the guitar in the view of left side"),
    ("prompt_31", "View", "a <class word> and a <class word> holding a bottle of red wine, upper body"),
    ("prompt_32", "View", "a <class word> and a <class word> wearing sunglasses and necklace, close-up, in the view of right side"),
    ("prompt_33", "View", "a <class word> and a <class word> riding a horse, in the view of the top"),
    ("prompt_34", "View", "a <class word> and a <class word> wearing a doctoral cap, upper body, with the left side of the face facing the camera"),
    ("prompt_35", "View", "a <class word> and a <class word> crying disappointedly, with tears flowing, with left side of the face facing the camera"),
    ("prompt_36", "View", "a <class word> and a <class word> sitting in front of the camera, with a beautiful purple sunset at the beach in the background"),

    # Background
    ("prompt_37", "Background", "a <class word> and a <class word> swimming in the pool"),
    ("prompt_38", "Background", "a <class word> and a <class word> climbing a mountain"),
    ("prompt_39", "Background", "a <class word> and a <class word> skiing on the snowy mountain"),
    ("prompt_40", "Background", "a <class word> and a <class word> in the snow"),
    ("prompt_41", "Background", "a <class word> and a <class word> in space wearing a spacesuit")
]

os.makedirs("prompts", exist_ok=True)

for pid, category, text in prompts:
    data = {
        "prompt_id": pid,
        "category": category,
        "text": text
    }
    with open(f"prompts/{pid}.json", "w") as f:
        json.dump(data, f, indent=4)

print("All 41 prompt JSON files created in the 'prompts/' folder.")