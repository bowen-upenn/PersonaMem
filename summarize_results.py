import os
import json
from collections import defaultdict
import re


def summarize_evaluation_results_by_persona(directory, model):
    persona_summary = {str(i): defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0}) for i in range(10)}

    for filename in os.listdir(directory):
        if not filename.endswith(".json") or not filename.startswith(model) or filename.endswith("_full.json"):
            continue

        match = re.search(r"persona(\d)", filename)
        if not match:
            continue

        persona_id = match.group(1)

        file_path = os.path.join(directory, filename)

        with open(file_path, "r") as file:
            data = json.load(file)

            for key, value in data.items():
                if isinstance(value, dict) and "correct" in value and "total" in value:
                    persona_summary[persona_id][key]["correct"] += value.get("correct", 0)
                    persona_summary[persona_id][key]["total"] += value.get("total", 0)

    # Compute overall accuracy for each persona and key
    for persona_id, summary in persona_summary.items():
        for key, value in summary.items():
            if value["total"] > 0:
                value["accuracy"] = round((value["correct"] / value["total"]) * 100, 2)

    return {"Persona_" + str(persona_id): dict(summary) for persona_id, summary in persona_summary.items()}


# Example usage
if __name__ == "__main__":
    models = ['gpt-4o', 'gpt-4o-mini']
    eval_summary = {}
    for model in models:
        print(f"Model: {model}")
        eval_summary_by_persona = summarize_evaluation_results_by_persona(directory="./data/eval/", model=model)
        for persona_id, summary in eval_summary_by_persona.items():
            print(f"Persona {persona_id}:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        eval_summary[model] = eval_summary_by_persona

    # Save results into a JSON file
    with open("data/eval/eval_summary.json", "w") as file:
        json.dump(eval_summary, file, indent=4)
