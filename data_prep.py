# data_prep.py
# (Your SchemaDatasetPreparer class from the prompt goes here, unchanged)
import json
import os
import glob
from typing import List, Dict, Any
import random

# --- PASTE YOUR ENTIRE SchemaDatasetPreparer CLASS HERE ---


class SchemaDatasetPreparer:
    """
    Prepares a dataset from JSON schema files for training.
    """

    def __init__(self, json_files_path: str = "data"):
        self.json_files_path = json_files_path

    def load_json_files(self) -> List[Dict[str, Any]]:
        """
        Loads all JSON files from the specified directory.
        """
        files = glob.glob(os.path.join(self.json_files_path, "*.json"))
        data = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                try:
                    data.append(
                        {"data": json.load(f), "source_file": os.path.basename(file)})
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        return data

    def create_dataset(self) -> List[Dict[str, Any]]:
        """
        Creates a dataset with instructions and outputs from the loaded JSON files.
        """
        data = self.load_json_files()
        dataset = []
        for item in data:
            schema = item["data"]
            source_file = item["source_file"]
            instruction = f"Given the following JSON schema, extract the field names and their types:\n{json.dumps(schema, indent=2)}"
            output = self.extract_fields(schema)
            dataset.append({
                "instruction": instruction,
                "output": output,
                "source_file": source_file
            })
        random.shuffle(dataset)
        return dataset

    def extract_fields(self, schema: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts field names and their types from a JSON schema.
        """
        fields = {}
        if "properties" in schema:
            for field, props in schema["properties"].items():
                field_type = props.get("type", "unknown")
                fields[field] = field_type
        return fields

    def save_dataset(self, dataset: List[Dict[str, Any]], output_file: str = "training_dataset.json"):
        """
        Saves the dataset to a JSON file.
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)


def main():
    """
    Main function to prepare and save the dataset.
    """
    # Initialize the preparer to look for JSON files in the 'data/' subdirectory
    preparer = SchemaDatasetPreparer(json_files_path="data")

    # Create the dataset
    dataset = preparer.create_dataset()

    if dataset:
        # Save the dataset to the main project directory
        preparer.save_dataset(dataset, output_file="training_dataset.json")

        print("\n" + "="*50)
        print("SAMPLE TRAINING EXAMPLE:")
        print("="*50)
        sample = dataset[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Output: {json.dumps(sample['output'], indent=2)}")
        print(f"Source: {sample['source_file']}")
    else:
        print("Failed to create dataset!")


if __name__ == "__main__":
    main()
