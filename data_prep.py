import json
import os
import glob
from typing import List, Dict, Any
import random
import itertools

# --- CONFIGURATION ---
# Set the desired number of training examples. The script will run until this target is met.
TARGET_EXAMPLE_COUNT = 8500 

class SchemaDatasetPreparer:
    """
    A class to prepare a rich, augmented training dataset from a small set of JSON schema files.
    It uses various data augmentation techniques to reach a target number of examples.
    """
    def __init__(self, json_files_path: str = "data"):
        """
        Initialize the dataset preparer.
        Args:
            json_files_path: Path to the directory containing JSON schema files.
        """
        self.json_files_path = json_files_path
        self.schemas = self._load_and_parse_schemas()

    def _load_and_parse_schemas(self) -> List[Dict[str, Any]]:
        """
        Loads all JSON schema files from the directory and pre-parses them for easier access.
        This is a one-time operation during initialization.
        """
        json_files = glob.glob(os.path.join(self.json_files_path, "*.json"))
        json_files = [f for f in json_files if not os.path.basename(f).startswith('training_dataset')]
        
        print(f"Found {len(json_files)} JSON schema files in '{self.json_files_path}'.")
        
        parsed_schemas = []
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                    
                    # --- Schema Normalization ---
                    # Handle cases where the top-level is a list of devices
                    if isinstance(schema, list):
                        schema = {'devices': schema, 'connections': [], 'metadata': {}}
                    
                    if 'devices' not in schema:
                        print(f"⚠️  Warning: Skipping {os.path.basename(file_path)} - no 'devices' key found.")
                        continue

                    # Pre-process and store useful lookups
                    devices_dict = {d['id']: d for d in schema.get('devices', [])}
                    
                    parsed_schemas.append({
                        'filename': os.path.basename(file_path),
                        'schema': schema,
                        'devices': schema.get('devices', []),
                        'connections': schema.get('connections', []),
                        'devices_dict': devices_dict,
                        'manufacturers': list(set(d.get('manufacturer') for d in schema.get('devices', []) if d.get('manufacturer'))),
                        'categories': list(set(d.get('category') for d in schema.get('devices', []) if d.get('category'))),
                    })
                    print(f"✓ Loaded and parsed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
        
        if not parsed_schemas:
            raise ValueError("No valid schema files could be loaded. Please check the 'data' directory and the JSON file format.")
            
        return parsed_schemas

    # --- AUGMENTATION STRATEGIES ---

    def generate_connection_examples(self, schema_info: Dict) -> List[Dict]:
        """Generates detailed examples for every single connection in the schema."""
        examples = []
        templates = [
            "Connect {source_manufacturer} {source_model} (ID: {source_id}) port '{source_port}' to {dest_manufacturer} {dest_model} (ID: {dest_id}) port '{dest_port}'.",
            "Establish a {signal_type} link between port '{source_port}' on device {source_id} and port '{dest_port}' on device {dest_id}.",
            "Create a connection from {source_id} to {dest_id} using a {signal_type} signal.",
            "Wire the output of {source_id} to the input of {dest_id}."
        ]

        for conn in schema_info['connections']:
            source_id = conn.get('source', {}).get('device_id')
            dest_id = conn.get('destination', {}).get('device_id')
            source_port = conn.get('source', {}).get('port_name')
            dest_port = conn.get('destination', {}).get('port_name')

            if not all([source_id, dest_id, source_port, dest_port]):
                continue

            source_device = schema_info['devices_dict'].get(source_id)
            dest_device = schema_info['devices_dict'].get(dest_id)

            if not source_device or not dest_device:
                continue
            
            # Create the instruction
            instruction = random.choice(templates).format(
                source_manufacturer=source_device.get('manufacturer', 'N/A'),
                source_model=source_device.get('model', 'N/A'),
                source_id=source_id,
                source_port=source_port,
                dest_manufacturer=dest_device.get('manufacturer', 'N/A'),
                dest_model=dest_device.get('model', 'N/A'),
                dest_id=dest_id,
                dest_port=dest_port,
                signal_type=conn.get('signal_type', 'standard')
            )
            
            # Create the output JSON
            output = {
                "devices": [source_device, dest_device],
                "connections": [conn]
            }

            examples.append({"instruction": instruction, "output": output})
        return examples

    def generate_combinatorial_examples(self, schema_info: Dict) -> List[Dict]:
        """
        Generates examples from small combinations of devices (2 or 3).
        This is a powerful way to create many diverse, smaller-scale examples.
        """
        examples = []
        if len(schema_info['devices']) < 3:
            return []

        # Take 5 random combinations of 2 or 3 devices to generate examples from
        for _ in range(5): 
            k = random.choice([2, 3])
            try:
                device_subset = random.sample(schema_info['devices'], k)
            except ValueError:
                continue

            device_ids = {d['id'] for d in device_subset}
            
            # Find connections that are exclusively between these devices
            relevant_connections = [
                c for c in schema_info['connections']
                if c.get('source', {}).get('device_id') in device_ids and c.get('destination', {}).get('device_id') in device_ids
            ]

            # Create instruction
            device_names = " and ".join([f"{d.get('manufacturer', '')} {d.get('model', '')} (ID: {d.get('id')})" for d in device_subset])
            instruction = f"Create a small system containing only these devices: {device_names} and their direct connections."
            
            # Create output
            output = {
                "devices": device_subset,
                "connections": relevant_connections
            }
            
            examples.append({"instruction": instruction, "output": output})
        return examples

    def generate_attribute_query_examples(self, schema_info: Dict) -> List[Dict]:
        """Generates examples querying devices by their specific attributes."""
        examples = []
        templates = {
            'manufacturer': "Find all devices made by {value}.",
            'category': "List all equipment in the {value} category.",
            'rack_position': "What device is located at rack position {value}?",
        }

        # Generate one query for each attribute type
        for attr, template in templates.items():
            # Find a device that has this attribute
            eligible_devices = [d for d in schema_info['devices'] if d.get(attr)]
            if not eligible_devices:
                continue
            
            target_device = random.choice(eligible_devices)
            value = target_device[attr]
            
            # Create instruction
            instruction = template.format(value=value)
            
            # Find all devices that match this query
            matching_devices = [d for d in schema_info['devices'] if d.get(attr) == value]
            
            # Create output
            output = {
                "query_attribute": attr,
                "query_value": value,
                "count": len(matching_devices),
                "devices": matching_devices
            }
            examples.append({"instruction": instruction, "output": output})
        return examples

    def generate_negative_examples(self, schema_info: Dict) -> List[Dict]:
        """Generates examples for queries that should return no results."""
        examples = []
        
        # Query for a non-existent manufacturer
        non_existent_mfr = "NonExistentCorp"
        instruction_mfr = f"Find all devices made by {non_existent_mfr}."
        output_mfr = {"query_attribute": "manufacturer", "query_value": non_existent_mfr, "count": 0, "devices": []}
        examples.append({"instruction": instruction_mfr, "output": output_mfr})

        # Query for a non-existent category
        non_existent_cat = "UnderwaterBasketWeaving"
        instruction_cat = f"List all equipment in the {non_existent_cat} category."
        output_cat = {"query_attribute": "category", "query_value": non_existent_cat, "count": 0, "devices": []}
        examples.append({"instruction": instruction_cat, "output": output_cat})
        
        return examples

    def generate_full_system_summary_examples(self, schema_info: Dict) -> List[Dict]:
        """Generates high-level summary examples for the entire schema."""
        examples = []
        templates = [
            "Give me a full summary of the broadcast system.",
            "Describe the entire system architecture.",
            "Generate a complete inventory list for the facility."
        ]
        
        instruction = random.choice(templates)
        
        # The output is the entire original schema
        output = schema_info['schema']
        
        examples.append({"instruction": instruction, "output": output})
        return examples

    def create_dataset(self) -> List[Dict[str, Any]]:
        """
        Main dataset creation loop.
        It repeatedly calls different generation functions until the target number of examples is reached.
        """
        print(f"\nStarting dataset generation. Target: {TARGET_EXAMPLE_COUNT} examples.")
        
        training_examples = []
        
        # List of all the different generator functions we can call
        generators = [
            self.generate_connection_examples,
            self.generate_combinatorial_examples,
            self.generate_attribute_query_examples,
            self.generate_negative_examples,
            self.generate_full_system_summary_examples,
        ]

        # Loop until we have enough examples
        loop_count = 0
        while len(training_examples) < TARGET_EXAMPLE_COUNT:
            # Randomly select a schema to work with
            schema_info = random.choice(self.schemas)
            
            # Randomly select a generator function
            generator = random.choice(generators)
            
            # Generate new examples
            new_examples = generator(schema_info)
            
            # Add a 'source_file' key to each example for traceability
            for ex in new_examples:
                ex['source_file'] = schema_info['filename']
                training_examples.append(ex)

            loop_count += 1
            if loop_count % 250 == 0:
                print(f"  ... generated {len(training_examples)} / {TARGET_EXAMPLE_COUNT} examples ...")

        print(f"Target reached. Total examples generated: {len(training_examples)}")
        
        # Shuffle the final dataset for randomness
        random.shuffle(training_examples)
        
        return training_examples[:TARGET_EXAMPLE_COUNT]

    def save_dataset(self, dataset: List[Dict[str, Any]], output_file: str = "training_dataset.json"):
        """Saves the generated dataset to a JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Dataset successfully saved to: {output_file}")
            print(f"✓ Total examples: {len(dataset)}")
        except Exception as e:
            print(f"✗ Error saving dataset: {e}")


def main():
    """
    Main function to initialize the preparer, create the dataset, and save it.
    """
    try:
        # Initialize the preparer. It will automatically load schemas from the 'data/' directory.
        preparer = SchemaDatasetPreparer(json_files_path="data")

        # Create the dataset by running the augmentation loop.
        dataset = preparer.create_dataset()

        if dataset:
            # Save the final dataset.
            preparer.save_dataset(dataset, output_file="training_dataset.json")

            # Print a sample to verify the output.
            print("\n" + "="*50)
            print("SAMPLE TRAINING EXAMPLE:")
            print("="*50)
            sample = dataset[0]
            print(f"Instruction: {sample['instruction']}")
            print(f"Output: {json.dumps(sample['output'], indent=2)}")
            print(f"Source: {sample['source_file']}")
            print("="*50)
        else:
            print("Could not generate a dataset. Please check for errors above.")

    except Exception as e:
        print(f"\nAn error occurred during dataset preparation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your JSON schema files are in a folder named 'data' inside your project directory.")
        print("2. Make sure the JSON files are valid and contain a 'devices' key with a list of device objects.")


if __name__ == "__main__":
    main()
