import tiktoken
import json
import yaml
import os

def load_config():
    # Adjust path to find config.yaml relative to this script or project root
    # Assuming the script is in src/utils and config is in src/config
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_accurate_cost(context, query, answer, prompt_template):
    config = load_config()
    cost_cfg = config.get('cost_tracking', {})
    
    encoding_model = cost_cfg.get('encoding_model', 'o200k_base')
    input_rate = cost_cfg.get('input_cost_per_million', 0.15)
    output_rate = cost_cfg.get('output_cost_per_million', 0.60)
    
    encoding = tiktoken.get_encoding(encoding_model)
    
    full_input = prompt_template.format(context=context, question=query)
    input_tokens = len(encoding.encode(full_input))
    
    output_tokens = len(encoding.encode(answer))

    cost = (input_tokens * (input_rate / 1_000_000)) + (output_tokens * (output_rate / 1_000_000))
    
    return cost

def total_cost():
    config = load_config()
    # Use the path from config, or a default relative to project root
    file_path = config.get('eval_results_path', './artifacts/test/rag_evaluation_results.json')
    
    # If the path is relative, make it relative to the project root
    if not os.path.isabs(file_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(project_root, file_path)

    total_cost_val = 0
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        print(f"Successfully loaded JSON data from {file_path}")
        total_cost_val = sum(d.get('cost', 0) for d in data)
        return total_cost_val

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return 0
    except Exception as e:
        print(f"Error loading cost data: {e}")
        return 0
