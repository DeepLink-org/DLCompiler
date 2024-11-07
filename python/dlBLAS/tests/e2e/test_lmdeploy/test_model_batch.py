import os
import torch
import subprocess
import yaml
import json

def load_yaml_config(config_file):
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    return {}

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_file, tested_models):
    with open(checkpoint_file, "w") as f:
        json.dump(tested_models, f, indent=4)

common_prefix = ""

def get_model_paths(config):
    model_paths = []
    global common_prefix
    common_prefix = config.get("common_prefix", "")
    for model in config["models"]:
        print("model", model)
        path = model.get("path") or os.path.join(common_prefix, model["name"])
        model_paths.append({"name": model["name"], "path": path})
    return model_paths

def batch_test_models(config, checkpoint_file="checkpoint.json"):
    model_paths = get_model_paths(config)
    tested_result = load_checkpoint(checkpoint_file)

    for model_info in model_paths:
        model_name = model_info["name"]
        model_path = model_info["path"]
        print("model_name: ", model_name)
        print("model_path: ", model_path)
        if model_name in tested_result and (tested_result[model_name]["status"] == "completed" or True):
            print(f"---------Skipping already tested model: {model_name}")
            continue
        subprocess.run(["python", "test_model_once.py", f"{model_name}", f"{model_path}", f"{checkpoint_file}", f"{common_prefix}"])



if __name__ == "__main__":
    print(torch.cuda.memory_allocated())
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)

    config = load_yaml_config("test_model_config.yaml")
    batch_test_models(config)

    print("Batch testing completed!")