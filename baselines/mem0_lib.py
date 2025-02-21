import os


def build_config(model_name, api_key, use_chroma=False):
    provider = "openai"

    config = {"version": "v1.1"}

    if use_chroma:
        path = "memory_db"
        vector_entry = {
            "provider": "chroma",
            "config": {"collection_name": "test", "path": path},
        }
        config["vector_store"] = vector_entry
        if os.path.exists(path):
            print(f"Warning: {path} already exists for Chroma DB, it will be loaded")
    # otherwise uses the default, qdrant in RAM

    llm_entry = {"provider": provider, "config": {"model": model_name, "api_key": api_key}}

    config["llm"] = llm_entry
    return config
