USER_DIR = "."


def build_config(model_name, api_key, collection_name="test", vector_store="qdrant"):
    provider = "openai"

    config = {"version": "v1.1"}

    path = f"{USER_DIR}/{vector_store}_db"

    vector_entry = {
        "provider": vector_store,
        "config": {
            "collection_name": collection_name,
            "path": path,
        },
    }
    if vector_store == "qdrant":
        vector_entry["on_disk"] = True
    config["vector_store"] = vector_entry
    # otherwise uses the default, qdrant in RAM

    llm_entry = {"provider": provider, "config": {"model": model_name, "api_key": api_key}}

    config["llm"] = llm_entry
    return config
