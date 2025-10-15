DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_HF_ENDPOINT_MODEL_NAME = "mistralai/Mistral-Nemo-Base-2407"
DEFAULT_HF_ENDPOINT_PROVIDER = "novita"
DEFAULT_OLLAMA_MODEL_NAME = "owl/t-lite:instruct"
DEFAULT_MODEL_PARAMETERS = {
    "max_new_tokens": 4000,
    "temperature": 0.01,
    "do_sample": False,
    "return_full_text": False,
}
DEFAULT_HF_MODEL_PARAMETERS = {"max_new_tokens": 4000, "temperature": 0.01}
DEFAULT_OUTLINES_MODEL_PARAMETERS = {"max_tokens": 4000, "temperature": 0}
