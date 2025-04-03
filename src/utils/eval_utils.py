import requests


HEADERS = {"Content-Type": "application/json"}


class Infer:
    """Inference helper class for vllm server"""

    def __init__(self, model_name,
                 server_url="http://localhost:8000/v1/completions",
                 model_generate_args={}):
        self.model_name = model_name
        self.server_url = server_url
        self.model_generate_args = model_generate_args

    def __call__(self, prompt, label_id=None, **call_params):
        """Label is needed to ensure label <-> prompt match"""
        generate_args = {**self.model_generate_args, **call_params}
        result = vllm_infer(
                prompt,
                self.model_name,
                server_url=self.server_url,
                **generate_args
        )
        if len(result) == 1:
            result = result[0]
        return result, label_id


def vllm_infer(
    prompt,
    model_name,
    stop_token_ids,
    server_url="http://localhost:8000/v1/completions",
    temperature=0.0,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0,
    timeout=100,
):
    """Про параметры читать тут: https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams"""
    with requests.Session() as session:
        payload = {
            "prompt": prompt,
            "model": model_name,
            "temperature": temperature,
            "n": n,
            "top_p": top_p,
            "stop": stop,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        response = session.post(
            server_url, json=payload, headers=HEADERS, timeout=timeout
        )
        completions = response.json().get("choices", [])
        return [completion["text"] for completion in completions]
