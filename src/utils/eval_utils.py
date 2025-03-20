import time
import requests


class Infer:
    """Inference helper class for vllm server
    """
    def __init__(self, model_name, server_url, model_generate_args):
        self.model_name = model_name
        self.server_url = server_url
        self.model_generate_args = model_generate_args

    def __call__(self, prompt, label_id):
        """Label is needed to ensure label <-> prompt match"""
        return vllm_infer(prompt, self.model_name, server_url=self.server_url, **self.model_generate_args), label_id


def vllm_infer(prompt, model_name, stop_token_ids,
            server_url = "http://localhost:8000/v1/chat/completions",
            temperature=0.0, n=1, top_p=1, stop=None, max_tokens=50,
                  presence_penalty=0, frequency_penalty=0, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
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
    retries = 0
    while True:
        try:
            r = requests.post(server_url,
                headers = {
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]