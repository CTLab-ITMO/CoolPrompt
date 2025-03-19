import time
import requests



def vllm_infer(prompt_token_ids, model_name, stop_token_ids,
            server_url = "http://localhost:8000/v1/chat/completions",
            temperature=0.7, n=1, top_p=1, stop=None, max_tokens=50,
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt_token_ids}]
    payload = {
        "prompt_token_ids": messages,
        "model": model_name,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "stop_token_ids": stop_token_ids,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
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