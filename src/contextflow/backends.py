import requests
import sseclient
import json


class LlamaCppBackend:
    def __init__(self, url, max_predict=1024):
        self.url = url
        self.max_predict = max_predict

    def get_request_object(self, request_tokens, stream, temp, top_p, min_p, dry_multiplier):
        return {"prompt": request_tokens,
                "stream": stream,
                "n_predict": self.max_predict,
                "temperature": temp,
                "top_p": top_p,
                "min_p": min_p,
                "top_k": 0,
                "stop": [self.stop_token, self.tokenizer.eos_token],
                "dry_multiplier": dry_multiplier,
                "dry_sequence_breakers": ["\n", ":", "\"", "*", "|", ""],
                "cache_prompt": True}

    def completion(self, request_tokens, temp=0.7, top_p=0.9, min_p=0.05, dry_multiplier=0.0):
        request = self.get_request_object(request_tokens, False, temp, top_p, min_p, dry_multiplier)
        response = requests.post(self.url, json=request)
        response.raise_for_status()
        resp_obj = response.json()
        return resp_obj["content"], resp_obj["stop_type"]

    async def async_completion(self, request_tokens, temp=0.7, top_p=0.9, min_p=0.05, dry_multiplier=0.0, callback=None):
        request = self.get_request_object(request_tokens, True, temp, top_p, min_p, dry_multiplier)
        response = requests.post(self.url, json=request, stream=True, headers={'Accept': 'text/event-stream'})
        response.raise_for_status()
        stream = sseclient.SSEClient(response).events()
        text_resp = ""
        for event in stream:
            parsed_event = json.loads(event.data)
            if parsed_event["stop"]:
                break
            content = parsed_event["content"]
            text_resp += content
            if callback:
                await callback(content)
        return text_resp, parsed_event["stop_type"]
