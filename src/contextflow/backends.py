import requests
import sseclient
import json


class LlamaCppBackend:
    def __init__(self, url, max_predict=1024):
        self.url = url
        self.max_predict = max_predict

    def get_request_object(self, request_tokens, stream, temp, top_p, min_p, top_k, dry_multiplier, id_slot):
        return {"prompt": request_tokens,
                "stream": True,
                "n_predict": self.max_predict,
                "temperature": temp,
                "top_p": top_p,
                "min_p": min_p,
                "top_k": top_k,
                "stop": [self.stop_token, self.tokenizer.eos_token],
                "id_slot": id_slot}

    def total_slots(self):
        response = requests.get(self.url + "/props")
        response.raise_for_status()
        resp_obj = response.json()
        return resp_obj["total_slots"]

    async def async_completion(self, request_tokens, temp=0.7, top_p=0.95, min_p=0.05, top_k=20, id_slot=-1, callback=None):
        request = self.get_request_object(request_tokens, True, temp, top_p, min_p, top_k, id_slot)
        response = requests.post(self.url + "/completion", json=request, headers={'Accept': 'text/event-stream'})
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
