import requests
import sseclient
import json


class OpenaAIBackend:
    def __init__(self, url, max_predict=1024, model="local", api_key="test"):
        self.url = url.rstrip("/")
        self.max_predict = max_predict
        self.model = model
        self.api_key = api_key

    async def async_completion(self, request_tokens, temp=0.7, top_p=0.95, min_p=0.05, top_k=20, id_slot=-1, callback=None):
        """
        Стримит из /v1/completions. Возвращает (text, finish_reason_like).
        finish_reason_like ~ 'stop' | 'length' | 'content_filter' | ...
        """
        payload = self.get_request_object(
            request_tokens=request_tokens,
            temp=temp,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            id_slot=id_slot,
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        # Если есть ключ — добавим авторизацию (полезно для api.openai.com или прокси)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # OpenAI-совместимый путь
        url = f"{self.url}/v1/completions"

        resp = requests.post(url, json=payload, headers=headers, stream=True)
        resp.raise_for_status()

        client = sseclient.SSEClient(resp)
        text_resp = ""
        finish_reason = None

        for event in client.events():
            if not event.data:
                continue
            if event.data.strip() == "[DONE]":
                break

            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                # пропустим служебные строки
                continue

            choices = data.get("choices") or []
            if not choices:
                continue

            ch0 = choices[0]
            delta_text = ch0.get("text", "")
            if delta_text:
                text_resp += delta_text
                if callback:
                    await callback(delta_text)

            # появится на финальном чанке
            if ch0.get("finish_reason") is not None:
                finish_reason = ch0["finish_reason"]

        return text_resp, (finish_reason or "stop")

    def get_request_object(self, request_tokens, temp, top_p, min_p, top_k, id_slot):
        return {
            "model": self.model,              # для «чистого» OpenAI обязателен; локальным сервером может игнорироваться
            "prompt": request_tokens,
            "max_tokens": self.max_predict,    # аналог n_predict
            "temperature": float(temp),
            "top_p": float(top_p),
            "stream": True,
            "stop": [self.stop_token, self.tokenizer.eos_token],
            # ---- параметры которые не являются частью OpenAI API
            "top_k": top_k,
            "min_p": float(min_p),
            # ---- llama.cpp специфичные параметры
            "id_slot": id_slot,
            "repeat_penalty": 1.0,
            # ---- vllm специфичные параметры
            "repetition_penalty": 1.0
        }

    def total_slots(self):
        """
        Пытаемся получить количество слотов у llama.cpp.
        Если эндпоинта нет или структура другая — возвращаем -1.
        """
        try:
            resp = requests.get(f"{self.url}/props", timeout=3)
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("total_slots", -1))
        except Exception:
            return -1
