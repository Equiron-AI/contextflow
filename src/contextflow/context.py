import os
import json
import logging
from transformers import AutoTokenizer, AutoConfig


logger = logging.getLogger(__name__)


class ContextFlow:
    def __init__(self, llm_backend, base_model, max_context=4096, prompt="", prompt_file="", cut_context_multiplier=1, cut_from=1):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, add_bos_token=False)
        self.max_predict = llm_backend.max_predict
        self.max_context = max_context
        self.cut_context_multiplier = cut_context_multiplier
        self.cut_from = cut_from

        if prompt_file:
            with open(prompt_file) as f:
                prompt = f.read()

        config = AutoConfig.from_pretrained(base_model)

        if config.model_type.startswith("cohere"):
            self.generation_promp_template = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            self.user_req_template = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_req}<|END_OF_TURN_TOKEN|>"
            self.system_injection_template = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_injection}<|END_OF_TURN_TOKEN|>"
            self.tokens = [self.tokenizer.apply_chat_template([{"role": "system", "content": prompt}])]
            self.stop_token = self.tokenizer.eos_token
        elif config.model_type.startswith("gemma"):
            self.generation_promp_template = "<start_of_turn>model\n"
            self.user_req_template = "<start_of_turn>user\n{user_req}<end_of_turn>\n"
            self.system_injection_template = "<start_of_turn>system\n{system_injection}<end_of_turn>\n"
            self.tokens = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}])]
            self.stop_token = "<end_of_turn>"
        elif config.model_type.startswith("mistral") or config.model_type.startswith("qwen"):
            self.generation_promp_template = "<|im_start|>assistant\n"
            self.user_req_template = "<|im_start|>user\n{user_req}<|im_end|>\n"
            self.system_injection_template = "<|im_start|>system\n{system_injection}<|im_end|>\n"
            self.tokens = [self.tokenizer(f"<|im_start|>system\n{prompt}<|im_end|>\n")["input_ids"]]
            self.stop_token = "<|im_end|>"
        else:
            raise RuntimeError("Unknown model: " + config.model_type)

        self.llm_backend = llm_backend
        self.llm_backend.stop_token = self.stop_token
        self.llm_backend.base_model = base_model
        self.llm_backend.tokenizer = self.tokenizer

        self.generation_prompt_tokens = self.tokenize(self.generation_promp_template)
        logger.info("System prompt size: " + str(len(self.tokens[0])))

    def tokenize(self, text):
        tokens = self.tokenizer(text)["input_ids"]
        if self.tokenizer.bos_token_id:
            if self.tokenizer.bos_token_id in tokens:
                tokens.remove(self.tokenizer.bos_token_id)
        return tokens

    def sanitize(self, text):
        return text.replace("#", "") \
                   .replace("<|", "") \
                   .replace("|>", "") \
                   .replace("<start_of_turn>", "") \
                   .replace("<end_of_turn>", "") \
                   .replace("[INST]", "") \
                   .replace("[/INST]", "")

    def add_user_request(self, user_request, system_injection="", unsanitized_raw_postfix=""):
        text = self.user_req_template.replace("{user_req}", self.sanitize(user_request.strip()) + unsanitized_raw_postfix)
        if system_injection:
            text += self.system_injection_template.replace("{system_injection}", system_injection)
        tokens = self.tokenize(text)
        self.tokens.append(tokens)
        self._cut_context()  # Освобождаем место под ответ модели

    def add_system_injection(self, system_injection):
        text = self.system_injection_template.replace("{system_injection}", system_injection)
        self.tokens.append(self.tokenize(text))
        self._cut_context()  # Освобождаем место под ответ модели

    async def async_completion(self, temp=0.7, top_p=0.9, min_p=0.05, repeat_last_n=256, repeat_penalty=1.1, callback=None):
        request_tokens = sum(self.tokens, [])
        request_tokens += self.generation_prompt_tokens
        text_resp = await self.llm_backend.async_completion(request_tokens, temp, top_p, min_p, repeat_last_n, repeat_penalty, callback)
        response_tokens = self.tokenize(text_resp.strip() + self.stop_token)
        response_tokens = self.generation_prompt_tokens + response_tokens
        self.tokens.append(response_tokens)
        return text_resp

    def load_context(self, file_name):
        if os.path.isfile(file_name):
            with open(file_name) as f:
                self.tokens = json.load(f)

    def save_context(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.tokens, f)

    def dump_context(self, file_name):
        with open(file_name, "w") as f:
            flat_tokens = sum(self.tokens, [])
            f.write(self.tokenizer.decode(flat_tokens))

    def _cut_context(self):
        busy_tokens = len(sum(self.tokens, []))
        free_tokens = self.max_context - busy_tokens
        if free_tokens < self.max_predict:
            while free_tokens < self.max_predict * self.cut_context_multiplier:  # обрезаем с большим запасом, чтобы кеш контекста работал лучше
                free_tokens += len(self.tokens[self.cut_from])
                del self.tokens[self.cut_from]

    def clear_context(self):
        del self.tokens[1:]
