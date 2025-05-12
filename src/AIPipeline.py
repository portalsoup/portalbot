from collections import deque

import torch
import json

from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
from dataclasses import dataclass, asdict

@dataclass
class Message:
    content: str
    role: str = "user"

@dataclass
class Response:
    thinking: str
    response: str

class AIContext:
    max_new_tokens: int
    dtype: torch.dtype = torch.float16
    device: str
    model: str
    # task: str
    kwargs: dict = {}

    def __init__(self, model, device, task: str, max_new_tokens: int, dtype: torch.dtype, **kwargs):
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.model = model
        self.device = device
        self.task = task
        self.kwargs = kwargs

class AIPipeline2:
    pipe: Pipeline
    message_history: deque[Message]
    ctx: AIContext
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, ctx: AIContext):
        self.message_history = deque(maxlen=10)
        self.ctx = ctx
        self.model = AutoModelForCausalLM.from_pretrained(ctx.model, device_map=ctx.device, torch_dtype=ctx.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(ctx.model)

    def add_template(self, template: Message):
        self.message_history.append(template)

    def query(self, query: Message, think: bool = True) -> Response:
        messages = [asdict(msg) for msg in self.message_history] + [asdict(query)]
        json_input = json.dumps(messages)

        text = self.tokenizer.apply_chat_template(json_input, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.ctx.max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True)
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip('\n')

        return Response(thinking_content, content)
