import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

from src.ai.AIPipelineContext import AIPipelineContext


@dataclass
class Response:
    thinking: str
    response: str


def parse_response(raw: str) -> Response:
    pattern = r"<think>(.*?)</think>(.*)"

    match = re.search(pattern, raw, re.DOTALL)
    if match:
        between_think_tags = match.group(1)
        after_think_tag = match.group(2)
        return Response(between_think_tags, after_think_tag)
    else:
        return Response("", raw)


class AIPipeline:
    ctx: AIPipelineContext
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, ctx: AIPipelineContext):
        self.ctx = ctx
        self.model = AutoModelForCausalLM.from_pretrained(ctx.model, device_map=ctx.device, torch_dtype=ctx.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(ctx.model)

    def query(self, query: list[dict[str, str]]) -> Response:
        response = self._tokenize_query(query)
        return parse_response(response[0])

    def _tokenize_query(self, query: list[dict[str, str]]):
        text = self.tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return self._multinomial_decode(model_inputs)

    def _multinomial_decode(self, model_inputs: torch.Tensor):
        torch.manual_seed(0)
        outputs = self.model.generate(**model_inputs, do_sample=True, max_length=self.ctx.max_new_tokens)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


