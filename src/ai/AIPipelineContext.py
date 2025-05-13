import torch

class AIPipelineContext:
    max_new_tokens: int
    dtype: torch.dtype = torch.float16
    device: str
    model: str
    kwargs: dict
    verbose: bool
    think: bool

    def __init__(self, model, device: str = "auto", max_new_tokens: int = 32768, dtype: torch.dtype = torch.float16, verbose: bool = False, think: bool = False, **kwargs):
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.model = model
        self.device = device
        self.verbose = verbose
        self.kwargs = kwargs
        self.think = think
