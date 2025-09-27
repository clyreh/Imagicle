import torch, torch.nn as nn
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=16, dropout=0.0):
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.lora_A = nn.Parameter(torch.zeros((r, base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, r)))
        self.scaling = alpha / r; self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5); nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) + \
               self.drop(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
def _target(name: str):
    n=name.lower(); keys=["attn","to_q","to_k","to_v","q_proj","k_proj","v_proj","out_proj","proj","fc1","fc2","mlp","ff","out"]
    return any(k in n for k in keys)
def inject_lora(model: nn.Module, r=16, alpha=16, dropout=0.0, verbose=True):
    c=0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and _target(f"{name}.{cname}"):
                setattr(mod, cname, LoRALinear(child, r=r, alpha=alpha, dropout=dropout)); c+=1
                if verbose: print(f"[LoRA] wrapped {name}.{cname}")
    if verbose: print(f"[LoRA] total wrapped: {c}")
    return model
def lora_params(model): return [p for n,p in model.named_parameters() if "lora_" in n and p.requires_grad]
