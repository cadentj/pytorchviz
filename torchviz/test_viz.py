# %%
import nnsight
import torch
from torchviz import make_dot
from torch._subclasses.fake_tensor import FakeTensorMode 
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Initialize ShapeEnv and FakeTensorMode
shape_env = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

# Load a Hugging Face model and tokenizer
model_name = "EleutherAI/pythia-14m"

config = AutoConfig.from_pretrained(model_name)

# with torch.device('meta'):
with fake_mode:
    model = AutoModelForCausalLM.from_config(config=config)
    model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Create a sample input
input_ids = torch.zeros((13, 31), dtype=torch.long)
inputs = {"input_ids": input_ids}


# Run the model with fake tensors
with fake_mode:
    fake_inputs = {k: fake_mode.from_tensor(v) for k, v in inputs.items()}
    out = model(**fake_inputs)


make_dot(out.logits, params=dict(model.named_parameters()))

# %%

import torch
from torchviz import make_dot
import torch.nn as nn   


class ToyLayer(nn.Module):
    def __init__(self):
        super(ToyLayer, self).__init__()
        self.mlp = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.mlp(x)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.layer_one = ToyLayer()
        self.layer_two = ToyLayer()

    def forward(self, x):
        x += self.layer_one(x)
        x += self.layer_two(x)
        return x

model = ToyModel()

x = torch.zeros(10)
y = model(x)
make_dot(y, params=dict(model.named_parameters()))