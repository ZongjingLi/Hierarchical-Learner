import torch
from torch import nn

class BoxRegistry(nn.Module):
    _init_methods = {"uniform": torch.nn.init.uniform_}

    def __init__(self, config):
        super().__init__()
        self.dim = config.concept_dim

        entries = config.entries

        init_config = config
        self.boxes = self._init_embedding_(entries, init_config)
        
        self.offset_clamp = config.offset
        self.offset_clamp = config.center
    
    def _init_embedding_(self, entries, config):
        init_method = config.method
        center = torch.Tensor(entries, self.dim)
        offset = torch.Tensor(entries, self.dim)
        self._init_methods[init_method](center, *config.center)
        self._init_methods[init_method](offset, *config.offset)
        return nn.Embedding(entries, self.dim * 2, _weight=torch.cat([center, offset], dim=1))
    
    def forward(self, x):return self.boxes(x)

    def clamp_dimensions(self):
        with torch.no_grad():
            self.boxes.weight[:, self.dim:].clamp_(*self.offset_clamp)
            self.boxes.weight[:, :self.dim].clamp_(*self.center_clamp)

    def __getitem__(self, key, item): self.boxes.weight[key] = item

    def __getitem__(self, key):return self.boxes.weight[key]

    @property
    def device(self):return self.boxes.weight.device

    @property
    def prototypes(self):return self.boxes.weight.detach()

    def __len__(self):return len(self.boxes.weight)

    @property
    def size(self):return self.dim ** 2

registry_map = {"box": BoxRegistry}

def build_box_registry(config):return registry_map[config.concept_type](config)