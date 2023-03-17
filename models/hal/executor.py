import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models.nn import build_entailment
from utils import freeze


class HalProgramExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, config):
        super().__init__()
        #etwork = self.NETWORK_REGISTRY[config.name](config)
        entailment = build_entailment(config)
        self.learner = PipelineLearner(0, entailment)

    def forward(self, q):
        return q(self.learner)


class MetaLearner(nn.Module):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = MetaLearner.get_name(cls.__name__)
        HalProgramExecutor.NETWORK_REGISTRY[name] = cls
        cls.name = name

    @staticmethod
    def get_name(name):
        return name[:-len('Learner')]

    def forward(self, p):
        return {}

    def compute_logits(self, p, **kwargs):
        return p.evaluate_logits(self, **kwargs)


class PipelineLearner(nn.Module):
    def __init__(self, network, entailment):
        super().__init__()
        self.network = network
        self.entailment = entailment

    def forward(self, p):
        shots = []
        for q in p.train_program:
            end = q(self)["end"]
            index = end.squeeze(0).max(0).indices
            shots.append(q.object_collections[index])
        shots = torch.stack(shots)
        if not p.is_fewshot:
            shots = shots[0:0]
        fewshot = p.to_fewshot(shots)
        return fewshot(self)

    def compute_logits(self, p, **kwargs):
        return self.network.compute_logits(p, **kwargs)