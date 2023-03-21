import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models.nn import build_entailment, build_box_registry
from utils import freeze
from utils.misc import *
from utils import *

class UnknownArgument(Exception):
    def __init__():super()

class HalProgramExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, config):
        super().__init__()
        self.config = config

        entailment = build_entailment(config)
        concept_registry = build_box_registry(config)

        # [Word Vocab]

        sents = ["concept red have same category as concept blue","same category concepts are considered synonyms"]

        self.concept_vocab = WordVocab()
        self.concept_vocab.update(sents)
        self.concept_vocab.freeze()


        self.learner = PipelineLearner(0, entailment)
        self.translator = config.translator

    def forward(self, q):
        return q(self.learner)

    def parse(self,string, translator = None):
        if translator == None: translator = self.translator
        def chain(p):
            head, paras = head_and_paras(p)
            if paras == None:
                q = head
            elif '' in paras:
                q = translator[head]()
            else:
                args = [chain(o) for o in paras]
                if head in translator: q = translator[head](*args)
                else: raise UnknownArgument
            return q
        program = chain(string)
        return program


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