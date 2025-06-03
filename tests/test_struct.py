# -*- coding: utf-8 -*-

import itertools

import torch
from supar.structs import (ConstituencyCRF)
from supar.structs.semiring import LogSemiring, MaxSemiring, Semiring
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class BruteForceStructuredDistribution(Distribution):

    def __init__(self, scores, **kwargs):
        self.kwargs = kwargs

        self.scores = scores.requires_grad_() if isinstance(scores, torch.Tensor) else [s.requires_grad_() for s in scores]

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @lazy_property
    def log_partition(self):
        return torch.stack([LogSemiring.sum(i, -1) for i in self.enumerate(LogSemiring)])

    @lazy_property
    def max(self):
        return torch.stack([MaxSemiring.sum(i, -1) for i in self.enumerate(MaxSemiring)])

    def kmax(self, k):
        return torch.stack([i.topk(k)[0] for i in self.enumerate(MaxSemiring)])

    @lazy_property
    def entropy(self):
        ps = [seq - self.log_partition[i] for i, seq in enumerate(self.enumerate(LogSemiring))]
        return -torch.stack([(i.exp() * i).sum() for i in ps])

    @lazy_property
    def count(self):
        structs = self.enumerate(Semiring)
        return torch.tensor([len(i) for i in structs]).to(structs[0].device).long()

    def cross_entropy(self, other):
        ps = [seq - self.log_partition[i] for i, seq in enumerate(self.enumerate(LogSemiring))]
        qs = [seq - other.log_partition[i] for i, seq in enumerate(other.enumerate(LogSemiring))]
        return -torch.stack([(i.exp() * j).sum() for i, j in zip(ps, qs)])

    def kl(self, other):
        return self.cross_entropy(other) - self.entropy

    def enumerate(self, semiring):
        raise NotImplementedError



class BruteForceConstituencyCRF(BruteForceStructuredDistribution):

    def __init__(self, scores, lens=None, label=False):
        super().__init__(scores)

        batch_size, seq_len = scores.shape[:2]
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.unsqueeze(1) & scores.new_ones(scores.shape[:3]).bool().triu_(1)
        self.label = label

    def enumerate(self, semiring):
        scores = self.scores if self.label else self.scores.unsqueeze(-1)

        def enumerate(s, i, j):
            if i + 1 == j:
                yield from s[i, j].unbind(-1)
            for k in range(i + 1, j):
                for t1 in enumerate(s, i, k):
                    for t2 in enumerate(s, k, j):
                        for t in s[i, j].unbind(-1):
                            yield semiring.times(t, t1, t2)
        return [torch.stack([i for i in enumerate(s, 0, length)]) for s, length in zip(scores, self.lens)]



def test_struct():
    torch.manual_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size, seq_len, n_tags, k = 2, 6, 3, 3
    lens = torch.randint(3, seq_len-1, (batch_size,)).to(device)

    def enumerate():
        s1 = torch.randn(batch_size, seq_len, seq_len, n_tags).to(device)
        s2 = torch.randn(batch_size, seq_len, seq_len, n_tags).to(device)
        yield (ConstituencyCRF(s1, lens, True), ConstituencyCRF(s2, lens, True),
               BruteForceConstituencyCRF(s1, lens, True), BruteForceConstituencyCRF(s2, lens, True))
        s1 = torch.randn(batch_size, seq_len, n_tags).to(device)
        s2 = torch.randn(batch_size, seq_len, n_tags).to(device)

    for _ in range(5):
        for struct1, struct2, brute1, brute2 in enumerate():
            assert struct1.max.allclose(brute1.max)
            assert struct1.kmax(k).allclose(brute1.kmax(k))
            assert struct1.log_partition.allclose(brute1.log_partition)
            assert struct1.entropy.allclose(brute1.entropy)
            assert struct1.cross_entropy(struct2).allclose(brute1.cross_entropy(brute2))
            assert struct1.kl(struct2).allclose(brute1.kl(brute2))
