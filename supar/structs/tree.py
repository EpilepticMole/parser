# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from supar.structs.dist import StructuredDistribution
from supar.structs.fn import mst
from supar.structs.semiring import LogSemiring, Semiring
from supar.utils.fn import diagonal_stripe, expanded_stripe, stripe
from torch.distributions.utils import lazy_property


class ConstituencyCRF(StructuredDistribution):
    r"""
    Constituency TreeCRF :cite:`zhang-etal-2020-fast,stern-etal-2017-minimal`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all constituents.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking.

    Examples:
        >>> from supar import ConstituencyCRF
        >>> batch_size, seq_len, n_labels = 2, 5, 4
        >>> lens = torch.tensor([3, 4])
        >>> charts = torch.tensor([[[-1,  0, -1,  0, -1],
                                    [-1, -1,  0,  0, -1],
                                    [-1, -1, -1,  0, -1],
                                    [-1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1]],
                                   [[-1,  0,  0, -1,  0],
                                    [-1, -1,  0, -1, -1],
                                    [-1, -1, -1,  0,  0],
                                    [-1, -1, -1, -1,  0],
                                    [-1, -1, -1, -1, -1]]])
        >>> s1 = ConstituencyCRF(torch.randn(batch_size, seq_len, seq_len, n_labels), lens, True)
        >>> s2 = ConstituencyCRF(torch.randn(batch_size, seq_len, seq_len, n_labels), lens, True)
        >>> s1.max
        tensor([3.7036, 7.2569], grad_fn=<IndexBackward0>)
        >>> s1.argmax
        [[[0, 1, 2], [0, 3, 0], [1, 2, 1], [1, 3, 0], [2, 3, 3]],
         [[0, 1, 1], [0, 4, 2], [1, 2, 3], [1, 4, 1], [2, 3, 2], [2, 4, 3], [3, 4, 3]]]
        >>> s1.log_partition
        tensor([ 8.5394, 12.9940], grad_fn=<IndexBackward0>)
        >>> s1.log_prob(charts)
        tensor([ -8.5209, -14.1160], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([6.8868, 9.3996], grad_fn=<IndexBackward0>)
        >>> s1.kl(s2)
        tensor([4.0039, 4.1037], grad_fn=<IndexBackward0>)
    """

    def __init__(
        self,
        scores: torch.Tensor,
        lens: Optional[torch.LongTensor] = None,
        label: bool = False
    ) -> ConstituencyCRF:
        super().__init__(scores)

        batch_size, seq_len, *_ = scores.shape
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.unsqueeze(1) & scores.new_ones(scores.shape[:3]).bool().triu_(1)
        self.label = label

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label})"

    def __add__(self, other):
        return ConstituencyCRF(torch.stack((self.scores, other.scores), -1), self.lens, self.label)

    @lazy_property
    def argmax(self):
        return [torch.nonzero(i).tolist() for i in self.backward(self.max.sum())]

    def topk(self, k: int) -> List[List[Tuple]]:
        return list(zip(*[[torch.nonzero(j).tolist() for j in self.backward(i)] for i in self.kmax(k).sum(0)]))

    def score(self, value: torch.LongTensor) -> torch.Tensor:
        mask = self.mask & value.ge(0)
        if self.label:
            scores = self.scores[mask].gather(-1, value[mask].unsqueeze(-1)).squeeze(-1)
            scores = torch.full_like(mask, LogSemiring.one, dtype=scores.dtype).masked_scatter_(mask, scores)
        else:
            scores = LogSemiring.one_mask(self.scores, ~mask)
        return LogSemiring.prod(LogSemiring.prod(scores, -1), -1)

    @torch.enable_grad()
    def forward(self, semiring: Semiring) -> torch.Tensor:
        batch_size, seq_len = self.scores.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (l->r)
        scores = semiring.convert(self.scores.movedim((1, 2), (0, 1)))
        scores = semiring.sum(scores, 3) if self.label else scores
        s = semiring.zeros_like(scores)
        s.diagonal(1).copy_(scores.diagonal(1))

        for w in range(2, seq_len):
            n = seq_len - w
            # [n, batch_size, ...]
            s_s = semiring.dot(stripe(s, n, w-1, (0, 1)), stripe(s, n, w-1, (1, w), False), 1)
            s.diagonal(w).copy_(semiring.mul(s_s, scores.diagonal(w).movedim(-1, 0)).movedim(0, -1))
        return semiring.unconvert(s)[0][self.lens, range(batch_size)]