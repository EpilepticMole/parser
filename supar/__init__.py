# -*- coding: utf-8 -*-

from .models import (CRFConstituencyParser, VAECRFConstituencyParser)
from .parser import Parser
from .structs import (ConstituencyCRF)

__all__ = [
    'Parser',
    'CRFConstituencyParser',
    'VAECRFConstituencyParser',
    'ConstituencyCRF'
]

__version__ = '1.1.4'

PARSER = {
    parser.NAME: parser for parser in [
        CRFConstituencyParser,
        VAECRFConstituencyParser
    ]
}

SRC = {
    'github': 'https://github.com/EpilepticMole/parser/releases/download/embedding'
}
NAME = {
    'con-crf-en': 'ptb.crf.con.lstm.char',
    'con-crf-zh': 'ctb7.crf.con.lstm.char',
    'con-crf-roberta-en': 'ptb.crf.con.roberta',
    'con-crf-electra-zh': 'ctb7.crf.con.electra',
    'con-crf-xlmr': 'spmrl.crf.con.xlmr'
}
MODEL = {src: {n: f"{link}/{m}.zip" for n, m in NAME.items()} for src, link in SRC.items()}
CONFIG = {src: {n: f"{link}/{m}.ini" for n, m in NAME.items()} for src, link in SRC.items()}


def compatible():
    import sys
    supar = sys.modules[__name__]
    if supar.__version__ < '1.2':
        sys.modules['supar.utils.config'] = supar.config
        sys.modules['supar.utils.transform'].Tree = supar.models.const.crf.transform.Tree
        sys.modules['supar.parsers'] = supar.models
        sys.modules['supar.parsers.con'] = supar.models.const


compatible()
