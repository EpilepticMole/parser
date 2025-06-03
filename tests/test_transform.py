# -*- coding: utf-8 -*-

import itertools

import nltk

from supar.models.const.crf.transform import Tree


class TestTree:

    def test_tree(self):
        tree = nltk.Tree.fromstring("""
                                    (TOP
                                      (S
                                        (NP (DT This) (NN time))
                                        (, ,)
                                        (NP (DT the) (NNS firms))
                                        (VP (VBD were) (ADJP (JJ ready)))
                                        (. .)))
                                    """)
        assert tree == Tree.build(tree, Tree.factorize(Tree.binarize(tree)[0]))
