# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='supar vae',
    version='1.0',
    author='Frederic Jehan',
    author_email='frederic.jehan@proton.me',
    description='Parser with Latent Variable using Variational Autoencoder',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/EpilepticMole/parser',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'numpy>1.21.6',
        'torch>=1.13.1',
        'transformers>=4.30.0',
        'nltk',
        'stanza',
        'omegaconf',
        'dill',
        'pathos',
        'opt_einsum'
    ],
    extras_require={
        'elmo': ['allennlp'],
        'bpe': ['subword-nmt']
    },
    entry_points={
        'console_scripts': [
            'con-crf=supar.cmds.const.crf:main',
            'con-vae-crf=supar.cmds.const.vae_crf:main'
        ]
    },
    python_requires='>=3.7',
    zip_safe=False
)
