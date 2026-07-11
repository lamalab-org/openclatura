# -*- coding: utf-8 -*-

"""
STOUT Python Package.
This repository contains STOUT-V2,
The second version of STOUT: SMILES TO IUPAC Translator.
Which is built using the same concept as a Neural Machine Translation(NMT) using Transformers.


Typical usage example:

from stout import translate_forward, translate_reverse

# SMILES to IUPAC translation
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
iupac_name = translate_forward(SMILES)
print(iupac_name)

# IUPAC to SMILES translation
iupac_name = "1,3,7-trimethylpurine-2,6-dione"
SMILES = translate_reverse(iupac_name)
print(SMILES)

For comments, bug reports or feature ideas,
please raise a issue on the Github repository.

"""

__version__ = "2.0.5"

__all__ = [
    "translate_forward",
    "translate_forward_batch",
    "translate_reverse",
    "get_device_info",
]


from .stout import (
    get_device_info,
    translate_forward,
    translate_forward_batch,
    translate_reverse,
)
