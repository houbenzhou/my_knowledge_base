# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from ..datasets.pascal_voc import pascal_voc

# Set up voc_<year>_<split>

for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}'.format(split)
    __sets[name] = (lambda split=split: pascal_voc(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_diff'.format(split)
    __sets[name] = (lambda split=split: pascal_voc(split, use_diff=True))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
