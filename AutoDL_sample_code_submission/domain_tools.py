#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-01-10

import numpy as np

from tools import log


def infer_domain(metadata):
    """Infer the domain from the shape of the 4-D tensor.

    Args:
      metadata: an AutoDLMetadata object.
    """
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    channel_to_index_map = metadata.get_channel_to_index_map()
    domain = None
    if sequence_size == 1:
        if row_count == 1 or col_count == 1:
            domain = "tabular"
        else:
            domain = "image"
    else:
        if row_count == 1 and col_count == 1:
            if len(channel_to_index_map) > 0:
                domain = "text"
            else:
                domain = "speech"
        else:
            domain = "video"
    return domain


def get_domain_metadata(metadata, domain, is_training=True):
    """Recover the metadata in corresponding competitions, esp. AutoNLP
    and AutoSpeech.

    Args:
      metadata: an AutoDLMetadata object.
      domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
    """
    if domain == 'text':
        # Fetch metadata info from `metadata`
        class_num = metadata.get_output_size()
        num_examples = metadata.size()
        language = 'ZH' if is_chinese(metadata) else 'EN'
        time_budget = 1200  # WARNING: Hard-coded

        # Create domain metadata
        domain_metadata = {}
        domain_metadata['class_num'] = class_num
        if is_training:
            domain_metadata['train_num'] = num_examples
            domain_metadata['test_num'] = -1
        else:
            domain_metadata['train_num'] = -1
            domain_metadata['test_num'] = num_examples
        domain_metadata['language'] = language
        domain_metadata['time_budget'] = time_budget

        return domain_metadata
    elif domain == 'speech':
        # Fetch metadata info from `metadata`
        class_num = metadata.get_output_size()
        num_examples = metadata.size()

        # WARNING: hard-coded properties
        file_format = 'wav'
        sample_rate = 16000

        # Create domain metadata
        domain_metadata = {}
        domain_metadata['class_num'] = class_num
        if is_training:
            domain_metadata['train_num'] = num_examples
            domain_metadata['test_num'] = -1
        else:
            domain_metadata['train_num'] = -1
            domain_metadata['test_num'] = num_examples
        domain_metadata['file_format'] = file_format
        domain_metadata['sample_rate'] = sample_rate

        return domain_metadata
    else:
        return metadata


def is_chinese(metadata):
    """Judge if the dataset is a Chinese NLP dataset. The current criterion is if
    each word in the vocabulary contains one single character, because when the
    documents are in Chinese, we tokenize each character when formatting the
    dataset.

    Args:
      metadata: an AutoDLMetadata object.
    """
    domain = infer_domain(metadata)
    if domain != 'text':
        return False
    for i, token in enumerate(metadata.get_channel_to_index_map()):
        # if len(token) != 1 and token != '\u2003':
        #     print("not chinese {} len {}".format(token, len(token)))
        #     return False
        if '\u4e00' <= token <= '\u9fff':
            print(token)
            return True
        if i >= 1000:
            break
    # return True
    return False
