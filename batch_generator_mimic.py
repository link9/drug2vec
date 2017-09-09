"""
Skimgram_model batch generator
    for MIMIC-III admission-visit-codes level list data
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from itertools import chain


def batch_generator_skipgram(adm_visit_drug_cd, model, context_window_size, batch_size):
    """
    :param model: data model
    :param adm_visit_drug_cd: nested list for addmission seq, visit seq, dx/rx code seq
    :param int_to_drug_cd_dict: dict for converting code seq to label
    :param context_window_size: skipgram window
    :param batch_size: batch_size
    :return: pairs generator
    """
    if model == "simple":
        single_gen = generate_pairs_simple_model(adm_visit_drug_cd, context_window_size)
        return get_batch(single_gen, batch_size)

def generate_pairs_simple_model(adm_seqs, context_window_size):
    """ Form training pairs along prescription record sequences per a admission """

    generators = []
    for visit_seqs in adm_seqs:
        for code_seqs in visit_seqs:
            generators.append(generate_pairs_seqs(code_seqs, context_window_size))
    return chain(*generators)

def generate_pairs_seqs(code_seqs, context_window_size):
    for index, center in enumerate(code_seqs):
        context = random.randint(1, context_window_size)  # 1-N random int

        # get a random target before the center word
        for target in code_seqs[max(0, index - context): index]:
            yield center, target

        # get a random target after the center word
        for target in code_seqs[index + 1: min(len(code_seqs), (index + 1) + context)]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch
