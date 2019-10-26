from dsl import *
from env_settings import *
from pipeline import *

import re
import sys

def learn_probs(base_class_name, program_generation_step_size, num_programs, iters = 10):
    # initialize blank probability dictionary
    object_types = get_object_types(base_class_name)
    grammar_regex = get_grammar_regex(object_types)
    probs_dicts = [{regex: 1./len(level_regex) for regex in level_regex} for level_regex in grammar_regex.values()]

    probs = {}
    for i in range(iters):
        updated_probs = {k: v for d in probs_dicts for k, v in d.items()}
        print("Probs:", updated_probs)
        if probs == updated_probs: # if converged
            break
        probs = updated_probs
        
        policy = train(base_class_name, range(11), program_generation_step_size, num_programs, 5, 25, probs)
        
        # update probabilities
        plps = str(policy.plps)
        for probs_dict in probs_dicts:
            probs_dict.update(update_probs(plps, probs_dict))

    return probs

def update_probs(plps, probs_dict):
    for feature in probs_dict:
        probs_dict[feature] = 1.0 * len(re.findall(feature, plps))
    return counts_to_probs(probs_dict)

def counts_to_probs(counts):
    return {key: counts[key] / sum(counts.values()) for key in counts}

if __name__  == "__main__":
    base_class_name = str(sys.argv[1])

    # parameters to train() based on game
    program_generation_step_size = 10
    num_programs = 1000
    if base_class_name == "TwoPileNim":
        program_generation_step_size = 1
        num_programs = 250

    params = learn_probs(base_class_name, program_generation_step_size, num_programs)
    print("Learned params:", params)