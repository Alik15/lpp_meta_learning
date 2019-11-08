from dsl import *
from env_settings import *
from pipeline import *

import math
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

def learn_probs(base_class_name, program_generation_step_size, num_programs,
            iters = 10, epsilon = 1, analyze_improvement = False):
    # initialize blank probability dictionary
    object_types = get_object_types(base_class_name)
    grammar_regex = get_grammar_regex(object_types)
    initial_probs = get_initial_probs(object_types)
    probs_dicts = [{grammar_regex[level][i]: initial_probs[level][i] for i in range(len(grammar_regex[level]))} for level in grammar_regex]
    probs = {k: v for d in probs_dicts for k, v in d.items()}
    print("Initial probs:", probs)

    if analyze_improvement:
        improvement_results = [test_num_programs(base_class_name, program_generation_step_size, num_programs, probs)]

    for i in range(iters):
        print("RUNNING META-LEARNING IERATION", i + 1, "OF", iters)

        # train a new policy with given probs
        policy = train(base_class_name, range(11), program_generation_step_size, num_programs, 5, 25, probs)
        results = test(policy, base_class_name, record_videos = False)
        print("Test results:", results)
        
        # update probabilities
        for probs_dict in probs_dicts:
            new_probs_dict = update_probs(policy.plps, policy.probs, probs_dict)
            probs_dict.update(adjust(probs_dict, new_probs_dict, epsilon = epsilon))
        probs = {k: v for d in probs_dicts for k, v in d.items()}
        print("Updated probs:", probs)

        if analyze_improvement:
            improvement_results += [test_num_programs(base_class_name, program_generation_step_size, num_programs, probs)]

    if analyze_improvement:
        print("Improvement results:", improvement_results)
        x = list(range(iters + 1))
        y = [res[0][-1] for res in improvement_results]

        plt.plot(x, y)
        plt.title('Meta-Learning Improvement for ' + base_class_name)
        plt.xlabel('Iterations of meta-learning')
        plt.ylabel('# features enumerated')

        # make sure axis ticks are integers
        xticks = range(min(x), math.ceil(max(x)) + 1)
        plt.xticks(xticks)
        yticks = range(min(y), math.ceil(max(y)) + 1, 3)
        plt.yticks(yticks)

        plt.savefig(base_class_name + "_improvement.png")

    print("Final probs:", probs)
    return probs

def test_num_programs(base_class_name, program_generation_step_size, max_num_programs, probs, full_curve = False):
    plt.clf() # clear figure

    # initialize data lists
    x = []
    y = []

    # test for every num_programs
    for num_programs in range(program_generation_step_size, max_num_programs + program_generation_step_size, program_generation_step_size):
        print("Testing with", num_programs, "programs")

        # train and test model with given num_programs
        blockPrint()
        policy = train(base_class_name, range(11), program_generation_step_size, num_programs + program_generation_step_size, 5, 25, probs)
        results = test(policy, base_class_name, record_videos = False)
        fraction = results.count(True) * 1./len(results)
        enablePrint()

        # update data lists
        x += [num_programs]
        y += [fraction]
        if not full_curve and fraction == 1:
            return (x, y)

    return (x, y)

def adjust(old, new, epsilon = 0.7):
    adjusted = {}
    for k in old:
        adjusted[k] = (1 - epsilon) * old[k] + epsilon * new[k]
    return adjusted

def update_probs(all_plps, plp_probs, probs_dict):
    counts_dict = {}
    for feature in probs_dict:
        plp_counts = [len(re.findall(feature, str(all_plps[i]))) for i in range(len(all_plps))] # get count of feature in each PLP
        counts_dict[feature] = np.dot(plp_probs, plp_counts)
    return counts_to_probs(counts_dict)

def counts_to_probs(counts):
    return {key: counts[key] / sum(counts.values()) for key in counts}

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

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
