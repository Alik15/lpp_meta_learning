from dsl import *
from env_settings import *
from pipeline import *

import math
import random
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

def learn_probs(base_class_name, program_generation_step_size, num_programs,
            iters = 21, epsilon = 1, analyze_improvement = False):
    # initialize blank probability dictionary
    object_types = get_object_types(base_class_name)
    grammar_regex = get_grammar_regex(object_types)
    grammar_labels = get_grammar_labels(object_types)
    initial_probs = get_initial_probs(object_types)

    # initialize feature probabilities
    probs_dicts = [{(grammar_regex[level][i], grammar_labels[level][i]): initial_probs[level][i] for i in range(len(grammar_regex[level]))}
        for level in grammar_regex]
    probs = {k[1]: v for d in probs_dicts for k, v in d.items()}
    print("Initial probs:", probs)

    # train initial policy
    min_num_programs = num_programs
    policy = train(base_class_name, range(11), program_generation_step_size, min_num_programs, 5, 25, probs)
    if analyze_improvement:
        improvement_results = [test_num_programs(base_class_name, program_generation_step_size, num_programs, probs)]
        min_num_programs = improvement_results[-1][0][-1] + program_generation_step_size

    curr_layer = 0
    for i in range(iters):
        print("RUNNING META-LEARNING ITERATION", i + 1, "OF", iters)

        # update probabilities
        old_probs_dict = {k: v for k, v in probs_dicts[curr_layer].items()}
        new_probs_dict = probs_dicts[curr_layer]
        new_probs_dict = update_probs(policy.plps, policy.probs, old_probs_dict)
        new_probs_dict = adjust(old_probs_dict, new_probs_dict, epsilon = epsilon)
        probs_dicts[curr_layer].update(new_probs_dict)
        probs = {k[1]: v for d in probs_dicts for k, v in d.items()}
        print("Updated probs:", probs)

        # train a new policy with given probs
        old_policy = policy
        policy = train(base_class_name, range(11), program_generation_step_size, min_num_programs, 5, 25, probs)
        results = test(policy, base_class_name, record_videos = False)
        print("Test results:", results)

        # revert changes if learned policy failed
        if False in results:
            improvement_results += [([None], [0])]
            print("Reverting", curr_layer)
            probs_dicts[curr_layer].update(old_probs_dict)
            probs = {k[1]: v for d in probs_dicts for k, v in d.items()}
            policy = old_policy

        # update minimum number of programs enumerated if learned policy succeeded
        elif analyze_improvement:
            improvement_results += [test_num_programs(base_class_name, program_generation_step_size, min_num_programs, probs)]
            min_num_programs = improvement_results[-1][0][-1] + program_generation_step_size

        curr_layer = (curr_layer + 1) % len(probs_dicts)

    if analyze_improvement:
        print("Improvement results:", improvement_results)
        plot_improvement(base_class_name, improvement_results)
    
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

def plot_improvement(base_class_name, improvement_results):
    x = list(range(len(improvement_results)))
    y = [res[0][-1] for res in improvement_results]

    plt.plot(x, y)
    plt.scatter(x, y)
    plt.title('Meta-Learning Improvement for ' + base_class_name)
    plt.xlabel('Iterations of meta-learning')
    plt.ylabel('# features enumerated')

    plt.savefig(base_class_name + "_improvement.png")

def adjust(old, new, epsilon = 0.7):
    adjusted = {}
    for k in old:
        adjusted[k] = (1 - epsilon) * old[k] + epsilon * new[k]
    return adjusted

def update_probs(all_plps, plp_probs, probs_dict):
    counts_dict = {}
    for (regex, label) in probs_dict:
        plp_counts = [len(re.findall(regex, str(all_plps[i]))) for i in range(len(all_plps))] # get count of feature in each PLP
        counts_dict[(regex, label)] = np.dot(plp_probs, plp_counts)
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
