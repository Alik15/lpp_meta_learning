from dsl import *
from env_settings import *
from pipeline import *

import re
import sys

import matplotlib.pyplot as plt

def learn_probs(base_class_name, program_generation_step_size, num_programs,
            iters = 10, epsilon = 1, plot = False):
    # initialize blank probability dictionary
    object_types = get_object_types(base_class_name)
    grammar_regex = get_grammar_regex(object_types)
    initial_probs = get_initial_probs(object_types)
    probs_dicts = [{grammar_regex[level][i]: initial_probs[level][i] for i in range(len(grammar_regex[level]))} for level in grammar_regex]
    probs = {k: v for d in probs_dicts for k, v in d.items()}
    print("Initial probs:", probs)

    if plot:
        test_num_programs(base_class_name, program_generation_step_size, num_programs, probs)
        plt.title("Meta-Learning Improvement for " + base_class_name + " (initial)")
        plt.savefig(base_class_name + "_initial.png")

    for i in range(iters):
        print("RUNNING META-LEARNING IERATION", i + 1, "OF", iters)

        # train a new policy with given probs
        policy = train(base_class_name, range(11), program_generation_step_size, num_programs, 5, 25, probs)
        results = test(policy, base_class_name, record_videos = False)
        print("Test results:", results)
        
        # update probabilities
        plps = str(policy.plps)
        for probs_dict in probs_dicts:
            probs_dict.update(adjust(probs_dict, update_probs(plps, probs_dict), epsilon = epsilon))
        probs = {k: v for d in probs_dicts for k, v in d.items()}
        print("Updated probs:", probs)

        if plot:
            test_num_programs(base_class_name, program_generation_step_size, num_programs, probs, iteration = str(i + 1))

    print("Final probs:", probs)
    return probs

def test_num_programs(base_class_name, program_generation_step_size, max_num_programs, probs, iteration = False, alpha = 1):
    plt.clf() # clear figure

    # initialize data lists
    x = list(range(program_generation_step_size, max_num_programs + program_generation_step_size, program_generation_step_size))
    y = []

    # test for every num_programs
    for num_programs in x:
        print("Testing with", num_programs, "programs")
        blockPrint()
        policy = train(base_class_name, range(11), program_generation_step_size, num_programs + program_generation_step_size, 5, 25, probs)
        results = test(policy, base_class_name, record_videos = False)
        fraction = results.count(True) * 1./len(results)
        y += [fraction]
        enablePrint()
    
    # plot the data
    plt.plot(x, y, color = 'b', alpha = alpha)
    plt.xlabel("# features enumerated")
    plt.ylabel("Test success fraction")
    if iteration:
        plt.title("Meta-Learning Improvement for " + base_class_name + " (iteration " + iteration + ")")
        plt.savefig(base_class_name + "_" + iteration + ".png")

def adjust(old, new, epsilon = 0.7):
    adjusted = {}
    for k in old:
        adjusted[k] = (1 - epsilon) * old[k] + epsilon * new[k]
    return adjusted

def update_probs(plps, probs_dict):
    counts_dict = {}
    for feature in probs_dict:
        counts_dict[feature] = 1.0 * len(re.findall(feature, plps))
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