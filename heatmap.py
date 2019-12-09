from matplotlib import gridspec

import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def parse_data(filename):
	# read input file
	data = np.array(pd.read_csv(filename + '.csv', header = None)) # read input file

	# determine number of meta-learning iterations and feature names
	iters = [int(i) for i in data[0][1:]]
	features = [row[0] for row in data[1:]]

	# eliminate column of feature names from data
	data = np.array([row[1:] for row in data[1:]], dtype=float)

	return iters, features, data

def get_layers(num_values):
	return [
		('Programs (P)', 2),
		('Conditions (C)', 2),
		('Base conditions (B)', 2),
		('Offsets (O)', 8),
		('Numbers (N)', 4),
		('Values (V)', num_values - 18)
	]

def plot_heatmap(data, xticklabels, yticklabels, layers, title):
	# set up figure
	fig = plt.figure(figsize = (7, 9))
	gs = gridspec.GridSpec(len(layers), 1, height_ratios = [layer[1] for layer in layers]) 

	# iterate through every layer in grammar
	start_index = 0
	for i in range(len(layers)):
		layer_name, num_features = layers[i]

		# obtain dataframe for layer
		end_index = start_index + num_features
		dataframe = pd.DataFrame(data[start_index : end_index])

		# determine label booleans
		first_layer = i == 0 # title = Nim, xaxis = top, xaxis = 'Programs'
		middle_layer = i == 3 # title = label, xaxis = None
		last_layer = i == len(layers) - 1 # title = label, xaxis = 'Iters'

		# initialize subplot
		axes = plt.subplot(gs[i])

		# plot heatmap for layer
		plot_layer_heatmap(dataframe,
			xticklabels if last_layer else [], # xticklabels
			yticklabels[start_index : end_index]) # yticklabels

		set_plot_labels(axes,
			title, 
			'Iterations of\nMeta-Learning', # xlabel
			'DSL Feature', # ylabel
			layer_name,
			first_layer, middle_layer, last_layer)

		start_index = end_index

def plot_layer_heatmap(dataframe, xticklabels, yticklabels):
	g = sns.heatmap(dataframe,
		annot = False, square = True,
		xticklabels = xticklabels, yticklabels = yticklabels,
		cbar = False)
	g.set_yticklabels(g.get_yticklabels(), rotation = 0)

def set_plot_labels(axes, title, xlabel, ylabel, layer_name, first_layer, middle_layer, last_layer):
	if first_layer: # title is plot title, xlabel is layer name
		plt.title(title, pad = 30)
		axes.xaxis.set_label_position('top')
		plt.xlabel(layer_name)
	elif last_layer: # title is layer name, xlabel is plot xlabel
		plt.title(layer_name, pad = 4, fontsize = 10)
		plt.xlabel(xlabel)
	else: # title is layer name, ylabel is plot ylabel
		plt.title(layer_name, pad = 4, fontsize = 10)
		plt.ylabel(ylabel if middle_layer else '', labelpad = 70)

if __name__  == "__main__":
    minigame = str(sys.argv[1])

    # get plotting parameters
    iters, features, data = parse_data(minigame)
    layers = get_layers(len(features))

    plot_heatmap(data, iters, features, layers, minigame)
    plt.savefig(minigame + "_heatmap.png")
