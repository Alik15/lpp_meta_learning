from matplotlib import gridspec

import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def parse_data(filename):
	# read input file
	data = np.array(pd.read_csv(filename + '.csv')) # read input file

	# determine number of meta-learning iterations and feature names
	iters = list(range(len(data[0]) - 1))
	features = [row[0] for row in data]

	# eliminate column of feature names from data
	data = np.array([row[1:] for row in data], dtype=float)

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

		# plot heatmap for layer
		plot_layer_heatmap(dataframe,
			gs[i], # subplot_params
			title if i == 0 else '', # title
			layer_name,
			num_features,
			'DSL Feature' if i == 3 else '', # ylabel
			xticklabels if i == len(layers) - 1 else [], # xticklabels
			yticklabels[start_index : end_index]) # yticklabels

		start_index = end_index

def plot_layer_heatmap(dataframe, subplot_params, title, layer_name, num_features, ylabel, xticklabels, yticklabels):
	# set up subplot
	ax = plt.subplot(subplot_params)
	ax.xaxis.set_label_position('top')
	
	# plot layer heatmap
	g = sns.heatmap(dataframe, annot = False, square = True, xticklabels = xticklabels, yticklabels = yticklabels, cbar = False)
	g.set_yticklabels(g.get_yticklabels(), rotation = 0)
	
	# set heatmap plot labels
	plt.title(title, pad = 30)
	plt.xlabel(layer_name)
	plt.ylabel(ylabel, labelpad = 60)

if __name__  == "__main__":
    minigame = str(sys.argv[1])

    # get plotting parameters
    iters, features, data = parse_data(minigame)
    layers = get_layers(len(features))

    plot_heatmap(data, iters, features, layers, minigame)
    plt.savefig(minigame + ".png")
