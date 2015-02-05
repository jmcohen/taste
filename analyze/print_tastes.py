# Prints the top recipes in every factor

import numpy as np

N = 20 # how many top recipes to print from each factor

f = open('../data/recipes.csv')
recipeLines = f.readlines()
recipeNames = [line.split(',')[1].strip() for line in recipeLines]
f.close()

recipeWeights = np.loadtxt('../results/recipe_weights.csv', dtype='float', delimiter=',')
numFactors = recipeWeights.shape[1]
for iFactor in range(numFactors):
	print "Taste %d" % (iFactor + 1)
	topRecipeIds = recipeWeights[:, iFactor].argsort()[::-1][0:N]
	print [recipeNames[recipeId] for recipeId in topRecipeIds], "\n"
