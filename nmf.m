% This is a MATLAB program for running non-negative matrix factorization 
% on user/recipe ratings data scraped from AllRecipes.com. 
% This program took 8 hours when I ran it on a machine with 128 GB RAM.
% By Jeremy Cohen

% Download the MATLAB NNMF toolbox from here:
% https://sites.google.com/site/nmftool/home/source-code
addpath('nmfv1_4'); % where 'nmfv1_4' is replaced by the directory where 
                    % the NNMF toolbox is located

% LOAD USER/RECIPE FAVORITES DATA

% The favorite recipes of 116K users are stored in data/favorites.csv.
% Each line in data/favorites.csv takes the form "[USER_ID],[RECIPE_ID]"
% These IDs begin at 0
% The ID of a recipe is its line number in the file data/recipes.csv  
load data/favorites.csv;

% CONVERT TO SPARSE MATRIX

% since user IDs and recipe IDs will refer to matrix coordinates, they must
% be one-indexed rather than zero-indexed
favorites = favorites + 1; 

% if a user has favorited a recipe, the corresponding entry in the 
% user/recipe matrix should be 1; otherwise, the entry should be 0
favorites(:, 3) = 1;

% make into a sparse matrix
ratings = spconvert(favorites);

% RUN MATRIX FACTORIZATION
% (or don't, and just look at the completed factorization in results/)

options = {};
options.distance = 'kl';
numFactors = 40;
[recipe_weights,user_weights,numIterations,timeElapsed,finalResidual] ...
    = nmfrule(ratings,numFactors,options);

% RECORD RESULTS

csvwrite('recipe_weights.csv', recipe_weights);
csvwrite('user_weights.csv', user_weights);
timeElapsed
numIterations
