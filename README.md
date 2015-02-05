# Non-Negative Matrix Factorization for AllRecipes Taste Data

This repository contains data and code for an application of non-negative matrix factorization to a data set of recipe favorites scraped from AllRecipes.com.

The project is described in my blog post [Decomposing the Human Taste Palate](http://www.jeremymcohen.net/posts/taste/).

The contents of this repository are:

`data/favorites.csv`: a data set of 116K AllRecipes users and their collections of favorite recipes.
Each line in this file takes the form "USER_INDEX,RECIPE_INDEX".

`data/recipes.csv`: the names and slugs of 9,546 recipes from AllRecipes.com.
Use this file to look up the name of a recipe given its index.

`nmf.m`: a MATLAB script that performs non-negative matrix factorization on `data/favorites.csv`.

`results/`: the results of running `nmf.m`.  

`analyze/print_tastes`, a Python script that prints out the top recipes in each factor ("taste"). 

`nmfv1_4/`: a copy of the [NNMF MATLAB library](https://sites.google.com/site/nmftool/) by Yifeng Li. 

I carried out this project in Spring 2014 with Rob Sami, Aaron Schild, and Spencer Tank.