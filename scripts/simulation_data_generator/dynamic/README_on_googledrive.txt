--------------------------

2020.12.19

Generating data for input into tensors:
simulation_data_dynamic.py

This script works with dS00N.csv(N=1,2,3...) files. 
dS00N.csv provides social score for this script to generate mazes.

Each maze will be generated in a new file called 'data/data_dynamic/dS00N/dS00N_S_n'

dS00N means number N set of social score
dS00N_S_n means S agent in the center of the maze (same to A,B,C,D agent); n means the number of the maze(should be 10000 at total)

So far this script generates one maze at a time, I will add a loop to make it generate 10000 mazes at a time later.