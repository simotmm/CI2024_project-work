# Computational Intelligence - Final Project Work: Symbolic Regression with Genetic Programming Algorithm
Repository for the final project work of Computational Intelligence (01URROV) 2024-2025 course at Politecnico di Torino.

Simone Tumminelli (s333017)


## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Genetic Algorithm](#genetic-algorithm)
- [Features](#features)
- [Steps of the Algorithm](#steps-of-the-algorithm)
  - [Generation of the population](#generation-of-the-population)
  - [Fitness Evaluation](#fitness-evaluation)
  - [Generation of the Offspring and Elite](#generation-of-the-offspring-and-elite)
  - [Genetic Operators](#genetic-operators)
  - [Iteration](#iteration)
  - [End of the process](#end-of-the-process)
- [Data Structures](#data-structures)
  - [Node](#node)
  - [Problem](#problem)
  - [Settings](#settings)



## Introduction
The goal of this project is to find the best mathematical function that approximates the values of a gicen dataset.

The approach used is Symbolic Regression with Genetic Programming Algorithm, the function is encoded using sintax tree.



## Requirements
- Python 3.12.8 +
- matplotlib
- numpy
- tqdm



## Genetic Algorithm
In the genetic algorithm every syntax tree is treated as an individual in a population. The population evolves by generation in generation and the best individuals are preserved. At the end of the process the result is the best individual.


## Features
- **Supported operations**
  - Arithmetic operations: `+`, `-`, `*`, `/`.
  - Trigonometric operations: `sin`, `cos`, `tan`
  - Others: `sqrt`, `log`, `abs`
  - *Note*: `sqrt`, `log` and `div` are implemented in a safe way to avoid runtime errors. The operand of `sqrt` and `log` will be always in absolute value, and a very small constant (`1e-40`) is added to thsecond operand of `div`, as well as to the operand `sqrt` and `log`. This operations are preserved in the final result tree function.
  All the 
  - *Note*: all the operators are expressed in `numpy` (e.g. `a+b` will be `np.add(a,b)`).
- **Visualization**
  - During the evolution process: progress bar.
  - At the end of the evolution process: is shown a graph that illustrates the trend of evolution over the generations.


## Steps of the Algorithm
### Generation of the population
Given a maximum depth a list of individuals is generated with ramped method: a small part is a set of full trees (depth is maximum) and the other part is a set of not complete trees with depth between `0` and `max_depth` (`0` depth: a node with no children).

### Fitness evaluation
All the individuals in the population are evaluated and sorted by the fitness metric. The best individual is eventually updated.
Fitness Metric: based on mean square error. (`fitness`=`-mse`, higher values mean better results).

### Generation of the Offspring and Elite
The `offspring` is the result of **Genetic Operators**, it will be used, as next population, combined with the `elite`, the individuals with the best fitness values, they will be preserved in the next generation.

#### Genetic Operators
##### Mutation or Crossover
According to the mutation probability in the `Settings` is chosen Mutation or Crossover as genetic operator.
###### Parent Selection and Crossover
- Parent Selection: two individuals are randomly chosen from the population.
- *Note*: the first idea for the selection was a double tournament selection with fitness hole (choose n=2 individuals, and select the one with the best fitness with a hight probability) but the trend of the results suggested to remove that computational cost and choose randomly, to increase diversity and add a litte speed up to the process.
- Crossover: features of the two individuals are combined to create a new one.
###### Mutation
An individual is randomly selected from the population and their features are changed. Types of mutation used are: subtree mutation, hoist_mutation, collapse mutation, expansion mutation and point mutation.

### Iteration
The process from Fitness Evaluation to Mutation is repeated `max_generation` times.

### End of the process
At the end of the process the best individual and the history of best fitness values are returned.


## Data Structures
### Node
The `Node` object represents an individual of the popolation. It is, in fact, a tree with an associated value and eventually one or two child nodes, depending on the type of the value.
#### Node types
- Constant: an `integer` or a `floating point` value. It has no child nodes.
- Variable: a `string` in the form of `x[i]`, with `i` depending on the first dimension of the input dataset. It has no child nodes.
- Operator:
  - Single Operator: a custom `function` among `sin`, `cos`, `tan`, `log` and `sqrt`. It has one child node.
  - Double Operator: a custom `function` among `add`, `subtract`, `multiply` and `divide`. I has two child nodes.
### Problem
The `Problem` object represents the problem to solve, it contains the data from the corresponding `.npz` file and an object `Settings`.
### Settings
The `Settings` object is a set of parameters such as population dimension, number of generation, maximum tree depth, probabiliy of mutation and percentage of elitism.


## Repository Structure
- [s333017.py](./s333017.py): module that contains the best functions fuond by the algorithm for each problem proposed.
- [/data](./data/): it contains the dataset for the problems proposed, the form of each dataset is: input values `x` and output values `y` (`x` can be multi-dimensional).
- [log.md](./log.md): report that contains the log of activities during the course.
- [/src](./src/): root folder of genetic algorithm program. it contain a [runnable example](./src/example.ipynb), the main that runs the algorithm for all the problems and all the modules.
  - `main.py`: runs the algoritm for all the problems. It accept a command line parameter to run the algorithm for a specific problm. (example: to run the algoritm for the `problem_4.npz`: type in terminal "python main.py 4").
  - modules: 
    - `classes.py`: definition of classes for `Node`, `Problem` and `Settings` and global constants.
    - `operators.py`: definition of operators dictionaries and functions.
    - `init.py`: initialization of randomness and settings space for all the problems.
    - `fitness.py`: fitness evaluation functions.
    - `genetic_algoritm.py`: main algorithm and offspring functions.
    - `genetic_operators.py`: crossover and mutation functions.
    - `population.py`: population generation and parent selection functions.
    - `tree_functions.py`: functions generate full trees and nodes.
    - `utils.py`: utility functions to get data from dataset,to plot fitness graph, and others.
  - [/scripts](./src/scripts/): simple .bat scripts to run all the probems in parallel or run a single problem.


## Table of Result
| Problem | Mean Square Error      |
|:-------:|:----------------------:|
| 1       | 7.125940794232773e-34  |
| 2       | 13987717013674.275     |
| 3       | 5.306486117932578e-29  |
| 4       | 0.022938281157499518   |
| 5       | 7.804262963600968e-19  |
| 6       | 1.4783925196927175e-05 |
| 7       | 32.653988150939874     |
| 8       | 29932.624601487092     |


