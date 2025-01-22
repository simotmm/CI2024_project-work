# Computational Intelligence - Final Project Work: Symbolic Regression with Genetic Programming Algorithm
Repository for the final project work of Computational Intelligence (01URROV) 2024-2025 course at Politecnico di Torino.


## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Genetic Algorithm](#genetic-algorithm)
- [Features](#features)


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

### Genetic Operators
#### Mutation or Crossover
According to the mutation probability in the `Settings` is chosen Mutation or Crossover as genetic operator.
- Mutation:
##### Parent Selection and Crossover
- Parent Selection: two individuals are randomly chosen from the population.
- *Note*: the first idea for the selection was a double tournament selection with fitness hole (choose n=2 individuals, and select the one with the best fitness with a hight probability) but the trend of the results suggested to remove that computational cost and choose randomly, to increase diversity and add a litte speed up to the process.
- Crossover: features of the two individuals are combined to create a new one.
#### Mutation
According to the mutation probability in the `Settings` is chosen Mutation or Crossover as genetic operator
- Mutation:



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



````````````

````````````






