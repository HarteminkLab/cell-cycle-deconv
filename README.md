# Readme documentation for cell cycle deconvolution


## Introduction: 

The primary purpose of this document is to better convey and explain the cell cycle deconvolution matlab code, and eventually the python code.

There are multiple components to the project.

First, we will define the inputs and outputs and the algorithm.

## Outline:
1. Inputs
2. The model/algorithm
   1. Objective
   2. Smoothing/wavelet regularization
   3. Gamma regularization term
3. Outputs
4. Installation 
5. Usage
6. Code/Model structure
## 1. Inputs:
- **Genes:** List of all genes we plan to deconvolve, this will match up row-wise the input data
- **Data:** experimental time course, such as the gene expression of a population of cells going through the cell cycle.
- **Model:** we have different cell cycle parameters. This is fed in through a run of another model called CLOCCS (*to expand), in which we estimate parameters that define the parameters of the cell cycle from a parallel flow cytometry or budding index data set that corresponds with the experimental data set.
- **Configuration:** A file that defines the file locations for the data, model, and any other parameterization that isn't strictly defined as cell cycle parameters that the model file contains.

## 2. The model/algorithm

(Brief overview of branching process, and branches to define H and f, refer to Orlando paper for details).

The model is set up as a convex optimization problem in which we define a convolution kernel from the cell cycle parameters to deconvolve the experimental input data:

g = H * f

The model aims to minimize:

()

Additionally, we also add smoothing constraints using wavelet regularization:

()

Find Gamma
## 3. Output

The final output will be a matrix of deconvolved f values for each gene. As well as the optimal gamma value.

## 4. Installation

Currently the project is written in Matlab. It requires the cvx library to run. Run setup.m to install cvx for the first time. 

## 5. Usage

`mainSingleGene.m` - Runs the deconvolution for a single gene, and finds the optimal gamma for that gene.
`main.m` - Runs the deconvolution for all genes

(TODO: Expand the algorithm explanation more, add references, write out the important code files and API structure of the model)