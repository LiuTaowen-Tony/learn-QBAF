# Learning Argumentation Frameworks Classifier using Genetic Algorithm

This is a repo for Argumentation framework classifier research project. In this repo, we extended the baseline algorithm with 4 extensions

The baseline model repo is [QBAF-Learning](https://github.com/jspieler/QBAF-Learning)

- Improving the input transformation module using fuzzy sets
- Allowing input arguments to attack or support output arguments directly
- Exploring the impact of incorporating multiple layers of hidden arguments
- Enabling the joint attack or support semantics

In the baseline model, the input transformation converts the data into binary values. We employ fuzzy sets in the input transformation to convert values in the range [0, 1]. One idea for improving the classification performance is to increase the number of layers in the QBAF classifier. More layers of hidden arguments might capture argumentation involving more steps. We investigate the impact of increasing the depth on the model's interpretability and classification performance. In the baseline model, the input arguments only affect the output arguments through hidden arguments, increasing the model's complexity and reducing the explainability. To solve this problem, we extend the baseline model by allowing input arguments to attack or support output arguments directly. Joint attack and support can capture the idea that a single argument is insufficient for defeating another argument, but multiple arguments can. Joint attack and support generalise the graphical structure of the model into a hyper-graph, making it more expressive. Experiments show that fuzzy input, direct attack and support, and joint attack and support improve the model's interpretability.

We also did initial feasibility experiments to use iterative pruning to learn the QBAF structure.

## Installation

- Clone the repo
- Install the prerequisites

## Usage

Install all the prerequisites

```
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

Run all experiments, this might takes long (a few days). This will run experiment with all the single extensions then the combination of the extension. However, due to `multi-layer of hidden arguments` doesn't receive good effect, we didn't include it into combination of experiments.

```
./run_all
```

if you would like to run one experiment, you can specify your configuration in 
command line

run `fuzzy input`, `joint attack and support`, and `direct attack and support`
experiment

```
python run_exp.py dataset joint_connection_number size_of_QBAF has_direct_connections use_relative_sparsity is_fuzzy
```

run `multi-layer of hidden arguments`

```
python run_exp_multigbag.py dataset size_of_QBAF use_relative_sparsity is_fuzzy
```

options

- dataset: "iris" "mushroom" "adult"
- joint_connection_number: 1 2
- size_of_QBAF: 1 2 3 4 5 where 1 is the largest, and 5 is the smallest
- has_direct_connections: "direct" "no"
- use_relative_sparsity: "sp" "no" representing whether having sparse connections or not
- is_fuzzy: "fuzzy" "no"

## model names

JASGBAG : QBAF classifier with joint attack and support

DAGBAG : QBAF classifier with direct attack and support

JASDAGBAG : QBAF classifier with both direct attack and support and joint attack and support

MULTIGBAG : QBAF classifier with 2 hidden layers

<!-- ## code structure
`run_exp.py` is the entrance of `fuzzy input`, `Joint attack and support`, and `direct attack and support`. This file parse the command line input  -->


# Iterative Pruning to learn QBAF structure


## Algorithm

```
procedureITERATIVEPRUNING(D,M,η,p,s)
  while Sparsity of M < s do
    Train M on D using learning rate η with l2 regularisation α
    Compute the absolute value of all weights |wi|
    Sort the weights by |wi|
    Set the lowest p percent of weights to zero
    Set the pruned weight to be not trainable
  end while
end procedure
```

## Usage

Our algorithm supports 3 different structure of QBAF, baseline QBAF, QBAF with direct connections, and QBAF with 2 hidden layers. The structure of QBAFs are described in our paper. 

To run an algorithm

```
python iterative_pruning_algo.py
```




