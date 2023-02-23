# Chernoff distance measure of Dirichlet distributions
A tiny Python package that provides an implementation of the Chernoff distance of two Dirichlet distributions.

The Chernoff distance is completely implemented in Pytorch and is differentiable. It can be used as a loss function for certain applications.

For more background on the topic (also on the question why one should use Chernoff instead of e.g. Kullback-Leibler divergence), see the paper by Rauber, Braun and Kerns. If you find the implemented formula useful, please consider citing the paper.

```
@article{rauber2008probabilistic,
  title={Probabilistic distance measures of the Dirichlet and Beta distributions},
  author={Rauber, Thomas W and Braun, Tim and Berns, Karsten},
  journal={Pattern Recognition},
  volume={41},
  number={2},
  pages={637--645},
  year={2008},
  publisher={Elsevier}
}
```
## Installation

The package can be installed directly via pip.
```
$ git clone https://github.com/cklugmann/probabilistic_distance
$ python -m pip install -e probabilistic_distance
```

## Usage

The package implements the [Chernoff distance](https://www.jstor.org/stable/2236576) and the [Bhattacharyya distance](https://www.jstor.org/stable/25047882) as a special case of it (for lambda=1). We can import the latter and use it according to.

```python
from probabilistic_distance.dirichlet import BhattacharyyaDistance
```

We instantiate the class and create a callable object. The constructor allows to pass a flag `from_logspace` which is set to true by default. The idea behind this is that we typically work with model predictions. We treat these values as logarithms of the Dirichlet distribution parameters we are actually interested in, so as not to run into problems with the constraint of positivity of all parameters. Note, however, that you can set the argument to false to operate directly on the untransformed parameters.

````python
chernoff_dist = BhattacharyyaDistance(from_logspace=False)
````

We construct two `torch.Tensor` objects and pass them to our distance measure. Note that the first argument conceptually stands for the predictions and the second for the ground truth or target parameters. Also, the interface supports batch dimensions, i.e., our input tensors are typically of the shape (batch_size, d), where d is the number of parameters in the distribution.

```python
import torch
alpha_a = torch.tensor([
    [1.5, 17.5],
    [1.5, 14.5],
    [2.5, 5.5]
])
alpha_b = torch.tensor([
    [1.5, 17.5],
    [1.5, 17.5],
    [1.5, 17.5]
])
dist = chernoff_dist(alpha_a, alpha_b)
```

`dist` is a `torch.Tensor` containing the respective distances of the distributions. We can convert this to a list and output it. The values are:

```
[0.0, 0.006, 0.609]
```

The results make intuitive sense: the distance between identical distributions is zero and it increases if we make one distribution wider while fixing the other.