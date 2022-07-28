# Parameter inference of Sinusoidal Waves

# Introduction

Here we will take a closer look at the technical implementation of an inverse autoregressive flow model with spline transformation for sampling based inference. In effect this closely follows the implementation of the algorithm used in my Thesis _Parameter Inference of Gravitational Waves using Inverse Autoregressive Spline Flow_, and produced the results in Appendix A.

The network and data are implemented using Pytorch, the visualization is done with matplotlib and corner.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import distributions
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
from typing import List, Tuple
from typing_extensions import Literal
import math
import matplotlib.pyplot as plt
import corner
import numpy as np
```

# Sampling Based Inference

We would like to perform inference on a model’s parameters $\xi$ given an observation _S_

$$p(\xi|S)$$.

This can be expressed with Bayes rule

$$p(\xi|S) = \frac{p(S|\xi)p(\xi)}{p(S)}$$.

However, this computation requires the evaluation of the likelihood $p(S|\xi)$. But what happens when we don't know the likelihood? What if the best we can do is simulate $S \sim Simulator(\xi)$. In that case we simulate enough samples and fit a function $p_{\theta}(\xi|S)$ which maximizes the likelihood over the data.

# Sinusoidal Waves data.

Given a set of parameters $\xi = \{A, f, \varphi\}$ and a signal $S(t) = A \sin(2\pi f t + \varphi) + U(t) (1)$ we can simulate our dataset as a pair of parameters and signal $\{(\xi,S)_i\}$ for any number of times.

The parameters have the following properties:

| Parameter | Prior   | Minimum | Maximum |
|-----------|---------|---------|---------|
| Amplitude | Uniform | 0.2     | 1       |
| Frequency | Uniform | 0.1     | 0.25    |
| Phase     | Uniform | 0       | 2$\pi$  |

And our simulator is implemented as follows:

```python
class SineWave(Dataset):
    """ A sine wave whose parameters (amplitude, frequency, phase) are sampled from distributions """

    def __init__(self, num_samples: int, noise_strength: float, context:int=24):
        """
        :param num_samples   : number of samples this dataset will generate per epoch
        :param noise_strength: noise is sampled from a uniform distribution[-noise_strength, +noise_strength]
        :param context       : the length of the signal
        """
        self.num_samples = num_samples

        # Setup the distributions
        self.amp_dist = distributions.Uniform(low=0.2, high=1)
        self.freq_dist = distributions.Uniform(low=0.1, high=0.25)
        self.phase_dist = distributions.Uniform(low=0, high=2*math.pi)
        self.noise_dist = distributions.Uniform(low=-noise_strength-1e-6, high=noise_strength+1e-6)
        self.context = context

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        amp = self.amp_dist.sample((1,))
        freq = self.freq_dist.sample((1,))
        phase = self.phase_dist.sample((1,))
        x = torch.linspace(start=-3 * math.pi, end=3 * math.pi, steps=self.context)

        # Equation (1)
        return amp * torch.sin(2 * math.pi * freq * x + phase) + self.noise_dist.sample((self.context,)), torch.cat([amp, freq, phase])
```

We can visualize a single sample as follows:

```python
torch.set_printoptions(precision=2)
trn_data = SineWave(num_samples=100, noise_strength=0,context=24)
for (S,xi) in trn_data:
    plt.plot(S)
    print('Amplitude,\t Frequency, \t Phase')
    print(str(xi[0].item())[:10]+'\t '+str(xi[1].item())[:10]+'\t '+str(xi[2].item())[:10])
    break
plt.show()
```

# Inverse Autoregressive Flow

A normalizing flow is a transformation from a normal distribution to some complex distribution where this is allowed given the change of variables formula.

It can sample $\mathbf{z} = f_{\theta}(\mathbf{x})$ and evaluation densities

$$p_{\theta}(\mathbf{z}) = p_x(\mathbf{x}) |det(\frac{\partial \mathbf{x}}{\mathbf{z}})|$$

, where $p_{\theta}$ is the posterior distribution, $p_x$ the normal distribution and $|det(\frac{\partial \mathbf{x}}{\mathbf{z}})|$.

Normalizing flows must be:
1. Invertible.
2. Differentiable.
3. Easy to calculate the determinant.





# Part 1: Data
