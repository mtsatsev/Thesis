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

from nflows.transforms.made import MADE
```

# Sampling Based Inference

We would like to perform inference on a model’s parameters $\xi$ given an observation _S_

$$p(\xi|S)$$.

This can be expressed with Bayes rule

$$p(\xi|S) = \frac{p(S|\xi)p(\xi)}{p(S)}$$.

However, this computation requires the evaluation of the likelihood $p(S|\xi)$. But what happens when we don't know the likelihood? What if the best we can do is simulate $S \sim Simulator(\xi)$. In that case we simulate enough samples and fit a function $p_{\theta}(\xi|S)$ which maximizes the likelihood over the data.

# Sinusoidal Waves data.

Given a set of parameters $\xi = \{A, f, \varphi\}$ and a signal
$\begin{align}
S(t) = A \sin(2\pi f t + \varphi) + U(t)
end{\align}$, we can simulate our dataset as a pair of parameters and signal $\{(\xi,S)_i\}$ for any number of times.

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

# Spline trasnfrom


$$
\begin{algorithm}
\caption{Rational Quadratic Spline for input \textbf{x} and context \textbf{h}}\label{alg:splinealg}
\begin{algorithmic}
\State $\theta = \mathcal{NN}(\mathbf{x,h})$
\State $[\theta^W,\theta^H,\theta^D] = \theta$ \Comment{Split}
\State \textbf{W} = [\textbf{0}, softmax($\theta^W$)] \Comment{Pad with \textbf{0} vector}
\State \textbf{H} = [\textbf{0}, softmax($\theta^H$)] \Comment{Pad with \textbf{0} vector}
\For {i in \textbf{W} and \textbf{H}}
    \State \textbf{X}$_i$ = $\sum^i_{k=1}\mathbf{W}_k$ \Comment{Cumulative sum}
    \State \textbf{Y}$_i$ = $\sum^i_{k=1}\mathbf{H}_k$ \Comment{Cumulative sum}
\EndFor
\State $\mathbf{X}$ = 2 $\cdot$ B $\cdot \mathbf{X} - B$
\State $\mathbf{Y}$ = 2 $\cdot$ B $\cdot \mathbf{Y} - B$
\State \textbf{D} = softplus([\textbf{C},$\theta^D$,\textbf{C}]) \Comment{Pad with C=$\log(e^{1-1e-3}-1)$}
\If { forward }
\State \textbf{P} = Search(\textbf{x},\textbf{X})
\Else 
\State \textbf{P} = Search(\textbf{x},\textbf{Y})
\EndIf
\State $x_i,x_{i+1}$ = Gather(\textbf{X},\textbf{P}) \Comment{Search the \textit{x} values}
\State $y_i,y_{i+1}$ = Gather(\textbf{Y},\textbf{P}) \Comment{Search the \textit{y} values}
\State $\delta_i,\delta_{i+1}$ = Gather(\textbf{D},\textbf{P}) \Comment{Search the $\delta$ values}
\If{ forward }
    \State $\mathbf{z} = y_i + \mathlarger{\frac{(y_{i+1} - y_i)\Big[ s_i\phi^2 + \delta_i \phi(1-\phi) \Big]}{s_i + \Big[ \delta_i + \delta_{i+1} - 2s_i \Big]\phi(1-\phi)}}$ \Comment{Equations 2.25, 2.18, 2.19}
    \State  $ \mathbf{J}_f =  \log(s_i^2\Big[\delta_{i+1} \phi^2 + 2s_i\phi(1-\phi) + \delta_i(1-\phi)^2 \Big]) - 2 * \log(\Big[ \beta_i(\phi)\Big])$
\Else
    \State  $\mathbf{z} = \mathlarger{\frac{2c}{-b-\sqrt{b^2-4ac}}w_i +x_i}$ \Comment{Equations 2.33, 2.34, 2.35, 2.36}
    
    \State  $ \mathbf{J}_f =  -\log(s_i^2\Big[\delta_{i+1} \phi^2 + 2s_i\phi(1-\phi) + \delta_i(1-\phi)^2 \Big]) - 2 * \log(\Big[ \beta_i(\phi)\Big])$
\EndIf
\end{algorithmic}
\end{algorithm}
$$



# Inverse Autoregressive Flow

A normalizing flow is a transformation from a normal distribution to some complex distribution where this is allowed given the change of variables formula.

It can sample $\mathbf{z} = f_{\theta}(\mathbf{x})$ and evaluation densities

$$p_{\theta}(\mathbf{z}) = p_x(\mathbf{x}) |det(\frac{\partial \mathbf{x}}{\mathbf{z}})|$$

, where $p_{\theta}$ is the posterior distribution, $p_x$ the normal distribution and $|det(\frac{\partial \mathbf{x}}{\mathbf{z}})|$.

Normalizing flows must be:
1. Invertible.
2. Differentiable.
3. Easy to calculate the determinant(Autoregressive).


An inverse autoregressive layer consists of a list of autoregressive networks (MADE) which we take from the nflows library [nflows](https://github.com/bayesiains/nflows/blob/master/nflows/transforms/made.py) that produces autoregressive parameters $\theta$ based on the prior distribution, which we denote as X and uses those parameters to transform them to the posterior, denoted z during sampling and from z to x during training. The prior is a three-dimensional Gaussian $\mathcal{N}(0:1)^3$.


The reason why we want a list of autoregressive networks is because we might want to permute the parameters $\xi$, each with its own dedicated MADE network, between which the features are permuted to ensure full dependence of every feature on all others.

To implement the splines the output of the last MADE has to be 3 * the number of knots - 1.

```python
class IAFBlock(nn.Module):
    """ An inverse autoregressive flow block that uses a MADE network to parameterize an affine transformation """

    def __init__(self, dim: int, context_dim: int, hidden: int, made_num_blocks: int, num_mades: int, rotations: bool , K: int, B: int):
        '''
        :param dim            : input dimensions. In this case these are the A,f,phi therefor dim = 3
        :param context dim    : corresponds to the length of the wave
        :param hidden         : the number of hidden neurons in the residual blocks in MADE
        :param made_num_blocks: the depth of MADE (for this problem 2 is enough)
        :param num_mades      : the number of MADE networks in each IAF layer
        :param rotations      : Do we want to rotate the parameters between the mades. This introduces hopfield like dependency in the network
        :param K              : number of knots for the spline
        :param B              : boundary of the spline
        '''
        super().__init__()
        if rotations and (num_mades%dim != 0):
            raise ValueError("If using rotations then number of mades must be a multiple of the number of input dimensions. Input dimensions: {}, number of mades: {}".format(dim,num_mades))
        self.dim = dim
        self.context_dim = context_dim
        self.net = nn.ModuleList()
        for _ in range(num_mades-1):
            self.net.append(MADE(features=dim,
                                  hidden_features=hidden,
                                  num_blocks=made_num_blocks,
                                  context_features=context_dim,
                                  output_multiplier=1,
                                  activation=F.selu))

        self.net.append(MADE(features=dim,
                              hidden_features=hidden,
                              num_blocks=made_num_blocks,
                              context_features=context_dim,
                              output_multiplier=3*K-1,
                              activation=F.selu))

```
