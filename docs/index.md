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
$$\begin{align}
S(t) = A \sin(2\pi f t + \varphi) + U(t)
end{\align}$$, we can simulate our dataset as a pair of parameters and signal $\{(\xi,S)_i\}$ for any number of times.

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

# Spline transform

Based on ``Algorithm 1 Rational Quadratic Spline for input x and context h`` in the Thesis

## Important Equations 

$$
\begin{align}
a &= (y_{i+1}s_i - y_{i+1}\delta_i - y_is_i + y_i\delta_i) + (-y_i\delta_i -y_i\delta_{i+1} + 2y_is_i +\text{y}\delta_i + \text{y}\delta_{i+1} - 2\text{y}s_i)
&=(y_{i+1}-y_i)\Big[s_i - \delta_i\Big] + (\text{y}-y_i)\Big[ \delta_{i+1} + \delta_i - 2s_i \Big],
b &= (y_{i+1}\delta_i + y_i \delta_i) + (y_i\delta_i + y_i\delta_{i+1}-2y_is_i - \text{y}\delta_i - \text{y}\delta_{i+1} + \text{y}2s_i)
&= (y_{i+1} - y-i) \delta_i + (y_i - \text{y})\Big[ \delta_i + \delta_{i+1} + 2s_i\Big]
&= (y_{i+1} - y-i) \delta_i - (\text{y}-y_i)\Big[ \delta_i + \delta_{i+1} + 2s_i\Big],
c &= y_is_i - \text{y}s_i = -s_i(\text{y}-y_i)
\end{align}
$$


To implement the transform we create a class for it. 

The skeleton structure of the class looks as follows:

```python
splineShared = collections.namedtuple('splineShared','x_k,y_k,delta_k,delta_kp1,w_k,h_k,s_k')

class RationalQuadraticSpline():
  
    def __init__(self,W,H,D,bounds):
        self.W = W
        self.H = H
        self.D = self.create_derivatives(D,1e-3)
        self.B = bounds

    def set_theta(self,theta):
          W,H,D = torch.chunk(theta,3,dim=-1)
          self.W = W / np.sqrt(128)
          self.H = H / np.sqrt(128)
          self.D = self.create_derivatives(D,1e-3)

    def create_derivatives(self,derivatives,min_deriv):
        pass    

    def compute_shared(self,x=None,y=None,W=None,H=None,D=None):
        pass

    def forward(self,x):
        pass

    def backward(self,z):
        pass

    def derivative(self,d,phi):
        pass
```

We can observe that forward(Equation 2.25), backward(Eqations 2.33, 2.34, 2.35, 2.36) and derivative(Equation 2.30) computations all use the same parameters $x_i,x_{i+1},y_i,y_{i+1},\delta_i,\delta_{i+1},s_i,w_i,h_i$ and therefor we consider them to be shared variables and we construct them in a different function ``compute_shared``. 

The implementation of ``compute_shared`` represents the whole algorithm from splitting $\theta$ to gathering the variables. To do that we need 3 additional functions:
1. We see that $\theta^W$ and $\theta^H$ that are treated as the unnormalized widths and heights undergo the same normalization to obtain the widths and heights. So we can use a function to represent that update, although, this is not necessary it shortens the code base. The function is called ``update_WH``. 
2. We need a function to search in knots (widths and heights). In section (3.3.1) a formula is presented which acts like a drop in replacement for binary search ``search_knot``. 
3. We need a function to gather from the widths, heights and derivatives. Pytorch already has ``.gather`` function implemented for its tensors. 

```python

def update_WH(self,WH):
      '''Implementation 3. from the paper for a single theta^W or theta^H'''

      # Apply softmax
      param = F.softmax(WH,dim=-1) 

      # This normalization is required
      param = 1e-3 + (1 - 1e-3 * WH.shape[-1]) * param

      # Cumulative sum
      cumulative_param = torch.cumsum(param,dim=-1)

      # Pad and place within the bounds
      cumulative_param = F.pad(cumulative_param,pad=(1,0),mode='constant',value=0.0)
      cumulative_param = 2 * self.B * cumulative_param - self.B
      cumulative_param[...,0]  = -self.B
      cumulative_param[...,-1] =  self.B
      
      return cumulative_param

def search_knot(self,x,WH,eps=1e-6):
    WH[...,-1] +=eps
    return torch.sum(x[...,None] >= WH,dim=-1) - 1  
```

The implementation of ``compute_shared`` goes in a few steps:
1. Update all $\theta$'s. 
2. Check if we need the forward or inverse function. 
  * If forward search in the widths.
  * If inverse search in the heights.
3. Gather $x_i,x_{i+1},y_i,y_{i+1},\delta_i,\delta_{i+1}$
4. Compute $s_i,w_i,h_i$.
5. Return a ``splineShared`` tuple with the values.

```python
def compute_shared(self,x=None,y=None,W=None,H=None,D=None):
      # At least x or y must be specified
      assert (x is None) != (y is None)
      # Check if we need to search in the widths or heights
      is_x = (x is not None)

      # Update theta^W and theta^H to obtain {x,y}_k
      xs = self.update_WH(W)
      ys = self.update_WH(H)

      # Update the derivatives (1e-3 + is needed for numeric stability)
      derivatives = 1e-3 + F.softplus(D)

      # Search in the widths or the heights
      if is_x:
          knot_positions = self.search_knot(x,xs)[...,None]
      else:
          knot_positions = self.search_knot(y,ys)[...,None]

      # Point 3
      x_k   = xs[...,:-1].gather(-1,knot_positions)[...,0] 
      x_kp1 = xs[...,1:].gather(-1,knot_positions)[...,0]

      y_k   = ys[...,:-1].gather(-1,knot_positions)[...,0] 
      y_kp1 = ys[...,1:].gather(-1,knot_positions)[...,0]

      delta_k   = derivatives.gather(-1,knot_positions)[...,0]
      delta_kp1 = derivatives[...,1:].gather(-1,knot_positions)[...,0]

      # Point 4
      w_k = (x_kp1 - x_k) #input_bin_widths = widths.gather
      h_k = (y_kp1 - y_k) #input_heights
      s_k = h_k / w_k # input_delta = (heights/widths).gather

      # Point 5
      return splineShared(
          x_k=x_k,
          y_k=y_k,
          delta_k=delta_k,
          delta_kp1=delta_kp1,
          w_k=w_k,
          h_k=h_k,
          s_k=s_k
      )
```

Next, before we plug those values in the formulas of the equations, first we need to make sure that we only do that for values that fall within the range of the bounds ``B``.
 
```python
def forward(self,x):

    # Transformations and piecewise so dimensions are identical in this case that would be [batch_size,3]
    z      = torch.zeros_like(x)
    logdet = torch.zeros_like(x)

    # To ensure that values outside the bound are unchange, we mask those values with a binary tensor. 
    inside_mask = (x >= -self.B) & (x <= self.B)
    outside_mask = ~inside_mask

    # Copy the outside values and set their derivative to 0
    z[outside_mask] = x[outside_mask]
    logdet[outside_mask] = 0

    # Perform computations inside the bounds
    inp = x[inside_mask]
    if torch.any(inside_mask):
      
        # d: a palceholder for all variables
        d = self.compute_shared(x=inp,W=self.W[inside_mask,:],H=self.H[inside_mask,:],D=self.D[inside_mask,:])

        # Equation 2.20
        phi = (inp - d.x_k) / d.w_k

        # Equation 2.25
        z[inside_mask] = (
        d.y_k + (d.h_k * (d.s_k * phi.pow(2) + d.delta_k * phi*(1-phi) )) /
                (d.s_k + (d.delta_kp1 + d.delta_k - 2 * d.s_k) * phi*(1-phi))
        )
        logdet[inside_mask] = self.derivative(d,phi)
    return z,logdet
```

And the inverse:

```python

def backward(self,z):

    # Transformations and piecewise so dimensions are identical in this case that would be [batch_size,3]
    x      = torch.zeros_like(z)
    logdet = torch.zeros_like(z)

    # To ensure that values outside the bound are unchange, we mask those values with a binary tensor. 
    inside_mask = (z >= -self.B) & (z <= self.B)
    outside_mask = ~inside_mask

    # Copy the outside values and set their derivative to 0
    x[outside_mask] = z[outside_mask]
    logdet[outside_mask] = 0

    # Perform computations inside the bounds
    inp    = z[inside_mask]
    if torch.any(inside_mask):
        # d: a placeholder for all variables
        d = self.compute_shared(y=inp,W=self.W[inside_mask,:],H=self.H[inside_mask,:],D=self.D[inside_mask,:])

        
        input_term = (inp - d.y_k)
        delta_term = input_term * (d.delta_kp1 + d.delta_k - 2 * d.s_k)

        # Equation 2.34
        a = d.h_k * (d.s_k - d.delta_k) + delta_term
        # Equation 2.35
        b = d.h_k * d.delta_k - delta_term
        # Equation 2.36
        c = -d.s_k * input_term

        discriminant = b.pow(2) - 4 * a * c
        discriminant[discriminant <= 0] = 0

        # Equation 2.33. 
        phi = (2*c)/(-b - torch.sqrt(discriminant))

        # Sometimes it leaves the [0,1] range so we need to return it back.
        phi[phi>=1] = 1 - 1e-6
        phi[phi<=0] = 0 + 1e-6

        # Transform
        x[inside_mask] = phi * d.w_k + d.x_k
        logdet[inside_mask] = self.derivative(d,phi)
    return x,-logdet
```

Where the derivative for both forward and backward is

```python
def derivative(self,d,phi):
    '''
    :d  : placeholder for the values 
    :phi: function of the input 
    '''
    # Equation 2.30
    numerator = d.s_k.pow(2) * (
        d.delta_kp1 * phi.pow(2)
        + 2 * d.s_k * phi*(1-phi)
        + d.delta_k  * (1-phi).pow(2)
    )
    denominator = d.s_k  + (
        (d.delta_k + d.delta_kp1 - 2 * d.s_k)
        * phi*(1-phi)
    )
    return torch.log(numerator) - 2 * torch.log(denominator)

```

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
