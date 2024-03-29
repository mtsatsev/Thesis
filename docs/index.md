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

from IPython.core.pylabtools import figsize
from nflows.transforms.made import MADE
```

# Sampling Based Inference

We would like to perform inference on a model’s parameters $\xi$ given an observation _S_

$$p(\xi|S)$$.

This can be expressed with Bayes rule

$$p(\xi|S) = \frac{p(S|\xi)p(\xi)}{p(S)}$$.

However, this computation requires the evaluation of the likelihood $p(S|\xi)$. But what happens when we don't know the likelihood? What if the best we can do is simulate $S \sim Simulator(\xi)$. In that case we simulate enough samples and fit a function $p_{\theta}(\xi|S)$ which maximizes the likelihood over the data.

# Sinusoidal Waves data.

Given a set of parameters $\xi = \{A, f, \varphi\}$ and a signal $S(t) = A \sin(2\pi f t + \varphi) + U(t)$,
we can simulate our dataset as a pair of parameters and signal $\{(\xi,S)_i\}$ for any number of times.

The parameters have the following properties:

| Parameter | Prior   | Minimum | Maximum |
|-----------|---------|---------|---------|
| Amplitude | Uniform | 0.2     | 1       |
| Frequency | Uniform | 0.1     | 0.25    |
| Phase     | Uniform | 0       | 2 $\pi$  |

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
\phi(\text{x}) &= \frac{(\text{x} - x_i)}{w_i} && (1)\\
\frac{\alpha_i(\phi(\text{x}))}{\beta_i(\phi(\text{x}))}&= y_i + \frac{(y_{i+1} - y_i)\Big[ s_i\phi^2 + \delta_i \phi(1-\phi) \Big]}{s_i + \Big[ \delta_i + \delta_{i+1} - 2s_i \Big]\phi(1-\phi)} && (2)\\
\frac{d}{d\text{x}}\Big[ \frac{\alpha_i(\phi(\text{x})}{\beta_i(\phi(\text{x})} \Big]&=w_is_i^2 \Big[ \delta_{i+1}\phi^2 + 2s_i\phi(1-\phi) + \delta_i(1-\phi)^2 \Big] && (3)\\
a &=(y_{i+1}-y_i)\Big[s_i - \delta_i\Big] + (\text{y}-y_i)\Big[ \delta_{i+1} + \delta_i - 2s_i \Big] && (4)\\
b &= (y_{i+1} - y-i) \delta_i - (\text{y}-y_i)\Big[ \delta_i + \delta_{i+1} + 2s_i\Big] && (5)\\
c &= y_is_i - \text{y}s_i = -s_i(\text{y}-y_i) && (6)\\
\phi(\text{x}) &= \frac{2c}{-b-\sqrt{b^2-4ac}} && (7)
\end{align}
$$




To implement the transform we create a class for it.

The skeleton structure of the class looks as follows:

```python
splineShared = collections.namedtuple('splineShared','x_i,y_i,delta_i,delta_ip1,w_i,h_i,s_i')

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

We can observe that forward(Equation 2), backward(Eqations 4, 5, 6, 7) and derivative(Equation 3) computations all use the same parameters $x_i,x_{i+1},y_i,y_{i+1},\delta_i,\delta_{i+1},s_i,w_i,h_i$ and therefor we consider them to be shared variables and we construct them in a different function ``compute_shared``.

The implementation of ``compute_shared`` represents the whole algorithm from splitting $\theta$ to gathering the variables. To do that we need 3 additional functions:
1. We see that $\theta^W$ and $\theta^H$ that are treated as the unnormalized widths and heights undergo the same normalization to obtain the widths and heights. So we can use a function to represent that update, although, this is not necessary it shortens the code base. The function is called ``update_WH``.
2. We need a function to search in knots (widths and heights). In section (3.3.1) a formula is presented which acts like a drop in replacement for binary search. The formula is $\mathbf{P}_{\mathbf{x}_n} = (\sum \mathbf{x}_n \geq \mathbf{X}_n+\epsilon) - 1$ implemented in ``search_knot``.
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

      # Update theta^W and theta^H to obtain {x,y}_i
      xs = self.update_WH(W)
      ys = self.update_WH(H)

      # Update the derivatives (1e-3 + is needed for numeric stability)
      derivatives = 1e-3 + F.softplus(D)

      # Search in the widths or the heights
      if is_x:
          knot_positions = self.search_inot(x,xs)[...,None]
      else:
          knot_positions = self.search_inot(y,ys)[...,None]

      # Point 3
      x_i   = xs[...,:-1].gather(-1,knot_positions)[...,0]
      x_ip1 = xs[...,1:].gather(-1,knot_positions)[...,0]

      y_i   = ys[...,:-1].gather(-1,knot_positions)[...,0]
      y_ip1 = ys[...,1:].gather(-1,knot_positions)[...,0]

      delta_i   = derivatives.gather(-1,knot_positions)[...,0]
      delta_ip1 = derivatives[...,1:].gather(-1,knot_positions)[...,0]

      # Point 4
      w_i = (x_ip1 - x_i)
      h_i = (y_ip1 - y_i)
      s_i = h_i / w_i

      # Point 5
      return splineShared(
          x_i=x_i,
          y_i=y_i,
          delta_i=delta_i,
          delta_ip1=delta_ip1,
          w_i=w_i,
          h_i=h_i,
          s_i=s_i
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

        # Equation 1
        phi = (inp - d.x_i) / d.w_i

        # Equation 2
        z[inside_mask] = (
        d.y_i + (d.h_i * (d.s_i * phi.pow(2) + d.delta_i * phi*(1-phi) )) /
                (d.s_i + (d.delta_ip1 + d.delta_i - 2 * d.s_i) * phi*(1-phi))
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


        input_term = (inp - d.y_i)
        delta_term = input_term * (d.delta_ip1 + d.delta_i - 2 * d.s_i)

        # Equation 4
        a = d.h_i * (d.s_i - d.delta_i) + delta_term
        # Equation 5
        b = d.h_i * d.delta_i - delta_term
        # Equation 6
        c = -d.s_i * input_term

        discriminant = b.pow(2) - 4 * a * c
        discriminant[discriminant <= 0] = 0

        # Equation 7.
        phi = (2*c)/(-b - torch.sqrt(discriminant))

        # Sometimes it leaves the [0,1] range so we need to return it back.
        phi[phi>=1] = 1 - 1e-6
        phi[phi<=0] = 0 + 1e-6

        # Transform
        x[inside_mask] = phi * d.w_i + d.x_i
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
    numerator = d.s_i.pow(2) * (
        d.delta_ip1 * phi.pow(2)
        + 2 * d.s_i * phi*(1-phi)
        + d.delta_i  * (1-phi).pow(2)
    )
    denominator = d.s_i  + (
        (d.delta_i + d.delta_ip1 - 2 * d.s_i)
        * phi*(1-phi)
    )
    return torch.log(numerator) - 2 * torch.log(denominator)

```

Full code can be seen at [RationalQuadraticSpline](https://github.com/mtsatsev/Thesis)

# Inverse Autoregressive Flow

![docs/IAF.jpg](https://github.com/mtsatsev/Thesis/blob/main/docs/IAF.png)


A normalizing flow is a transformation from a normal distribution to some complex distribution where this is allowed given the change of variables formula.

It can sample $\mathbf{z} = f_{\theta}(\mathbf{x})$ and evaluation densities

$$p_{\theta}(\mathbf{z}) = p_x(\mathbf{x}) |det(\frac{\partial \mathbf{x}}{\mathbf{z}})|$$

, where $p_{\theta}$ is the posterior distribution, $p_x$ the normal distribution and $|det(\frac{\partial \mathbf{x}}{\mathbf{z}})|$ is the absolute values of the determinant.

Normalizing flows must be:
1. Invertible.
2. Differentiable.
3. Easy to calculate the determinant(Autoregressive).


An inverse autoregressive layer consists of a list of autoregressive networks (MADE) which we take from the nflows library [nflows](https://github.com/bayesiains/nflows/blob/master/nflows/transforms/made.py) that produces autoregressive parameters $\theta$ based on the prior distribution, which we denote as X and uses those parameters to transform them to the posterior, denoted Z during sampling and from z to x during training.

1. The prior is a three-dimensional Gaussian $\mathcal{N}(0:1)^3$.

2. The reason why we want a list of autoregressive networks is because we might want to permute the parameters $\xi$, each with its own dedicated MADE network, between which the features are permuted to ensure full dependence of every feature on all others.

We begin with a skeleton implementation of a single layer:

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

    def forward(self, x: Tensor, context: Tensor):
        pass
    def Backward(self, z: Tensor, context: Tensor):
        pass
```

If we want to use permutations it only makes sense to use the same amount of MADEs as the number of dimensions.

To implement the splines the output of the last MADE has to be 3 * the number of knots - 1. The rest of the MADEs will have output dimensions equal to the input dimensions. In cases where we set num_mades=1 we basically have no permutaions.

```python
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
                for m in range(num_mades-1):
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

                theta = self.net[-1](torch.zeros(dim),torch.zeros(context_dim)).reshape(-1,dim,3*K-1)
                W,H,D = torch.chunk(theta,3,dim=-1)
                self.spline = RationalQuadraticSpline(W,H,D,B)
                self.K = K
                self.B = B
                self.rotations = rotations
```

For the forward pass we have:

```python
def forward(self, x: Tensor, context: Tensor) -> [Tensor,Tensor]:
    """ Sequential, used during training """

    # Create the log determinant and the posterior
    x = torch.zeros_like(z)
    log_det = torch.zeros(z.shape[0]).to(z)

    # Complete the autoregressive series...
    for i in range(self.dim):
        # construct theta, condition NOT on the posterior
        made_out = x.clone()
        for idx,made in enumerate(self.net):
            if self.rotations:
                made_out = made(made_out.roll(-1,1),context)#.
            else:
                made_out = made(made_out,context)

        theta = made_out.reshape(-1,self.dim,self.K*3-1)

        # Create the splines
        self.spline.set_theta(theta)

        # forward function during training
        x, log_det = self.spline.forward(z)
    return x, log_det
```

And backward:


```python
def backward(self, x: Tensor, context: Tensor) -> [Tensor, Tensor]:
      """ Parallel, used for sampling """

      # Create thata, condition on the prior
      made_out = x.clone()
      for idx,made in enumerate(self.net):
          if self.rotations:
              made_out = made(made_out.roll(-1,1),context)#.
          else:
              made_out = made(made_out,context)
      theta = made_out.reshape(-1,self.dim,self.K*3-1)

      # Create the splines
      self.spline.set_theta(theta)

      # inverse function during sampling
      z,log_det = self.spline.backward(x)
      return z, log_det
```

Creating the whole flow is simply done by stacking a bunch of ``IAFBlock``s.

```python

class IAF(nn.Module):
    """ Inverse autoregressive Flows, mostly a wrapper to put multiple IAF Blocks on top of each other """

    def __init__(self, num_blocks: int, dim: int, context_dim: int, hidden: int, made_num_blocks: int, num_mades: int, rotations: bool, K: int, B: int):
        '''
        :param num_block      : number of IAFblock layers
        :param dim            : number of input dimensions   (3 parameters)
        :param context_dim    : number of context dimensions (24 time steps)
        :param hidden         : number of neurons in the hidden layers of MADE
        :param made_num_blocks: depth of each made
        :param rotations      : do we want permutaions or not
        :param K              : number of knots
        :param B              : boudary of the spline
        '''
        super().__init__()

        # Setup flows
        flows = []
        print(rotations_made)
        for i in range(num_blocks):
            block = IAFBlock(dim=dim,
                             context_dim=context_dim,
                             hidden=hidden,
                             made_num_blocks=made_num_blocks,
                             num_mades=num_mades,
                             rotations=rotations,
                             K=K,
                             B=B)
            flows.append(block)
        self.flows = nn.ModuleList(flows)

    def forward(self, x, context):
        """ Forward pass, SLOW!!! """
        log_det = torch.zeros(x.shape[0]).to(x)
        for flow in self.flows:
            x, ld = flow.forward(x, context)
            log_det += ld.sum(dim=-1)
        return x, log_det

    def backward(self, z, context):
        """ Backward pass, FAST """
        log_det = torch.zeros(z.shape[0]).to(z)
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, context)
            log_det += ld.sum(dim=-1)
        return z, log_det
```

With all components ready all we need to do is put it all together. If you want to run this experiment in colab/jupyter then read further if you want to run it locally then look at [swave](here)

1. Start by preparing settings and hyperparameters for the model:

```python
device='cuda' if torch.cuda.is_available() else 'cpu'
num_epochs  = 60
trn_samples = 50000
val_samples = 10000
batch_size  = 128
num_workers = 2
n_samples   = 10000
lr          = 1e-4
hidden      = 128
num_blocks  = 3
context_dim = 24
n_in = 3
K=9
B=6.5
rotations=False
rotations_made=True
num_mades=3
made_num_blocks=2
ns=0.0
```

2. Sample train, validation and testing datasets:

```python
trn_data = SineWave(num_samples=trn_samples, noise_strength=ns,context=context_dim)
val_data = SineWave(num_samples=val_samples, noise_strength=ns,context=context_dim)
test_data= SineWave(num_samples=val_samples, noise_strength=ns,context=context_dim)

trn_loader = DataLoader(trn_data, batch_size, num_workers=num_workers, pin_memory=True,drop_last=True)
val_loader = DataLoader(val_data, batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_data, batch_size, num_workers=num_workers, pin_memory=True)
```

3. Create the model

```python
model = IAF(num_blocks=num_blocks,
            dim=n_in,
            context_dim=context_dim,
            hidden=hidden,
            made_num_blocks=made_num_blocks,
            num_mades=num_mades,
            rotations=rotations,
            rotations_made=rotations_made,
            K=K,
            B=B)
model = model.to(device)
```

4. Set up optimizer and prior
```python
# Setup optimizer and scheduler
opt = torch.optim.Adam(params=model.parameters(), lr=lr, amsgrad=True)
shl = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)

# Setup prior, we go for a Gaussian since it is simple
prior = torch.distributions.MultivariateNormal(loc=torch.zeros(size=(n_in,)).to(device), covariance_matrix=torch.eye(n=n_in).to(device))
```

5. Start training and validate as the model train. The model is not saved at the end of training in this example.

```python
trn_losses = []
  val_losses = []
  lowest_loss = 0

  for epoch in range(args.num_epochs):
      # Loss setup
      trn_loss = 0
      val_loss = 0
      print("epoch: ",epoch)
      model.train(True)
      # Loading of the data
      loop = tqdm(trn_loader)
      loop.set_description(f"Training")
      for strain, para in loop:

          # Prepare the input and context
          x       = para.to(device)
          context = strain.to(device)
          opt.zero_grad()

          # Forward propagation
          z, log_det    = model.forward(x, context)

          prior_logprob = prior.log_prob(z)
          loss = -torch.mean(prior_logprob + log_det)

          # Backpropagation
          loss.backward()
          opt.step()

          # Update metrics
          trn_loss += (loss.item() * x.shape[0]) / len(trn_loader.dataset)
      print("training loss: ", trn_loss)
      # Validation

      with torch.no_grad():
          model.train(False)
          for (strain, para) in val_loader:

              # Prepare the input and context
              x       = para.to(device)
              context = strain.to(device)
              opt.zero_grad()

              z, log_det = model.forward(x, context)
              prior_logprob = prior.log_prob(z)
              loss = -torch.mean(prior_logprob + log_det)

              # Update Metrics
              val_loss += (loss.item() * x.shape[0]) / len(val_loader.dataset)

      print("validation loss",val_loss)
      trn_losses.append(trn_loss)
      val_losses.append(val_loss)

      # Update the scheduler
      shl.step()
```

After training you can visualize the loss

```python
fig,ax = plt.subplots(1,1,figsize=(12,8))
ax.plot(line,trn)
ax.plot(line,val)
ax.set_xlabel("Epochs")
ax.set_ylabel('Loss')
ax.set_title("Training loss")
ax.legend(['Train loss', 'Validation loss'],prop={'size': 25})
plt.show()
```
To test the performance of the network we can visualize the posterior that it createse for ``n_samples``

```python
with torch.no_grad():
    model.train(False)
    for context, x in test_loader:
        context = context.to(device)
        x = x.to(device)
        opt.zero_grad()
        #We now go from prior to posterior instead of the other way around, to keep it simple we only do it for a single context vector
        z = prior.sample([n_samples])
        context = torch.cat(n_samples * [context[0:1]])
        x_estimated, _ = model.backward(z, context)
        break

ss = x_estimated.cpu().detach().numpy()
xx = x.cpu().detach().numpy()

# You are free to play around with the numbers used here for the quantiles, levels etc.
dom = [[0,1],[0,1],[0,6.5]]
fig = corner.corner(data=ss,
                        truths=(xx[-0]),
                        labels=['ampli', 'frequency', 'phase'],
                        bins=50,
                        smooth=0.9,
                        color='tab:blue',
                        truth_color='#FF8C00',
                        quantiles=[0.39, 0.86, 0.98],
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2)),
                        plot_density=True,
                        fill_contours=True,
                        hist_kwargs=dict(density=True),
                        range=dom)
```
![solve](https://github.com/mtsatsev/Thesis/blob/f19813b11e7d9a2cc62c7761fecfe2de0599bea1/docs/3layerssolve.png)


This concludes the toy problem. For Gravitational Waves take a look at the [gwave][folder].  

