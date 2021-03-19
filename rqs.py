import torch
import torch.nn.functional as F
import numpy as np

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_rqs(inputs,W,H,D,shape,tail=1,inverse=False):
    inside_mask = (inputs >= -tail) & (inputs <= tail)
    outside_mask = ~inside_mask
    # mask = [784]

    outputs = torch.zeros_like(inputs)
    log_det = torch.zeros_like(inputs)
    # lg, out = [784]
    print("INPUTS")
    print(inputs.size())
    print("INPUTS")


    D = F.pad(D,pad=(1,1))
    D[..., 0] = 1
    D[...,-1] = 1
    outputs[outside_mask] = inputs[outside_mask]
    # W = [784,3]

    outputs[inside_mask], log_det[inside_mask] = rqs(
        inputs[inside_mask],W[inside_mask,:],H[inside_mask,:],D[inside_mask,:],inverse
        ,left=-tail
        ,right=tail
        ,bottom=-tail
        ,top=tail
    )
    return outputs.reshape(1,1,shape,-1), log_det.reshape(shape,-1)

def rqs(inputs,W,H,D,inverse,left:int,right:int,bottom:int,top:int,min_W=1e-3):
    B = W.shape[-1]

    W = F.softmax(W, dim=-1)
    W = min_W + (1 - min_W * B) * W
    cumW = torch.cumsum(W,dim=-1)
    cumW = F.pad(cumW,pad=(1,0),mode='constant',value=0.0)
    cumW = (right - left) * cumW + left
    cumW[...,0]  = left
    cumW[...,-1] = right
    W = cumW[...,1:] - cumW[...,:-1]

    D = D + F.softplus(D)

    H = F.softmax(H,dim=-1)
    H = min_W + (1 - min_W * B) * H
    cumH = torch.cumsum(H,dim=-1)
    cumH = F.pad(cumH,pad=(1,0),mode='constant',value=0.0)
    cumH = (top - bottom) * cumH + bottom
    cumH[...,0] = bottom
    cumH[...,-1] = top
    H = cumH[...,1:] - cumH[...,:-1]

    if inverse:
        bin_idx = searchsorted(cumH,inputs)[...,None]
    else:
        bin_idx = searchsorted(cumW,inputs)[...,None]

    in_cumW = cumW.gather(-1,bin_idx)[...,0]
    in_cumH = cumH.gather(-1,bin_idx)[...,0]

    in_W = W.gather(-1,bin_idx)[...,0]
    in_H = H.gather(-1,bin_idx)[...,0]

    delta = H/W
    in_delta = delta.gather(-1,bin_idx)[...,0]

    in_D = D.gather(-1, bin_idx)[..., 0]
    in_D_p = D[...,1:].gather(-1,bin_idx)[...,0]

    if inverse:
        a = ((inputs - in_cumH) * (in_D + in_D_p - 2 * in_delta)) + (in_H * (in_delta - in_D))
        b = ((in_H * in_D - (inputs - in_cumH) * (in_D + in_D_p - 2 * in_delta)))
        c = -in_delta * (inputs - in_cumH)

        epsilon = (2 * c)/(-b - torch.sqrt(b.pow(2) - 4 * a * c))
        outputs = epsilon * in_W + in_cumW
        theta = epsilon * (1 - epsilon)
        alpha = in_delta.pow(2) * (in_D_p * epsilon.pow(2) + 2 * in_delta * theta + in_D * (1-epsilon).pow(2))
        beta  = (in_delta + (in_D + in_D_p - 2 * in_delta)) * theta
        log_det = torch.log(alpha) - 2 * torch.log(beta)

        return outputs, -log_det

    else:
        theta = (inputs-in_cumW)/in_W
        theta_one_minus_theta = theta * (1 - theta)

        alpha = in_H * (in_delta * theta.pow(2) + in_D * theta_one_minus_theta)
        beta  = in_delta + ((in_D + in_D_p - 2 * in_delta) * theta_one_minus_theta)
        outputs = in_cumH + (alpha/beta)
        dalpha = in_delta.pow(2) * (in_D_p * theta.pow(2) + 2  * in_delta * theta_one_minus_theta + in_D * (1-theta).pow(2))
        log_det = torch.log(dalpha) - 2 * torch.log(beta)

        return outputs, log_det
