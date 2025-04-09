import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0., 1.)
    # Return perturbed samples
    return x_adv

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # x_nat is the natural (clean) data batch
    x_nat = dat.clone().detach()
    
    # Initialize x_adv
    if rand_start:
        # Random initialization within epsilon ball
        x_adv = x_nat + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    else:
        # Start from clean image
        x_adv = x_nat.clone().detach()
    
    # Make sure we're in [0,1] bounds
    x_adv = torch.clamp(x_adv, 0., 1.)
    
    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data
        data_grad = gradient_wrt_data(model, device, x_adv, lbl)
        
        # Perturb the image using the gradient
        x_adv = x_adv + alpha * data_grad.sign()
        
        # Project back to epsilon ball around x_nat
        delta = torch.clamp(x_adv - x_nat, -eps, eps)
        x_adv = x_nat + delta
        
        # Make sure we're in [0,1] bounds
        x_adv = torch.clamp(x_adv, 0., 1.)
    
    return x_adv


def FGSM_attack(model, device, dat, lbl, eps):
    return PGD_attack(model, device, dat, lbl, eps, eps, 1, False)


def rFGSM_attack(model, device, dat, lbl, eps):
    return PGD_attack(model, device, dat, lbl, eps, eps, 1, True)


def FGM_L2_attack(model, device, dat, lbl, eps):
    x_nat = dat.clone().detach()
    x_nat.requires_grad = True

    output = model(x_nat)
    loss = F.cross_entropy(output, lbl)
    model.zero_grad()
    loss.backward()

    grad = x_nat.grad.data

    # Flatten grad per sample and compute L2 norm
    grad_view = grad.view(grad.shape[0], -1)
    grad_norm = torch.norm(grad_view, p=2, dim=1).view(-1, 1, 1, 1)
    grad_norm = torch.clamp(grad_norm, min=1e-12)  # Prevent division by zero

    perturbation = eps * grad / grad_norm
    x_adv = x_nat + perturbation
    x_adv = torch.clamp(x_adv, 0., 1.)

    return x_adv

