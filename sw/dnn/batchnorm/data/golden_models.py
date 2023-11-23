import torch


def golden_model_forward_eval(
    ifmap, eps, running_mean, running_var, weight, bias, dtype
) -> torch.Tensor:
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    bn.running_mean = running_mean
    bn.running_var = running_var
    bn.eval()
    return bn(ifmap)


def golden_model_backward(
    ifmap, grad_ofmap, weight, bias, running_mean, running_var, eps, dtype
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    bn.running_mean = running_mean.clone()
    bn.running_var = running_var.clone()
    bn.eval()
    ofmap = bn(ifmap)
    ofmap.retain_grad()
    ofmap.flatten().dot(grad_ofmap.flatten()).backward()
    return ifmap.grad, bn.weight.grad, bn.bias.grad


def golden_model_backward_training(
    ifmap, grad_ofmap, weight, bias, eps, dtype
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    ofmap = bn(ifmap)
    ofmap.retain_grad()
    ofmap.flatten().dot(grad_ofmap.flatten()).backward()
    return ifmap.grad, bn.weight.grad, bn.bias.grad


# Implementation of backprop for batchnorm training mode without autograd
def my_golden_model_backward_training(
    ifmap, grad_ofmap, weight, bias, current_mean, current_var, eps, dtype
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    num_points = n * ih * iw
    invstd = torch.rsqrt(current_var + eps)
    sum = grad_ofmap.sum(dim=(0, 2, 3))
    dotp = torch.zeros(ci)
    for c in range(ci):
        dotp[c] = torch.sum(
            grad_ofmap[:, c, :, :] * ((ifmap[:, c, :, :] - current_mean[c]))
        )
    k = dotp * invstd * invstd / num_points
    grad_mean = sum / num_points
    dx = torch.zeros(ifmap.shape)
    for c in range(ci):
        dx[:, c, :, :] = (ifmap[:, c, :, :] - current_mean[c]) * k[c]
    grad_ifmap = torch.zeros(ifmap.shape)
    for c in range(ci):
        grad_ifmap[:, c, :, :] = (
            (grad_ofmap[:, c, :, :] - grad_mean[c] - dx[:, c, :, :])
            * invstd[c]
            * weight[c]
        )
    grad_weight = dotp * invstd
    grad_bias = sum
    return grad_ifmap, grad_weight, grad_bias
