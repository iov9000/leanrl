import numpy as np

import torch
from torch import optim


def gsw(X, Y, theta=None, ftype="linear", degree=5, radius=1, n_proj=10):
    N, dn = X.shape
    M, dm = Y.shape
    assert dn == dm and M == N
    if theta is None:
        theta = random_slice(dn, n_proj, ftype, degree, radius)

    Xslices = get_slice(X, theta, ftype, degree, radius)
    Yslices = get_slice(Y, theta, ftype, degree, radius)

    Xslices_sorted = torch.sort(Xslices, dim=0)[0]
    Yslices_sorted = torch.sort(Yslices, dim=0)[0]
    return (
        torch.sqrt(torch.sum((Xslices_sorted - Yslices_sorted) ** 2)),
        theta,
        Xslices_sorted,
        Yslices_sorted,
    )


def max_gsw(X, Y, n_iter=50, lr=1e-4, ftype="linear", degree=5, radius=1):
    N, dn = X.shape
    M, dm = Y.shape
    assert dn == dm and M == N
    if ftype == "linear":
        theta = torch.randn((1, dn), requires_grad=True)
        theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
    elif ftype == "poly":
        dpoly = homopoly(dn, degree)
        theta = torch.randn((1, dpoly), requires_grad=True)
        theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
    elif ftype == "circular":
        theta = torch.randn((1, dn), requires_grad=True)
        theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
        theta.data *= radius

    optimizer = optim.Adam([theta], lr=lr)
    total_loss = np.zeros((n_iter,))
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -gsw(X, Y, theta, ftype, degree, radius)
        total_loss[i] = loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        theta.data /= torch.sqrt(torch.sum(theta.data**2))

    return gsw(X, Y, theta, ftype, degree, radius)


def get_slice(X, theta, ftype="linear", degree=5, radius=1):
    if ftype == "linear":
        return linear(X, theta)
    elif ftype == "poly":
        return poly(X, theta, degree)
    elif ftype == "circular":
        return circular(X, theta, radius)
    else:
        raise Exception("Defining function not implemented")


def random_slice(dim, n_proj=10, ftype="linear", degree=5, radius=1):
    if ftype == "linear":
        theta = torch.randn((n_proj, dim))
        theta = theta / torch.norm(theta, dim=1, keepdim=True)
    elif ftype == "poly":
        dpoly = homopoly(dim, degree)
        theta = torch.randn((n_proj, dpoly))
        theta = theta / torch.norm(theta, dim=1, keepdim=True)
    elif ftype == "circular":
        theta = torch.randn((n_proj, dim))
        theta = radius * theta / torch.norm(theta, dim=1, keepdim=True)
    return theta


def linear(X, theta):
    if len(theta.shape) == 1:
        return torch.matmul(X, theta)
    else:
        return torch.matmul(X, theta.t())


def poly(X, theta, degree):
    """The polynomial defining function for generalized Radon transform
    Inputs
    X:  Nxd matrix of N data samples
    theta: Lxd vector that parameterizes for L projections
    degree: degree of the polynomial
    """
    N, d = X.shape
    assert theta.shape[1] == homopoly(d, degree)
    # vectorized polynomial feature map
    device = X.device if hasattr(X, "device") else None
    dtype = X.dtype
    powers = torch.tensor(list(get_powers(d, degree)), device=device, dtype=dtype)
    HX = torch.ones((N, powers.shape[0]), device=device, dtype=dtype)
    for i in range(d):
        HX = HX * (X[:, i : i + 1] ** powers[:, i].unsqueeze(0))
    if len(theta.shape) == 1:
        return torch.matmul(HX, theta)
    else:
        return torch.matmul(HX, theta.t())


def circular(X, theta, radius):
    """The circular defining function for generalized Radon transform
    Inputs
    X:  Nxd matrix of N data samples
    theta: Lxd vector that parameterizes for L projections
    """
    N, d = X.shape
    if len(theta.shape) == 1:
        return torch.sqrt(torch.sum((X - theta) ** 2, dim=1))
    else:
        # compute pairwise distances between X [N,d] and centers theta [L,d]
        # result shape [N, L]
        diff = X.unsqueeze(1) - theta.unsqueeze(0)
        return torch.sqrt(torch.sum(diff * diff, dim=2))


def get_powers(dim, degree):
    """
    This function calculates the powers of a homogeneous polynomial
    e.g.

    list(get_powers(dim=2,degree=3))
    [(0, 3), (1, 2), (2, 1), (3, 0)]

    list(get_powers(dim=3,degree=2))
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    """
    if dim == 1:
        yield (degree,)
    else:
        for value in range(degree + 1):
            for permutation in get_powers(dim - 1, degree - value):
                yield (value,) + permutation


def homopoly(dim, degree):
    """
    calculates the number of elements in a homogeneous polynomial
    """
    return len(list(get_powers(dim, degree)))
