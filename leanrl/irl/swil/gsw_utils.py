from __future__ import annotations

import numpy as np
import torch
from torch import optim
from typing import Iterator, Optional, Tuple

class GSW:
    VALID_FTYPES = {"linear", "poly", "circular", "index"}

    def __init__(
        self,
        ftype: str = "poly",
        nofprojections: int = 10,
        degree: int = 2,
        radius: float = 2.0,
        use_cuda: bool = False,
    ) -> None:
        if ftype not in self.VALID_FTYPES:
            raise ValueError(f"Unsupported GSW ftype={ftype}; expected one of {sorted(self.VALID_FTYPES)}")
        self.ftype = ftype
        self.nofprojections = nofprojections
        self.degree = degree
        self.radius = radius
        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.theta = None  # Thi is for max-GSW
        self.Y = None

    def update_theta(self) -> None:
        self.theta = self.random_slice(self.theta_shape)

    def gsw(
        self, X: torch.Tensor, Y: torch.Tensor, new_theta: bool = False
    ) -> torch.Tensor:
        """
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        # print(X.shape)
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N
        self.Y = Y
        self.theta_shape = X.shape[-1]
        if self.ftype == "index":
            self.theta = torch.randint(0, X.shape[-1], (self.nofprojections,))
        else:
            if self.theta is None or new_theta:
                self.theta = self.random_slice(X.shape[-1])

        Xslices = self.get_slice(X, self.theta)
        Yslices = self.get_slice(Y, self.theta)

        self.Xslices_sorted = torch.sort(Xslices, dim=0)[0]
        self.Yslices_sorted = torch.sort(Yslices, dim=0)[0]

        return torch.sqrt(torch.sum((self.Xslices_sorted - self.Yslices_sorted) ** 2))

    def max_gsw(
        self, X: torch.Tensor, Y: torch.Tensor, iterations: int = 50, lr: float = 1e-4
    ) -> torch.Tensor:
        # N, dn = X.shape
        # M, dm = Y.shape
        dn = X.shape[-1]
        device = self.device
        # assert dn==dm and M==N
        if self.ftype == "linear":
            theta = torch.randn((1, dn), device=device, requires_grad=True)
            theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
        elif self.ftype == "poly":
            dpoly = self.homopoly(dn, self.degree)
            theta = torch.randn((1, dpoly), device=device, requires_grad=True)
            theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
        elif self.ftype == "circular":
            theta = torch.randn((1, dn), device=device, requires_grad=True)
            theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
            theta.data *= self.radius
        self.theta = theta

        optimizer = torch.optim.Adam([self.theta], lr=lr)
        total_loss = np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss = -self.gsw(X.to(self.device), Y.to(self.device))
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data /= torch.sqrt(torch.sum(self.theta.data**2))

        return self.gsw(X.to(self.device), Y.to(self.device))

    def gsl2(
        self, X: torch.Tensor, Y: torch.Tensor, theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        if theta is None:
            theta = self.random_slice(dn)

        Xslices = self.get_slice(X, theta)
        Yslices = self.get_slice(Y, theta)

        Yslices_sorted = torch.sort(Yslices, dim=0)

        return torch.sqrt(torch.sum((Xslices - Yslices) ** 2))

    def get_slice(
        self, X: torch.Tensor, theta: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Slices samples from distribution X~P_X
        Inputs:
            X:  Nxd matrix of N data samples
            theta: parameters of g (e.g., a d vector in the linear case)
        """
        if theta is None:
            theta = self.random_slice(X.shape[-1])
            self.theta = theta

        if self.ftype == "linear":
            return self.linear(X, theta)
        elif self.ftype == "poly":
            return self.poly(X, theta)
        elif self.ftype == "circular":
            return self.circular(X, theta)
        elif self.ftype == "index":
            return self.random_data_index(X, theta)
        else:
            raise Exception("Defining function not implemented")

    def random_slice(self, dim: int) -> torch.Tensor:
        if self.ftype == "linear":
            theta = torch.randn((self.nofprojections, dim))
            theta = torch.stack([th / torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype == "poly":
            dpoly = self.homopoly(dim, self.degree)
            theta = torch.randn((self.nofprojections, dpoly))
            theta = torch.stack([th / torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype == "circular":
            theta = torch.randn((self.nofprojections, dim))
            theta = torch.stack(
                [self.radius * th / torch.sqrt((th**2).sum()) for th in theta]
            )
        elif self.ftype == "index":
            theta = torch.randint(0, dim, (self.nofprojections,))
        else:
            raise ValueError(f"Unsupported GSW ftype={self.ftype}; expected one of {sorted(self.VALID_FTYPES)}")
        return theta.to(self.device)

    def random_data_index(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return X[..., theta]

    def linear(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if len(theta.shape) == 1:
            return torch.matmul(X, theta)
        else:
            return torch.matmul(X, theta.t())

    def poly(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """The polynomial defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        degree: degree of the polynomial
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        N, d = X.shape
        assert theta.shape[1] == self.homopoly(d, self.degree)
        # vectorize polynomial feature map using exponent matrix
        powers = torch.tensor(
            list(self.get_powers(d, self.degree)), device=X.device, dtype=X.dtype
        )  # [P, d]
        HX = torch.ones((N, powers.shape[0]), device=X.device, dtype=X.dtype)
        # multiply across dimensions with broadcasted powers
        for i in range(d):
            HX = HX * (X[:, i : i + 1] ** powers[:, i].unsqueeze(0))
        if len(theta.shape) == 1:
            return torch.matmul(HX, theta)
        else:
            return torch.matmul(HX, theta.t())

    def circular(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """The circular defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        """
        N, d = X.shape
        if len(theta.shape) == 1:
            return torch.sqrt(torch.sum((X - theta) ** 2, dim=1))
        else:
            return torch.stack(
                [torch.sqrt(torch.sum((X - th) ** 2, dim=1)) for th in theta], 1
            )

    def get_powers(self, dim: int, degree: int) -> Iterator[Tuple[int, ...]]:
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
                for permutation in self.get_powers(dim - 1, degree - value):
                    yield (value,) + permutation

    def homopoly(self, dim: int, degree: int) -> int:
        """
        calculates the number of elements in a homogeneous polynomial
        """
        return len(list(self.get_powers(dim, degree)))
