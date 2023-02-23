from functools import partial
import torch


class ChernoffDistance:

    lamb: float
    from_logspace: bool

    def __init__(self, lamb: float, from_logspace: bool = True):
        if lamb < 0 or lamb > 1:
            raise ValueError("Lambda needs to be between 0 and 1.")
        self.lamb = lamb
        self.from_logspace = from_logspace

    def __call__(self, alpha_a: torch.Tensor, alpha_b: torch.Tensor) -> torch.Tensor:
        """

        Calculates the Chernoff distance between two batches of Dirichlet distributions, represented by their
        parameters.

        The first argument typically denotes the predictions of a model, whereas the second argument represents the
        parameters of the target. Note: if `self.from_logspace` is set, the first tensor is considered as the logarithm
        of the Dirichlet parameters, so it is exponentiated first before the distance is calculated.

        :param alpha_a: A torch tensor of the shape (batch_size, d), where d is the dimensionality of the Dirichlet
            distribution represented.
        :param alpha_b: A torch tensor of the shape (batch_size, d), where d is the dimensionality of the Dirichlet
            distribution represented.
        :return: A tensor of the shape (batch_size,) containing the computed pointwise Chernoff distances.

        """

        if self.from_logspace:
            alpha_a = torch.exp(alpha_a)

        blend = partial(torch.lerp, weight=self.lamb)

        alpha_interpolated = blend(alpha_a, alpha_b)
        alpha_a_norm = torch.sum(alpha_a, dim=-1)
        alpha_b_norm = torch.sum(alpha_b, dim=-1)
        distance = (
            torch.lgamma(torch.sum(alpha_interpolated, dim=-1))
            + blend(torch.sum(torch.lgamma(alpha_a), dim=-1), torch.sum(torch.lgamma(alpha_b), dim=-1))
            - torch.sum(torch.lgamma(alpha_interpolated), dim=-1)
            - blend(torch.lgamma(alpha_a_norm), torch.lgamma(alpha_b_norm))
        )

        return distance
