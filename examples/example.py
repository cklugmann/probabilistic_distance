import torch

from probabilistic_distance.dirichlet import BhattacharyyaDistance


def main():

    chernoff_dist = BhattacharyyaDistance(from_logspace=False)
    alpha_a = torch.tensor([
        [1.5, 17.5],
        [1.5, 14.5],
        [2.5, 5.5],
        [1.5, 5.5],
        [1.5, 1.5]
    ])
    alpha_b = torch.tensor([
        [1.5, 17.5],
        [1.5, 17.5],
        [1.5, 17.5],
        [1.5, 17.5],
        [1.5, 17.5]
    ])
    dist = chernoff_dist(alpha_a, alpha_b)
    print(dist.numpy().tolist())


if __name__ == "__main__":
    main()
