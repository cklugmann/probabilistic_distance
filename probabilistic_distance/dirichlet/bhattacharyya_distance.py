from probabilistic_distance.dirichlet.chernoff_distance import ChernoffDistance


class BhattacharyyaDistance(ChernoffDistance):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, lamb=0.5, **kwargs)
