import torch


class SquashedNormal(torch.distributions.TransformedDistribution):
    def __init__(self, loc, scale):
        self._loc = loc
        self.scale = scale
        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [torch.distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self):
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc
