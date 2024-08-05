import spp

class SpatialPyramidPooling(spp):

    def __init__(self, levels, mode="max"):
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out 

class TemporalPyramidPooling(spp):
    def __init__(self, levels, mode="max"):
        uper(TemporalPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.temporal_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level
        return out
