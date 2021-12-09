from .dataset import BlockworkDataset
from .generator import UnetGenerator

def test():
    l = BlockworkDataset("data/TNG100-3_SNAP099_MASS1.00e+12_5.00e+12_NGASMIN500/nvoxel_64/train", 16, 64, 8, 4)
    g = UnetGenerator(1, 1, 4).double().cuda()
    print(g(l[0][0].cuda()).shape)