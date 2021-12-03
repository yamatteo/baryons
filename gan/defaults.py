from argparse import Namespace

defaults = Namespace(
    b1=0.5,
    b2=0.999,
    batch_size=2,
    channels=1,
    criterion_gan="mse",
    criterion_pixelwise="l1",
    decay_epoch=100,
    generator_depth=5,
    lambda_pixel=100,
    lr=0.0002,
    mass_range="MASS_1.00e+12_5.00e+12_MSUN",
    n_epochs=5,
    n_voxel=128,
    num_filters=4,
    patch_side=16,
    sim_name="TNG300-1",
)