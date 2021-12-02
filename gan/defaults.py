from types import SimpleNamespace

defaults = SimpleNamespace(
    b1=0.5,
    b2=0.999,
    batch_size=2,
    channels=1,
    checkpoint_interval=-1,
    criterion_gan="mse",
    criterion_pixelwise="l1",
    decay_epoch=100,
    generator_depth=5,
    lambda_pixel=100,
    lr=0.0002,
    mass_range="MASS_1.00e+12_5.00e+12_MSUN",
    n_cpu=8,
    n_epochs=5,
    n_voxel=128,
    num_filters=4,
    patch_side=16,
    sample_interval=5,
    sim_name="TNG300-1",
    skip_to_epoch=0,
)