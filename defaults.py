from argparse import Namespace

runtime_options = dict(
    batch_size=64,
    checkpoint_interval=10,
    data_path=None,
    dataset_path=None,
    log_level="info",
    log_mode="w",
    n_cpu=2,
    n_epochs=100,
    output_path=None,
    sample_interval=10,
    may_resume=True,
    train_mode="train",
)

simulation_options = dict(
    sim_name="TNG300-1",
    snap_num=99,
    mass_min=1e12,
    mass_max=5e12,
    n_gas_min=500,
)

network_options = dict(
    b1=0.9,
    b2=0.999,
    channels=1,
    criterion_gan="mse",
    criterion_pixelwise="l1",
    decay_epoch=100,
    generator=["flat", "original"],
    generator_depth=4,
    discriminator=["multimse", "original"],
    lambda_pixel=100,
    lr=0.0002,
    num_filters=4,
    nvoxel=64,
    patch_side=8,
    metrics=("mse", "l1"),
)

opts = {
    **runtime_options,
    **simulation_options,
    **network_options,
}