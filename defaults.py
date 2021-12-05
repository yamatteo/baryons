from argparse import Namespace

runtime_options = Namespace(
    batch_size=4,
    checkpoint_interval=-1,
    data_path="data",
    dataset_path="dataset",
    log_level="debug",
    log_mode="w",
    n_cpu=2,
    n_epochs=5,
    root=".",
    sample_interval=5,
    skip_to_epoch=0,
)

simulation_options = Namespace(
    sim_name="TNG100-3",
    snap_num=99,
    mass_min=1e12,
    mass_max=5e12,
    n_gas_min=500,
)

network_options = Namespace(
    b1=0.9,
    b2=0.999,
    channels=1,
    criterion_gan="mse",
    criterion_pixelwise="l1",
    decay_epoch=100,
    generator_depth=4,
    lambda_pixel=100,
    lr=0.0002,
    num_filters=4,
    nvoxel=128,
    patch_side=16,
)

opts = {
    **vars(runtime_options),
    **vars(simulation_options),
    **vars(network_options),
}