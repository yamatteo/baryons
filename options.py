import torch
from dotenv import dotenv_values

opts = {
    "cuda": torch.cuda.is_available()
}

opts.update(**dotenv_values(".env"))

opts.update(dict(
    batch_size=40,
    checkpoint_interval=10,
    n_cpu=8,
    n_epochs=500,
    sample_interval=10,

    sim_name="TNG100-3",
    snap_num=99,
    mass_min=1e12,
    mass_max=1e20,
    # n_gas_min=6000,  # Calculated to be nvoxel^3

    b1=0.9,
    b2=0.999,
    channels=1,
    criterion_gan="mse",
    criterion_pixelwise="l1",
    decay_epoch=100,
    mod_main_generator="sunet",
    mod_main_features=16,
    mod_main_levels=4,
    mod_patch_generator="sunet",
    mod_patch_features=16,
    mod_patch_levels=4,
    trainer="l1",
    lambda_pixel=100,
    lr=0.0002,
    num_filters=4,
    nvoxel=64,
    patch_side=8,
    metrics=("mse", "l1", "totalmass"),
))

opts["n_gas_min"] = int(0.1 * opts["nvoxel"] ** 3)  # Probability less than 0.1 that a voxel has a gas particle

opts["preprocessing_name"] = (
        f"{opts['sim_name']}"
        + f"_SNAP{opts['snap_num']:03d}"
        + f"_MASS{opts['mass_min']:.2e}"
        + f"_{opts['mass_max']:.2e}"
        + f"_NGASMIN{opts['n_gas_min']}"
)
