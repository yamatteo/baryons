from dotenv import dotenv_values

opts = dotenv_values(".env")

opts.update(dict(
    batch_size=40,
    checkpoint_interval=10,
    voxels_base_path="data",
    simulation_base_path="dataset",
    models_base_path="saved_models",
    log_level="info",
    log_mode="w",
    n_cpu=8,
    n_epochs=1000,
    run_base_path="runs",
    sample_interval=10,
    may_resume=True,
    train_mode="train",

    sim_name="TNG100-3",
    snap_num=99,
    mass_min=10e12,
    mass_max=200e12,
    n_gas_min=6000,

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

opts["preprocessing_name"] = (
    f"{opts['sim_name']}"
    + f"_SNAP{opts['snap_num']:03d}"
    + f"_MASS{opts['mass_min']:.2e}"
    + f"_{opts['mass_max']:.2e}"
    + f"_NGASMIN{opts['n_gas_min']}"
)