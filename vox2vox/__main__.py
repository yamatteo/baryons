import argparse
import itertools
import logging
import os
import pprint
import random
import shutil
import time
from pathlib import Path

import torch.cuda
from adabelief_pytorch import AdaBelief
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from options import opts
from preprocessing import preprocessing_is_complete, preprocess
from vox2vox import init, train, get_hash, evaluate, apply, Vox2Vox, Dataset, right_to, write_images
from vox2vox.models import X4Compressor, LinearConvolution, UNet

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--train", nargs='?', type=bool, const=True, default=False)
    # parser.add_argument("-r", "--root",              type=str,   default='/gpfswork/rech/qbf/uiu95bi/predicting_baryons/data/TNG300-1_MASS_1.00e+12_5.00e+12_MSUN/nvoxel_256/3d_on_the_fly/',   help="folder where data is")
    # parser.add_argument("--n_voxel",                 type=int,   default=256,    help="number of voxels set for images")
    # parser.add_argument("--epoch",                   type=int,   default=0,      help="epoch to start training from")
    # parser.add_argument("-n", "--n_epochs", type=int)
    # parser.add_argument("--console_log_level", type=str, default="info")
    parser.add_argument("--clear_preprocessing", nargs='?', type=bool, const=True, default=False)
    parser.add_argument("--do_preprocessing", nargs='?', type=bool, const=True, default=False)
    parser.add_argument("--clear_runs", nargs='?', type=bool, const=True, default=False)
    # parser.add_argument("--delete_previous_runs", type=bool, default=False)
    # parser.add_argument("--preprocessing_fixed_size", type=eval, default=None)
    # parser.add_argument("-b", "--batch_size", type=eval)
    # parser.add_argument("--lr",                      type=float, default=0.0002, help="adam: learning rate")
    # parser.add_argument("--b1",                      type=float, default=0.5,    help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--b2",                      type=float, default=0.999,  help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--decay_epoch",             type=int,   default=10,     help="epoch from which to start lr decay")
    # parser.add_argument("--n_cpu", type=int, help="cpus to use in batch generation")
    # parser.add_argument("--img_height",              type=int,   default=256,    help="size of image height")
    # parser.add_argument("--img_width",               type=int,   default=256,    help="size of image width")
    # parser.add_argument("--channels",                type=int,   default=1,      help="number of image channels")
    # parser.add_argument("-s", "--sample_interval",   type=int,   default=None,   help="interval between sampling of images from generators")
    # parser.add_argument("--checkpoint_interval",     type=int,   default=50,     help="interval between model checkpoints")
    # parser.add_argument("-e", "--evaluate", dest='evaluate', action='store_true', help="evaluates model if present")
    # parser.add_argument("--use_adabelief", dest='use_adabelief', action='store_true', help="whether to use AdaBelief instead of Adam")
    opts.update(vars(parser.parse_args()))

    logger.info(f'Found GPU: {opts["cuda"]}')

    # Preprocessing operations
    if opts["clear_preprocessing"]:
        shutil.rmtree(Path(opts["preprocessing_path"]) / opts["preprocessing_name"], ignore_errors=True)
    if opts["do_preprocessing"]:
        preprocess(
            source_path=Path(opts["simulation_base_path"]) / opts['sim_name'] / "output",
            target_path=Path(opts["preprocessing_path"]) / opts["preprocessing_name"],
            sim_name=opts['sim_name'],
            snap_num=opts['snap_num'],
            mass_min=opts['mass_min'],
            mass_max=opts['mass_max'],
            nvoxel=opts['nvoxel'],
            n_gas_min=opts['n_gas_min'],
        )
    if not preprocessing_is_complete(
            target_path=Path(opts["preprocessing_path"]) / opts["preprocessing_name"],
            nvoxel=opts['nvoxel'],
            fixed_size=opts.get("fixed_size", None),
    ):
        raise ValueError("Preprocessing is not complete!")

    if opts["clear_runs"]:
        shutil.rmtree(Path(opts["output_path"]) / opts["preprocessing_name"], ignore_errors=True)

    opts["now"] = str(int(time.time()))

    # Training
    if opts["train"]:

        writer = SummaryWriter(
            Path(opts["output_path"]) / opts["preprocessing_name"] / opts["now"]
        )

        # dataloader = torch.utils.data.DataLoader(
        #     Dataset(
        #         path=(
        #                 Path(opts["preprocessing_path"])
        #                 / opts["preprocessing_name"]
        #                 / f"nvoxel_{opts['nvoxel']}"
        #                 / "train"
        #         ),
        #         # blur=True,
        #         # crop=(opts["nvoxel"], 16),
        #     ),
        #     batch_size=opts["batch_size"],
        #     shuffle=True,
        #     num_workers=opts["n_cpu"],
        #     drop_last=True,
        # )
        # valid_dl = torch.utils.data.DataLoader(
        #     Dataset(
        #         path=(
        #                 Path(opts["preprocessing_path"])
        #                 / opts["preprocessing_name"]
        #                 / f"nvoxel_{opts['nvoxel']}"
        #                 / "valid"
        #         ),
        #         # blur=True,
        #         # crop=(opts["nvoxel"], 16),
        #     ),
        #     batch_size=opts["batch_size"]//2,
        #     shuffle=True,
        #     num_workers=opts["n_cpu"],
        #     drop_last=True,
        # )
        # g = right_to(LinearConvolution())
        # torch.nn.init.constant_(g.model.bias, 0)
        # torch.nn.init.constant_(g.model.weight, 0.078)
        # g_opt = AdaBelief(
        #     g.parameters(),
        #     lr=1e-3,
        #     eps=1e-8,
        #     betas=(0.9, 0.999),
        #     weight_decouple=True,
        #     rectify=True,
        #     print_change_log=False,
        # )

        g2 = right_to(UNet())
        g2.merger.weight.requires_grad = False
        torch.nn.init.constant_(g2.merger.weight, 1)

        # Second cycle of training
        # g2.merger.weight.data[0, 1, 0, 0, 0] = 1
        # for param in g2.first.parameters():
        #     param.requires_grad = False
        # for param in g2.second.parameters():
        #     param.requires_grad = True
        #     torch.nn.init.normal_(param, 0, 0.1)
        opt = AdaBelief(
            g2.parameters(),
            lr=1e-2,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=True,
            print_change_log=False,
        )
        dataloader = torch.utils.data.DataLoader(
            Dataset(
                path=(
                        Path(opts["preprocessing_path"])
                        / opts["preprocessing_name"]
                        / f"nvoxel_{opts['nvoxel']}"
                        / "train"
                ),
                # blur=True,
                # crop=(opts["nvoxel"], 28),
            ),
            batch_size=opts["batch_size"],
            shuffle=True,
            num_workers=opts["n_cpu"],
            drop_last=True,
        )
        dm, rg = [right_to(x) for x in next(iter(dataloader))]
        dm_p1 = functional.pad(dm, (1,)*6, value=0)
        dm_p3 = functional.pad(dm, (3,)*6, value=0)
        for epoch in range(opts["n_epochs"]):
            g2.zero_grad()
            pg = g2(dm, dm_p1, dm_p3)
            loss = functional.l1_loss(pg, rg)
            loss.backward()
            opt.step()
            if (epoch+1) % 10 == 0:
                writer.add_scalar(
                    "global",
                    loss.item(),
                    global_step=epoch+1,
                )

        # # First cycle of training
        # g2.merger.weight.data[0, 0, 0, 0, 0] = 1
        # for param in g2.second.parameters():
        #     param.requires_grad = False
        #     # torch.nn.init.constant_(param, 0)
        # for param in g2.first.parameters():
        #     param.requires_grad = True
        #     torch.nn.init.normal_(param, 0, 0.1)
        # opt = AdaBelief(
        #     g2.first.parameters(),
        #     lr=1e-2,
        #     eps=1e-8,
        #     betas=(0.9, 0.999),
        #     weight_decouple=True,
        #     rectify=True,
        #     print_change_log=False,
        # )
        # dataloader = torch.utils.data.DataLoader(
        #     Dataset(
        #         path=(
        #                 Path(opts["preprocessing_path"])
        #                 / opts["preprocessing_name"]
        #                 / f"nvoxel_{opts['nvoxel']}"
        #                 / "train"
        #         ),
        #         nvoxel=opts["nvoxel"],
        #         batch_size=opts["batch_size"],
        #         # fold=2,
        #         # blur=True,
        #         # crop=(opts["nvoxel"], 32),
        #     ),
        #     batch_size=opts["batch_size"],
        #     shuffle=True,
        #     num_workers=opts["n_cpu"],
        #     drop_last=True,
        # )
        # dm, rg = [right_to(x) for x in next(iter(dataloader))]
        # dm_p1 = functional.pad(dm, (1,)*6, value=0)
        # dm_p3 = functional.pad(dm, (3,)*6, value=0)
        for epoch in range(opts["n_epochs"]):
            g2.zero_grad()
            pg = g2(dm_p1, dm_p3)
            loss = functional.l1_loss(pg, rg)
            loss.backward()
            opt.step()
            if (epoch+1) % 10 == 0:
                writer.add_scalar(
                    "first",
                    loss.item(),
                    global_step=epoch+1,
                )

        assert False
        # valid_dm, valid_rg = [right_to(x) for x in next(iter(valid_dl))]
        # for epoch in range(opts["n_epochs"]):
        #     # for i, [dm, rg] in enumerate(dataloader):
        #     # dm, rg = right_to(dm), right_to(rg)
        #
        #     # g_opt.zero_grad()
        #     # pg = g(dm)
        #     # degdm, degrg, degpg = v2v.deg_step(dm, rg, opts)
        #     # loss = functional.l1_loss(pg, rg)
        #     # loss.backward()
        #     # g_opt.step()
        #
        #     g2.zero_grad()
        #     pg = g2(dm)
        #     # degdm, degrg, degpg = v2v.deg_step(dm, rg, opts)
        #     loss = functional.l1_loss(pg, rg)
        #     loss.backward()
        #     opt.step()
        #
        #     # patch_generator_optimizer.zero_grad()
        #     # badpg = functional.interpolate(degpg.detach(), size=opts["nvoxel"], mode="nearest")
        #     # rg_crop = rg[:, :, cx:cx + 16, cy:cy + 16, cz:cz + 16]
        #     # badpg_crop = badpg[:, :, cx:cx + 16, cy:cy + 16, cz:cz + 16]
        #     # betterpg_crop = v2v.patch_step(dm_crop, rg_crop, badpg_crop, opts=opts)
        #     # croploss = main_trainer(dm_crop, rg_crop, betterpg_crop)
        #     # croploss.backward()
        #     # patch_generator_optimizer.step()
        #
        #     # valid_dm, valid_rg = [ right_to(x) for x in next(iter(valid_dl))]
        #     # valid_loss = functional.l1_loss(g(valid_dm), valid_rg)
        #     # writer.add_scalars(
        #     #     "losses",
        #     #     {"train":loss.item(), "valid":valid_loss.item()},
        #     #     global_step=epoch,
        #     # )
        #     if (epoch+1) % 10 == 0:
        #         scalars = {}
        #         for _model, model in (("g2", g2), ):
        #             # for _model, model in (("g", g), ("g2", g2)):
        #             for _dl, (xx, yy) in (("train", (dm, rg)), ):
        #                 # for _dl, (xx, yy) in (("train", (dm, rg)), ("valid", (valid_dm, valid_rg))):
        #                 with torch.no_grad():
        #                     # dm, rg = [right_to(x) for x in next(iter(dl))]
        #                     pg = model(xx)
        #                     scalars[f"{_model}_{_dl}"] = functional.l1_loss(pg, yy).item()
        #             i = random.randint(0, opts["batch_size"]//2-2)
        #         write_images(f"{_model}:{epoch+1}", [[xx[i:i+1, :, :, :, :]], [pg[i:i+1, :, :, :, :], yy[i:i+1, :, :, :, :]]], writer, global_step=epoch+1)
        #         writer.add_scalars(
        #             "losses",
        #             scalars,
        #             global_step=epoch+1,
        #         )
        #         # named_params = {f"{name}.{i}":value for name, param in g2.named_parameters() for i, value in enumerate(param.detach().cpu().numpy().flatten().tolist()) }
        #         #
        #         # writer.add_scalars(
        #         #     "parameters",
        #         #     {name: item for name, item in named_params.items()},
        #         #     global_step=epoch,
        #         # )
        #
        #         # torch.save(
        #         #     g.state_dict(),
        #         #     Path(opts["output_path"]) / opts["preprocessing_name"] / opts["now"] / f"g.pth",
        #         # )

        # writer.add_scalar("croploss", croploss.item(), global_step=epoch)

        # v2v.save()

    # Setup eventual multiple runs
    # simple_opts = {
    #     key: value
    #     for (key, value) in opts.items()
    #     if isinstance(value, (int, float, str, tuple))
    # }
    #
    # multi_opts = {
    #     key: value
    #     for (key, value) in opts.items()
    #     if isinstance(value, list)
    # }
    #
    # combinations = list(itertools.product(*multi_opts.values()))
    # multi_labels = tuple(multi_opts.keys())
    # command = opts["command"]
    #
    # if len(multi_opts) == 0:
    #     logger.info(
    #         f" Launching single action with options:\n {pprint.pformat({key: value for (key, value) in opts.items()}, indent=4)}")
    #     model_id = get_hash(opts)
    #     if command == "create":
    #         if os.path.exists(Path(opts["models_base_path"]) / model_id):
    #             logger.warning(f"Aborting creation of {model_id}: it already exists.")
    #         else:
    #             os.mkdir(Path(opts["models_base_path"]) / model_id)
    #             init(model_id, opts)
    #             logger.info(f"Model {model_id} created and initialized.")
    #     elif command == "reset":
    #         if os.path.exists(Path(opts["models_base_path"]) / model_id):
    #             init(model_id, opts)
    #             logger.info(f"Model {model_id} reset to initial state.")
    #         else:
    #             logger.info(f"Model {model_id} does not exists.")
    #     elif command == "remove":
    #         if os.path.exists(Path(opts["models_base_path"]) / model_id):
    #             shutil.rmtree(Path(opts["models_base_path"]) / model_id, ignore_errors=True)
    #             logger.info(f"Model {model_id} removed.")
    #         else:
    #             logger.info(f"Model {model_id} does not exists.")
    #     elif command == "clear_reports":
    #         if os.path.exists(Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id):
    #             shutil.rmtree(Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id, ignore_errors=True)
    #             logger.info(f"Reports of {model_id} cleared.")
    #         else:
    #             logger.info(f"No reports for {model_id}.")
    #     elif command == "train":
    #         if os.path.exists(Path(opts["models_base_path"]) / model_id):
    #             train(model_id, opts)
    #             logger.info(f"Model {model_id} trained for {opts['n_epochs']} epochs.")
    #         else:
    #             logger.info(f"Model {model_id} does not exists.")
    #     elif command == "evaluate":
    #         if os.path.exists(Path(opts["models_base_path"]) / model_id):
    #             output = evaluate(model_id, opts)
    #             logger.info(f"Evaluated a batch of {opts['batch_size']} halos.")
    #             for label, std_mean in output.items():
    #                 logger.info(f"  {label}: {std_mean[1].item():.3e} ±{std_mean[0].item():.0e}")
    #         else:
    #             logger.info(f"Model {model_id} does not exists.")
    #     elif command == "apply":
    #         if os.path.exists(Path(opts["models_base_path"]) / model_id):
    #             apply(model_id, opts)
    #             logger.info(f"Applied {model_id}")
    #         else:
    #             logger.info(f"Model {model_id} does not exists.")
    #     # roundrun(opts["rounds"], opts, "opts")
    # else:
    #     logger.info(
    #         f"Launching multiple actions with base options:\n{pprint.pformat({key: value for (key, value) in simple_opts.items()}, indent=4)}")
    #     for i, possible_opt in enumerate(combinations):
    #         logger.info(
    #             f" Launching run {i + 1}/{len(combinations)} with options:\n {pprint.pformat({key: value for (key, value) in zip(multi_labels, possible_opt)}, indent=4)}"
    #         )
    #         extra_opt = dict(zip(multi_labels, possible_opt))
    #         run_opts = dict(simple_opts, **extra_opt, run_index=i)
    #         model_id = get_hash(run_opts)
    #         if command == "create":
    #             if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
    #                 logger.warning(f"Aborting creation of {model_id}: it already exists.")
    #             else:
    #                 os.mkdir(Path(run_opts["models_base_path"]) / model_id)
    #                 init(model_id, run_opts)
    #                 logger.info(f"Model {model_id} created and initialized.")
    #         elif command == "reset":
    #             if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
    #                 init(model_id, run_opts)
    #                 logger.info(f"Model {model_id} reset to initial state.")
    #             else:
    #                 logger.info(f"Model {model_id} does not exists.")
    #         elif command == "remove":
    #             if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
    #                 shutil.rmtree(Path(run_opts["models_base_path"]) / model_id, ignore_errors=True)
    #                 logger.info(f"Model {model_id} removed.")
    #             else:
    #                 logger.info(f"Model {model_id} does not exists.")
    #         elif command == "clear_reports":
    #             if os.path.exists(Path(run_opts["run_base_path"]) / run_opts["restricted_sim_name"] / model_id):
    #                 shutil.rmtree(Path(run_opts["run_base_path"]) / run_opts["restricted_sim_name"] / model_id, ignore_errors=True)
    #                 logger.info(f"Reports of {model_id} cleared.")
    #             else:
    #                 logger.info(f"No reports for {model_id}.")
    #         elif command == "train":
    #             if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
    #                 train(model_id, run_opts)
    #                 logger.info(f"Model {model_id} trained for {run_opts['n_epochs']} epochs.")
    #             else:
    #                 logger.info(f"Model {model_id} does not exists.")
    #         elif command == "evaluate":
    #             if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
    #                 output = evaluate(model_id, run_opts)
    #                 logger.info(f"Evaluated a batch of {run_opts['batch_size']} halos.")
    #                 for label, std_mean in output.items():
    #                     logger.info(f"  {label}: {std_mean[1].item():.3e} ±{std_mean[0].item():.0e}")
    #             else:
    #                 logger.info(f"Model {model_id} does not exists.")
    #         elif command == "apply":
    #             if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
    #                 apply(model_id, run_opts)
    #                 logger.info(f"Applied {model_id}")
    #             else:
    #                 logger.info(f"Model {model_id} does not exists.")
    #         # roundrun(run_opts["rounds"], run_run_opts, str("~").join([k + str(o) for k, o in extra_opt.items()]))
    # # print(opts)
