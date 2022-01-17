import datetime
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from preprocessing import preprocess
from .trainers import ClassifierTrainer, QuantifierTrainer, MseTrainer, L1Trainer
from .dataset import BasicDataset
from .generators import UnetGenerator, SUNet
from .metrics import metrics_dict

logger = logging.getLogger("baryons")


class Vox2Vox:
    def __init__(
            self,
            run_base_path,
            generator,
            lr,
            trainer,
            sim_name,
            snap_num=None,
            mass_min=None,
            mass_max=None,
            n_gas_min=None,
            voxels_base_path=None,
            nvoxel=None,
            n_cpu=None,
            dataset_fixed_size=None,
            batch_size=None,
            id=None,
            **opts,
    ):
        self.cuda = cuda = torch.cuda.is_available()
        if not cuda:
            logger.warning("Running without cuda!")
        self.run_path = Path(run_base_path)

        def tensor_type(x):
            if isinstance(x, (list, tuple)):
                return torch.Tensor(x).to(
                    device="cuda" if cuda else "cpu", dtype=torch.float
                )
            elif isinstance(x, (torch.Tensor, torch.nn.Module)):
                return x.to(device="cuda" if cuda else "cpu", dtype=torch.float)
            else:
                return x

        self.tensor_type = tensor_type

        voxelization_name = f"{sim_name}_SNAP{snap_num:03d}_MASS{mass_min:.2e}_{mass_max:.2e}_NGASMIN{n_gas_min}"
        # if generator == "lunet":
        #     self.dataloader = torch.utils.data.DataLoader(
        #         LogarithmicDataset(
        #             path=(
        #                     Path(voxels_base_path)
        #                     / voxelization_name
        #                     / f"nvoxel_{nvoxel}"
        #                     / "train"
        #             ),
        #             fixed_size=dataset_fixed_size,
        #         ),
        #         batch_size=batch_size,
        #         shuffle=True,
        #         num_workers=n_cpu,
        #         drop_last=True,
        #     )
        #     self.valid_dataset = LogarithmicDataset(
        #         path=(
        #                 Path(voxels_base_path)
        #                 / voxelization_name
        #                 / f"nvoxel_{nvoxel}"
        #                 / "valid"
        #         ),
        #         fixed_size=dataset_fixed_size,
        #     )
        # else:
        self.dataloader = torch.utils.data.DataLoader(
            BasicDataset(
                path=(
                        Path(voxels_base_path)
                        / voxelization_name
                        / f"nvoxel_{nvoxel}"
                        / "train"
                ),
                fixed_size=dataset_fixed_size,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_cpu,
            drop_last=True,
        )
        self.valid_dataset = BasicDataset(
            path=(
                    Path(voxels_base_path)
                    / voxelization_name
                    / f"nvoxel_{nvoxel}"
                    / "valid"
            ),
            fixed_size=dataset_fixed_size,
        )

        if generator == "unet114":
            self.generator = self.tensor_type(
                UnetGenerator(
                    1, 1, num_downs=4, use_dropout=True, norm_layer=nn.InstanceNorm3d
                )
            )
        elif generator == "sunet":
            self.generator = self.tensor_type(
                SUNet(
                    features=opts["sunet_feat"],
                    levels=opts["sunet_levels"],
                )
            )
        # elif generator == "lunet":
        #     self.generator = self.tensor_type(
        #         LUNet(
        #             features=opts["lunet_feat"],
        #             levels=opts["lunet_levels"],
        #         )
        #     )
        # elif generator == "alunet":
        #     self.generator = self.tensor_type(
        #         AluNet(
        #             features=opts["lunet_feat"],
        #             levels=opts["lunet_levels"],
        #         )
        #     )
        if opts['optimizer'] == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.g_optimizer = AdaBelief(
                self.generator.parameters(),
                lr=lr,
                eps=1e-8,
                betas=(0.9, 0.999),
                weight_decouple=True,
                rectify=True,
            )
        else:
            self.g_optimizer = torch.optim.SGD(
                self.generator.parameters(),
                lr=lr,
            )

        if trainer == "classifier":
            self.trainer = ClassifierTrainer(cuda, batch_size, lr)
        elif trainer == "quantifier":
            self.trainer = QuantifierTrainer(cuda, batch_size, lr)
        elif trainer == "mse":
            self.trainer = MseTrainer()
        elif trainer == "l1":
            self.trainer = L1Trainer()

        self.writer = SummaryWriter(
            Path(run_base_path) / voxelization_name / (id if id is not None else "last_run")
        )

    # def save_checkpoint(self, epoch):
    #     torch.save(
    #         self.generator.state_dict(),
    #         self.run_path / f"generator_{epoch:03d}.pth",
    #     )
    #     torch.save(
    #         self.classifier.state_dict(),
    #         self.run_path / f"classifier_{epoch:03d}.pth",
    #     )

    # def load_checkpoint(self, epoch):
    #     logger.info(f"Loading pretrained models from epoch {epoch}")
    #     self.generator.load_state_dict(torch.load(self.run_path / f"generator_{epoch:03d}.pth"))
    #     self.classifier.load_state_dict(torch.load(self.run_path / f"classifier_{epoch:03d}.pth"))

    # def resume(self):
    #     try:
    #         epoch = max(
    #             [
    #                 int(
    #                     str(s)
    #                         .removeprefix(str(self.run_path / "generator_"))
    #                         .removesuffix(".pth")
    #                 )
    #                 for s in self.run_path.glob("generator_*.pth")
    #             ]
    #         )
    #         self.load_checkpoint(epoch)
    #         return epoch
    #     except ValueError:
    #         self.generator.init_weights()
    #         self.classifier.init_weights()
    #         return 0

    # def calculate_metrics(self, epoch, i, dm, rg, pg, gen_loss, dis_loss):
    #     self.writer.add_scalars(
    #         "metrics",
    #         dict(
    #             **{
    #                 name: metric(dm, rg, pg).item()
    #                 for name, metric in self.metrics.items()
    #             },
    #         ),
    #         global_step=epoch * len(self.dataloader) + i,
    #         walltime=float(time.time() - self.init_time),
    #     )
    #     self.writer.add_scalars(
    #         "losses",
    #         {
    #             "gen": gen_loss.item(),
    #             "dis": dis_loss.item(),
    #         },
    #         global_step=epoch * len(self.dataloader) + i,
    #         walltime=float(time.time() - self.init_time),
    #     )

    # def log_rg_training(self, epoch, i, n_epochs, drg):
    #     batches_done = epoch * len(self.dataloader) + i + 1
    #     batches_left = n_epochs * len(self.dataloader) - batches_done
    #     time_left = datetime.timedelta(
    #         seconds=batches_left * (time.time() - self.init_time) / batches_done
    #     )
    #     logging.getLogger("blockwork").info(
    #         f" [Epoch {epoch + 1:03d}/{n_epochs:03d}]"
    #         + f" [Batch {i + 1:01d}/{len(self.dataloader):01d}]"
    #         # + f" [D loss: {dis_loss:.2f}]"
    #         # + f" [G loss: {gen_loss:.2f}]"
    #         + " DRG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(drg, dim=(0, 1))]) + "]"
    #         # + " DPG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(dpg, dim=(0, 1))]) + "]"
    #         + f" ETA: {str(time_left).split('.')[0]}"
    #         + f" MEM {torch.cuda.max_memory_allocated() // (1024 * 1024):05d} MB" if self.cuda else ""
    #     )
    #
    # def log_dis_training(self, epoch, i, n_epochs, dpg, drg):
    #     batches_done = epoch * len(self.dataloader) + i + 1
    #     batches_left = n_epochs * len(self.dataloader) - batches_done
    #     time_left = datetime.timedelta(
    #         seconds=batches_left * (time.time() - self.init_time) / batches_done
    #     )
    #     logging.getLogger("blockwork").info(
    #         f" [Epoch {epoch + 1:03d}/{n_epochs:03d}]"
    #         + f" [Batch {i + 1:01d}/{len(self.dataloader):01d}]"
    #         # + f" [D loss: {dis_loss:.2f}]"
    #         # + f" [G loss: {gen_loss:.2f}]"
    #         + " DRG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(drg, dim=(0, 1))]) + "]"
    #         + " DPG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(dpg, dim=(0, 1))]) + "]"
    #         + f" ETA: {str(time_left).split('.')[0]}"
    #         + f" MEM {torch.cuda.max_memory_allocated() // (1024 * 1024):05d} MB" if self.cuda else ""
    #     )
    #
    # def log_gen_training(self, epoch, i, n_epochs, g_loss):
    #     batches_done = epoch * len(self.dataloader) + i + 1
    #     batches_left = n_epochs * len(self.dataloader) - batches_done
    #     time_left = datetime.timedelta(
    #         seconds=batches_left * (time.time() - self.init_time) / batches_done
    #     )
    #     logging.getLogger("blockwork").info(
    #         f" [Epoch {epoch + 1:03d}/{n_epochs:03d}]"
    #         + f" [Batch {i + 1:01d}/{len(self.dataloader):01d}]"
    #         # + f" [D loss: {dis_loss:.2f}]"
    #         + f" [G loss: {g_loss:.2f}]"
    #         # + " DRG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(drg, dim=(0, 1))]) + "]"
    #         # + " DPG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(dpg, dim=(0, 1))]) + "]"
    #         + f" ETA: {str(time_left).split('.')[0]}"
    #         + f" MEM {torch.cuda.max_memory_allocated() // (1024 * 1024):05d} MB" if self.cuda else ""
    #     )

    # def save_sample(self, dm, rg, pg, drg, dpg, epoch):
    #     torch.save(
    #         {
    #             "dm": dm,
    #             "rg": rg,
    #             "pg": pg,
    #             "drg": drg,
    #             "dpg": dpg,
    #         },
    #         self.run_path / f"sample_{epoch:03d}.npy",
    #     )

    def train(self, n_epochs, normalog=False, sample_interval=None, checkpoint_interval=None):
        logger.info(f"   training for {n_epochs} epochs...")
        self.init_time = time.time()

        for epoch in range(n_epochs):
            for i, halos in enumerate(self.dataloader):
                # if self.cuda:
                #     torch.cuda.reset_peak_memory_stats()
                dm, rg = self.tensor_type(halos["dm"]), self.tensor_type(halos["rg"])
                if normalog:
                    dm = torch.log(dm + 1)
                    rg = torch.log(rg + 1)

                self.g_optimizer.zero_grad()
                pg = self.generator(dm)
                # print(pg)
                loss = self.trainer(dm, rg, pg)
                loss.backward()
                self.g_optimizer.step()

                self.writer.add_scalar("loss", loss.item(), global_step=epoch * len(self.dataloader) + i)

    def evaluate(self, metrics, normalog=False):
        # for i in range(len(self.valid_dataset)):
        #     print(i, self.valid_dataset[i])
        with torch.no_grad():
            evaluation = torch.zeros((len(self.valid_dataset), 1, len(metrics)))
            for i, halos in enumerate(self.valid_dataset):
                dm, rg = self.tensor_type(halos["dm"]).unsqueeze(0), self.tensor_type(halos["rg"]).unsqueeze(0)
                if normalog:
                    dm = torch.log(dm + 1)
                    rg = torch.log(rg + 1)
                    pg = self.generator(dm)
                    dm = torch.exp(dm) - 1
                    rg = torch.exp(rg) - 1
                    pg = torch.exp(pg) - 1
                else:
                    pg = self.generator(dm)

                evaluation[i, 0] = torch.tensor([
                    metrics_dict[label](dm, rg, pg)
                    for label in metrics
                ])
        return evaluation

    def xy_distribution_sample(self, normalog=False):
        i = random.randint(0, len(self.valid_dataset)-1)
        halos = self.valid_dataset[i]
        dm, rg = self.tensor_type(halos["dm"]).unsqueeze(0), self.tensor_type(halos["rg"]).unsqueeze(0)
        if normalog:
            dm = torch.log(dm + 1)
            rg = torch.log(rg + 1)
            pg = self.generator(dm)
            # dm = torch.exp(dm) - 1
            rg = torch.exp(rg) - 1
            pg = torch.exp(pg) - 1
        else:
            pg = self.generator(dm)
        relupg = torch.nn.functional.relu(pg)

        rg_dist = torch.sum(rg, dim=(0, 4))
        pg_dist = torch.sum(pg, dim=(0, 4))
        relupg_dist = torch.sum(relupg, dim=(0, 4))

        rg_dist = rg_dist / torch.max(rg_dist)
        pg_dist = pg_dist / torch.max(pg_dist)
        relupg_dist = relupg_dist / torch.max(relupg_dist)

        self.writer.add_images("xy_dist rg--pg", torch.stack((rg_dist, pg_dist, relupg_dist), dim=0))


def roundrun(rounds, opts, vv_id=None):
    for r in range(rounds):
        vv = Vox2Vox(id=vv_id + ":" + str(r) if vv_id is not None else None, **opts)
        vv.train(n_epochs=opts["n_epochs"], normalog=opts.get("normalog", False))
        try:
            evaluation = torch.cat((evaluation, vv.evaluate(opts["metrics"], normalog=opts.get("normalog", False))), dim=1)
        except NameError:
            evaluation = vv.evaluate(opts["metrics"], normalog=opts.get("normalog", False))
        vv.xy_distribution_sample(normalog=opts.get("normalog", False))

    means = torch.mean(evaluation, dim=(0, 1))
    halo_var = torch.mean(torch.var(evaluation, dim=0, keepdim=True), dim=(0, 1))
    runs_var = torch.mean(torch.var(evaluation, dim=1, keepdim=True), dim=(0, 1))
    for i, label in enumerate(opts["metrics"]):
        logger.info(
            f"{label:>12s}: {means[i].item():.3e} (± {halo_var[i].item():.0e} across halos; ± {runs_var[i].item():.0e} across runs)")

    #  TODO store some cooler stuff, like radial distributions, slice-videos or meshes
