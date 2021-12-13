import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path, PosixPath, WindowsPath

import pandas as pd
import torch
import torch.nn.functional as functional

from .dataset import BlockworkDataset
from .generator import UnetGenerator
from .classifier import MseClassifier
from .logger import set_logger
from preprocess.monolith import preprocess
from torch.utils.tensorboard import SummaryWriter


class BlockworkVox2Vox:
    def __init__(self, opts):
        # preprocess(opts)

        opts.data_path = Path(os.getcwd()) / opts.data_path
        opts.dataset_path = Path(os.getcwd()) / opts.dataset_path
        opts.output_path = Path(os.getcwd()) / opts.output_path
        self.cuda = cuda = torch.cuda.is_available()

        def tensor_type(x):
            if isinstance(x, (list, tuple)):
                return torch.Tensor(x).to(device="cuda" if cuda else "cpu", dtype=torch.float)
            elif isinstance(x, (torch.Tensor, torch.nn.Module)):
                return x.to(device="cuda" if cuda else "cpu", dtype=torch.float)
            else:
                return x

        self.tensor_type = tensor_type

        self.run_path = Path(opts.output_path)
        self.batch_size = opts.batch_size

        self.generator = self.tensor_type(UnetGenerator(1, 1, num_downs=4, use_dropout=True))
        num_parameters = sum([2 ** sum(map(math.log2, list(p.shape))) for p in self.generator.parameters()])
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=1 / num_parameters**0.7, betas=(opts.b1, opts.b2)
        )

        self.classifier = self.tensor_type(MseClassifier())
        num_parameters = sum([2 ** sum(map(math.log2, list(p.shape))) for p in self.classifier.parameters()])
        self.c_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=1 / num_parameters**0.7, betas=(opts.b1, opts.b2)
        )

        self.seems_real = self.tensor_type([1., 0.]).unsqueeze(0).repeat(self.batch_size, 1)
        self.seems_fake = self.tensor_type([0., 1.]).unsqueeze(0).repeat(self.batch_size, 1)

        self.dataloader = torch.utils.data.DataLoader(
            BlockworkDataset(
                path=(
                        Path(opts.data_path)
                        / (
                                f"{opts.sim_name}"
                                + f"_SNAP{opts.snap_num:03d}"
                                + f"_MASS{opts.mass_min:.2e}"
                                + f"_{opts.mass_max:.2e}"
                                + f"_NGASMIN{opts.n_gas_min}"
                        )
                        / f"nvoxel_{opts.nvoxel}"
                        / "train"
                ),
            ),
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=opts.n_cpu,
            drop_last=True,
        )
        self.metrics = {
            "mse": (lambda dm, rg, pg: functional.mse_loss(pg, rg)),
            "l1": (lambda dm, rg, pg: functional.l1_loss(pg, rg)),
        }

        self.writer = SummaryWriter()

    def save_checkpoint(self, epoch):
        torch.save(
            self.generator.state_dict(),
            self.run_path / f"generator_{epoch:03d}.pth",
        )
        torch.save(
            self.classifier.state_dict(),
            self.run_path / f"classifier_{epoch:03d}.pth",
        )

    def load_checkpoint(self, epoch):
        logging.info(f"Loading pretrained models from epoch {epoch}")
        self.generator.load_state_dict(torch.load(self.run_path / f"generator_{epoch:03d}.pth"))
        self.classifier.load_state_dict(torch.load(self.run_path / f"classifier_{epoch:03d}.pth"))

    def resume(self):
        try:
            epoch = max(
                [
                    int(
                        str(s)
                            .removeprefix(str(self.run_path / "generator_"))
                            .removesuffix(".pth")
                    )
                    for s in self.run_path.glob("generator_*.pth")
                ]
            )
            self.load_checkpoint(epoch)
            return epoch
        except ValueError:
            self.generator.init_weights()
            self.classifier.init_weights()
            return 0

    def calculate_metrics(self, epoch, i, dm, rg, pg, gen_loss, dis_loss):
        self.writer.add_scalars(
            "metrics",
            dict(
                **{
                    name: metric(dm, rg, pg).item()
                    for name, metric in self.metrics.items()
                },
            ),
            global_step=epoch * len(self.dataloader) + i,
            walltime=float(time.time() - self.init_time),
        )
        self.writer.add_scalars(
            "losses",
            {
                "gen": gen_loss.item(),
                "dis": dis_loss.item(),
            },
            global_step=epoch * len(self.dataloader) + i,
            walltime=float(time.time() - self.init_time),
        )

    def log_training_step(self, epoch, i, n_epochs, gen_loss, dis_loss, dpg, drg):
        batches_done = epoch * len(self.dataloader) + i + 1
        batches_left = n_epochs * len(self.dataloader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - self.init_time) / batches_done
        )
        logging.getLogger("blockwork").info(
            f" [Epoch {epoch + 1:03d}/{n_epochs:03d}]"
            + f" [Batch {i + 1:01d}/{len(self.dataloader):01d}]"
            # + f" [D loss: {dis_loss:.2f}]"
            # + f" [G loss: {gen_loss:.2f}]"
            + " DRG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(drg, dim=(0, 1)).cpu().numpy()]) + "]"
            + " DPG [" + ", ".join([f'{x:-.2f}' for x in torch.mean(dpg, dim=(0, 1)).cpu().numpy()]) + "]"
            + f" ETA: {str(time_left).split('.')[0]}"
            + f" MEM {torch.cuda.max_memory_allocated() // (1024 * 1024):05d} MB" if self.cuda else ""
        )

    # def train_step(self, dm, rg):
    #     dm, rg = self.tensor_type(dm), self.tensor_type(rg)
    #
    #     # self.gen_opt.zero_grad()
    #     # pg = self.gen(dm)
    #     pg = torch.randn_like(rg)
    #     dpg = self.classifier(torch.cat([dm, pg], dim=1))
    #     # raise RuntimeError(f"{dpg=}")
    #     # print(f"{self.tensor_type([1., 0.]).view((1, 2)).repeat(64, 1).shape=}")
    #
    #     gen_loss = (
    #         functional.mse_loss(dpg, self.seems_real)
    #         # + 1e2 * aid_loss(rg, pg.detach())
    #     )
    #     # gen_loss = functional.mse_loss(dpg, self.tensor_type([1., 0.]))
    #     # gen_loss.backward()
    #     # self.gen_opt.step()
    #
    #     self.c_optimizer.zero_grad()
    #     drg = self.classifier(torch.cat([dm, rg], dim=1))
    #     dpg = dpg.detach()
    #     dis_loss = (
    #                        torch.max(torch.abs(dpg - self.seems_fake))
    #                        + torch.max(torch.abs(drg - self.seems_real))
    #                ) / 2
    #     dis_loss.backward()
    #     self.c_optimizer.step()
    #
    #     return dm, rg, pg, drg, dpg, gen_loss, dis_loss

    def save_sample(self, dm, rg, pg, drg, dpg, epoch):
        torch.save(
            {
                "dm": dm,
                "rg": rg,
                "pg": pg,
                "drg": drg,
                "dpg": dpg,
            },
            self.run_path / f"sample_{epoch:03d}.npy",
        )

    # def pre_train(self):
    #     for i, (dm, rg) in enumerate(self.dataloader):
    #         dm, rg = dm.type(self.tensor_type), rg.type(self.tensor_type)
    #         pg = self.gen(dm)
    #         loss = torch.Tensor([0.0]).cuda()
    #         for dim in (2, 3, 4):
    #             sum_rg = torch.mean(rg, dim=[d for d in (1, 2, 3, 4) if d != dim])
    #             sum_pg = torch.mean(pg, dim=[d for d in (1, 2, 3, 4) if d != dim])
    #             loss += functional.l1_loss(sum_rg, sum_pg)
    #         loss.backward()
    #         with torch.no_grad():
    #             for p in self.gen.parameters():
    #                 p -= 0.001 * (0.9 ** (i / 4)) * p.grad
    #     logging.getLogger("blockwork").info(
    #         ""
    #         + f"{sum_rg=}\n"
    #         + f"{sum_pg=}\n"
    #         + f"{loss.item()=}"
    #         # + f"Parameters = {[ p for p in self.gen.parameters() ]}\n"
    #         # + f"Gradients = {[ p.grad for p in self.gen.parameters() ]}\n"
    #     )

    def train(self, n_epochs, sample_interval, checkpoint_interval):
        start_from_epoch = self.resume()

        self.init_time = time.time()

        for epoch in range(start_from_epoch, n_epochs):

            for i, (dm, rg) in enumerate(self.dataloader):
                if self.cuda:
                    torch.cuda.reset_peak_memory_stats()
                dm, rg = self.tensor_type(dm), self.tensor_type(rg)

                if epoch % sample_interval == 0 and i == 0:
                    with torch.no_grad():
                        pg = self.generator(dm)
                        drg = self.classifier.forward(torch.cat([dm, rg], dim=1))
                        dpg = self.classifier.forward(torch.cat([dm, pg], dim=1))
                        self.save_sample(
                            dm=dm,
                            rg=rg,
                            pg=pg,
                            drg=drg,
                            dpg=dpg,
                            epoch=epoch,
                        )

                self.g_optimizer.zero_grad()
                pg = self.generator(dm)
                dpg = self.classifier(torch.cat([dm, pg], dim=1))
                # g_loss = self.classifier.loss(torch.cat([dm, pg], dim=1), self.seems_real)
                g_loss = functional.mse_loss(dpg, self.seems_real)
                g_loss.backward()
                self.g_optimizer.step()

                self.c_optimizer.zero_grad()
                pg = self.generator(dm)
                dpg = self.classifier(torch.cat([dm, pg], dim=1))
                drg = self.classifier(torch.cat([dm, rg], dim=1))
                c_loss = 0.5 * (
                    functional.mse_loss(drg, self.seems_real)
                    # self.classifier.loss(torch.cat([dm, rg], dim=1), self.seems_real)
                    + functional.mse_loss(dpg, self.seems_fake)
                    # + self.classifier.loss(torch.cat([dm, pg], dim=1), self.seems_fake)
                )
                c_loss.backward()
                self.c_optimizer.step()

                self.calculate_metrics(epoch=epoch, i=i, dm=dm, rg=rg, pg=pg, gen_loss=g_loss, dis_loss=c_loss)
                self.log_training_step(epoch=epoch, i=i, n_epochs=n_epochs, drg=drg.detach(), dpg=dpg.detach(), gen_loss=g_loss.item(), dis_loss=c_loss.item())

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch=epoch)

# class Classifier:
#     def __init__(self, opt):
#         preprocess(opt)
#
#         self.run_path = Path(opt.output_path)
#         self.batch_size = opt.batch_size
#         self.tensor_type = torch.cuda.FloatTensor
#
#         self.gen = UnetGenerator(1, 1, num_downs=4, use_dropout=True).cuda()
#         self.gen_opt = torch.optim.Adam(
#             self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
#         )
#         # self.dis = NLayerDiscriminator(input_nc=2).cuda()
#         self.dis = AltDiscriminator().cuda()
#         self.dis_opt = torch.optim.Adam(
#             self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
#         )
#         self.seems_real = self.tensor_type([1., 0.]).view((1, 2)).repeat(self.batch_size, 1)
#         self.seems_fake = self.tensor_type([0., 1.]).view((1, 2)).repeat(self.batch_size, 1)
#
#         self.dataloader = torch.utils.data.DataLoader(
#             BlockworkDataset(
#                 path=(
#                         Path(opt.data_path)
#                         / f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}"
#                         / f"nvoxel_{opt.nvoxel}"
#                         / "train"
#                 ),
#             ),
#             batch_size=opt.batch_size,
#             shuffle=True,
#             num_workers=opt.n_cpu,
#             drop_last=True,
#         )
#         self.metrics_functions = {
#             "mse": (lambda dm, rg, pg: functional.mse_loss(pg, rg)),
#             "l1": (lambda dm, rg, pg: functional.l1_loss(pg, rg)),
#         }
#         # self.running_metrics = pd.DataFrame(
#         #     columns=[
#         #         "epoch",
#         #         "time",
#         #         "gen_loss",
#         #         "dis_loss",
#         #         *self.metrics_functions.keys(),
#         #     ]
#         # )
#         self.writer = SummaryWriter()
