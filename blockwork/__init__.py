import datetime
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional

from .dataset import BlockworkDataset
from .generator import UnetGenerator
from .shallow_generator import ShallowGenerator
from .discriminator import NLayerDiscriminator
from .logger import set_logger
from preprocess.monolith import preprocess

class BlockworkVox2Vox:
    def __init__(self, opt):
        preprocess(opt)

        self.init_time = None
        self.run_path = Path(opt.run_path)
        self.tensor_type = torch.cuda.FloatTensor

        self.gen = UnetGenerator(1, 1, 4).cuda()
        self.gen_opt = torch.optim.Adam(
            self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )
        self.dis = NLayerDiscriminator(input_nc=2).cuda()
        self.dis_opt = torch.optim.Adam(
            self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )
        self.dataloader = BlockworkDataset(
            path=(
                    Path(opt.data_path)
                    / f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}"
                    / f"nvoxel_{opt.nvoxel}"
                    / "train"
            ),
            block_side=16,
            world_side=64,
            margin=8,
            batch_size=4,
        )
        self.metrics_functions = {
            "mse": (lambda dm, rg, pg: functional.mse_loss(pg, rg)),
            "l1": (lambda dm, rg, pg: functional.l1_loss(pg, rg)),
        }
        self.running_metrics = pd.DataFrame(
            columns=[
                "epoch",
                "time",
                "gen_loss",
                "dis_loss",
                *self.metrics_functions.keys(),
            ]
        )

    def save_checkpoint(self, epoch):
        torch.save(
            self.gen.state_dict(),
            self.run_path / f"gen_{epoch:03d}.pth",
        )
        torch.save(
            self.dis.state_dict(),
            self.run_path / f"dis_{epoch:03d}.pth",
        )

    def load_checkpoint(self, epoch):
        logging.info(f"Loading pretrained models from epoch {epoch}")
        self.gen.load_state_dict(torch.load(self.run_path / f"gen_{epoch:03d}.pth"))
        self.dis.load_state_dict(torch.load(self.run_path / f"dis_{epoch:03d}.pth"))

    def resume(self):
        try:
            epoch = max(
                [
                    int(
                        str(s)
                            .removeprefix(str(self.run_path / "gen_"))
                            .removesuffix(".pth")
                    )
                    for s in self.run_path.glob("gen_*.pth")
                ]
            )
            self.load_checkpoint(epoch)
            return epoch
        except ValueError:
            self.gen.init_weights()
            self.dis.init_weights()
            self.pre_train()
            return 0

    def calculate_metrics(self, epoch, dm, rg, pg, gen_loss, dis_loss):
        self.running_metrics.append(
            dict(
                {
                    "epoch": epoch,
                    "time": float(time.time() - self.init_time),
                    "gen_loss": gen_loss.item(),
                    "dis_loss": dis_loss.item(),
                },
                **{
                    name: metric(dm, rg, pg).item()
                    for name, metric in self.metrics_functions.items()
                },
            ),
            ignore_index=True,
        )

    def log_training_step(self, epoch, i, n_epochs, gen_loss, dis_loss):
        batches_done = epoch * len(self.dataloader) + i + 1
        batches_left = n_epochs * len(self.dataloader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - self.init_time) / batches_done
        )
        logging.getLogger("blockwork").info(
            f" [Epoch {epoch + 1:03d}/{n_epochs:03d}]"
            + f" [Batch {i + 1:03d}/{len(self.dataloader):03d}]"
            + f" [D loss: {dis_loss:.3e}]"
            + f" [G loss: {gen_loss:.3e}]"
            + f" ETA: {str(time_left).split('.')[0]}"
            + f" MEM {torch.cuda.max_memory_allocated() // (1024*1024):05d} MB"
        )

    def train_step(self, dm, rg):
        dm, rg = dm.type(self.tensor_type), rg.type(self.tensor_type)

        self.gen_opt.zero_grad()
        pg = self.gen(dm)
        dpg = self.dis(torch.cat([dm, pg], dim=1))
        gen_loss = functional.mse_loss(dpg, torch.ones_like(dpg)) + functional.mse_loss(rg, pg.detach())
        gen_loss.backward()
        self.gen_opt.step()

        self.dis_opt.zero_grad()
        drg = self.dis(torch.cat([dm, rg], dim=1))
        dpg = dpg.detach()
        dis_loss = functional.mse_loss(
            dpg, torch.zeros_like(dpg)
        ) + functional.mse_loss(drg, torch.ones_like(drg))
        dis_loss.backward()
        self.dis_opt.step()

        # logging.getLogger("blockwork").info(
        #     f"{torch.sum(rg)=} {torch.sum(pg)=}"
        # )

        return dm, rg, pg, drg, dpg, gen_loss, dis_loss

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

    def pre_train(self):
        for i, (dm, rg) in enumerate(self.dataloader):
            dm, rg = dm.type(self.tensor_type), rg.type(self.tensor_type)
            pg = self.gen(dm)
            loss = torch.Tensor([0.0]).cuda()
            for dim in (2, 3, 4):
                sum_rg = torch.mean(rg, dim=[d for d in (1, 2, 3, 4) if d!=dim])
                sum_pg = torch.mean(pg, dim=[d for d in (1, 2, 3, 4) if d!=dim])
                loss += functional.l1_loss(sum_rg, sum_pg)
            loss.backward()
            with torch.no_grad():
                for p in self.gen.parameters():
                    p -= 0.001 * (0.9 ** (i/4)) * p.grad
        logging.getLogger("blockwork").info(
            ""
            + f"{sum_rg=}\n"
            + f"{sum_pg=}\n"
            + f"{loss.item()=}"
            # + f"Parameters = {[ p for p in self.gen.parameters() ]}\n"
            # + f"Gradients = {[ p.grad for p in self.gen.parameters() ]}\n"
        )


    def train(self, n_epochs, sample_interval, checkpoint_interval):
        self.running_metrics = pd.DataFrame(
            columns=[
                "epoch",
                "time",
                "gen_loss",
                "dis_loss",
                *self.metrics_functions.keys(),
            ]
        )

        start_from_epoch = self.resume()

        self.init_time = time.time()

        for epoch in range(start_from_epoch, n_epochs):
            for i, (dm, rg) in enumerate(self.dataloader):
                torch.cuda.reset_peak_memory_stats()
                dm, rg, pg, drg, dpg, gen_los, dis_loss = self.train_step(dm, rg)
                self.calculate_metrics(epoch=epoch, dm=dm, rg=rg, pg=pg, gen_loss=gen_los, dis_loss=dis_loss)
                self.log_training_step(epoch=epoch, i=i, n_epochs=n_epochs, gen_loss=gen_los, dis_loss=dis_loss)

                if epoch % sample_interval == 0 and i == 0:
                    self.save_sample(
                        dm=dm,
                        rg=rg,
                        pg=pg.detach(),
                        drg=drg.detach(),
                        dpg=dpg.detach(),
                        epoch=epoch,
                    )

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch=epoch)
