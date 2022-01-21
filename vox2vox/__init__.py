import hashlib
import math
import os.path
import shutil
from datetime import datetime

import torch
from pathlib import Path

from adabelief_pytorch import AdaBelief
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from vox2vox.dataset import BasicDataset
from vox2vox.generators import SUNet

from logger import logger
from vox2vox.metrics import metrics_dict
from vox2vox.trainers import L1ProjectionsTrainer, L1SumTrainer

def write_images(tag, data, writer, projection=-1):
    data = [
        [
            torch.sum(torch.relu(box.squeeze(0)), dim=projection)
            for box in row
        ]
        for row in data
    ]
    data = [
        square / max( torch.max(_sq) for _sq in row )
        for row in data for square in row
    ]
    data = [ torch.log(20*square + 1) / math.log(21) for square in data ]
    print([ torch.max(square) for square in data ])
    writer.add_images(tag, torch.stack(data, dim=0))

class Vox2Vox:
    def __init__(
            self,
            id=None,
            **opts,
    ):
        self.id = id
        self.path = Path(opts["models_base_path"]) / id

        def right_to(x):
            return x.to(device="cuda" if opts["cuda"] else "cpu", dtype=torch.float)

        self.right_to = right_to

        if opts["mod_main_generator"] == "sunet":
            self.main_generator = self.right_to(
                SUNet(
                    features=opts["mod_main_features"],
                    levels=opts["mod_main_levels"],
                )
            )
        else:
            raise ValueError(f"Option {opts['mod_main_generator']=} is invalid.")

        if opts["mod_patch_generator"] == "sunet":
            self.patch_generator = self.right_to(
                SUNet(
                    features=opts["mod_patch_features"],
                    levels=opts["mod_patch_levels"],
                )
            )
        else:
            raise ValueError(f"Option {opts['mod_patch_generator']=} is invalid.")

    def initialize(self):
        self.main_generator.initialize()
        self.patch_generator.initialize()

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        torch.save(
            self.main_generator.state_dict(),
            self.path / f"main_generator.pth",
        )
        torch.save(
            self.patch_generator.state_dict(),
            self.path / f"patch_generator.pth",
        )

    def load(self):
        self.main_generator.load_state_dict(torch.load(
            self.path / f"main_generator.pth"
        ))
        self.patch_generator.load_state_dict(torch.load(
            self.path / f"patch_generator.pth"
        ))
        return self

    def deg_step(self, dm, rg, opts, writer=None):
        degdm = functional.avg_pool3d(dm, kernel_size=4)
        degrg = functional.avg_pool3d(rg, kernel_size=4)
        degpg = self.main_generator(degdm)
        if writer:
            write_images("deg step", [[degdm], [degrg, degpg]], writer)
        return degdm, degrg, degpg


def init(model_id, opts):
    v2v = Vox2Vox(id=model_id, **opts)
    v2v.initialize()
    v2v.save()


def train(model_id, opts):
    v2v = Vox2Vox(id=model_id, **opts).load()
    dataloader = torch.utils.data.DataLoader(
        BasicDataset(
            path=(
                    Path(opts["voxels_base_path"])
                    / opts["restricted_sim_name"]
                    / f"nvoxel_{opts['nvoxel']}"
                    / "train"
            ),
            fixed_size=opts.get("dataset_fixed_size", None),
        ),
        batch_size=opts["batch_size"],
        shuffle=True,
        num_workers=opts["n_cpu"],
        drop_last=False,
    )
    main_generator_optimizer = AdaBelief(
        v2v.main_generator.parameters(),
        lr=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=True,
        rectify=True,
        print_change_log=False,
    )
    main_trainer = L1SumTrainer()
    writer = SummaryWriter(
        Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id / datetime.now().isoformat()
    )
    for epoch in range(opts["n_epochs"]):
        for i, halos in enumerate(dataloader):
            dm, rg = v2v.right_to(halos["dm"]), v2v.right_to(halos["rg"])


            main_generator_optimizer.zero_grad()
            degdm, degrg, degpg = v2v.deg_step(dm, rg, opts)
            loss = main_trainer(degdm, degrg, degpg)
            loss.backward()
            main_generator_optimizer.step()

            badpg = functional.interpolate(degpg.detach(), size=opts["nvoxel"], mode="nearest")
            badloss = main_trainer(dm, rg, badpg)
        writer.add_scalar("loss", loss.item(), global_step=epoch)

    v2v.save()


def evaluate(model_id, opts):
    v2v = Vox2Vox(id=model_id, **opts).load()
    dataloader = torch.utils.data.DataLoader(
        BasicDataset(
            path=(
                    Path(opts["voxels_base_path"])
                    / opts["restricted_sim_name"]
                    / f"nvoxel_{opts['nvoxel']}"
                    / "valid"
            ),
            fixed_size=opts.get("dataset_fixed_size", None),
        ),
        batch_size=opts["batch_size"],
        shuffle=True,
        num_workers=opts["n_cpu"],
        drop_last=True,
    )
    writer = SummaryWriter(
        Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id / datetime.now().isoformat()
    )

    with torch.no_grad():
        # evaluation = torch.zeros((len(self.valid_dataset), 1, len(metrics_dict)))
        halos = next(iter(dataloader))
        dm, rg = v2v.right_to(halos["dm"]), v2v.right_to(halos["rg"])
        degdm, degrg, degpg = v2v.deg_step(dm, rg, opts)

        badpg = functional.interpolate(degpg.detach(), size=opts["nvoxel"], mode="nearest")

        output = {
            label: metrics_dict[label](dm, rg, badpg)
            for label in opts["metrics"]
        }
        for label in opts["metrics"]:
            writer.add_histogram(
                f"evaluate:{label}",
                output[label][:, 0]
            )
        return {
            label: torch.std_mean(t)
            for label, t in output.items()
        }


def apply(model_id, opts):
    v2v = Vox2Vox(id=model_id, **opts).load()
    dataset = BasicDataset(
        path=(
                Path(opts["voxels_base_path"])
                / opts["restricted_sim_name"]
                / f"nvoxel_{opts['nvoxel']}"
                / "valid"
        ),
        fixed_size=opts.get("dataset_fixed_size", None),
    )
    writer = SummaryWriter(
        Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id / datetime.now().isoformat()
    )
    halos = dataset.get_random()
    with torch.no_grad():
        dm, rg = v2v.right_to(halos["dm"]).unsqueeze(0), v2v.right_to(halos["rg"]).unsqueeze(0)
        degdm, degrg, degpg = v2v.deg_step(dm, rg, opts, writer)

        badpg = functional.interpolate(degpg.detach(), size=opts["nvoxel"], mode="nearest")

        dm_dist = torch.sum(dm, dim=(0, 4))
        rg_dist = torch.sum(rg, dim=(0, 4))
        badpg_dist = torch.sum(badpg, dim=(0, 4))
        relubadpg_dist = torch.sum(functional.relu(badpg), dim=(0, 4))

        dm_dist = dm_dist / torch.max(dm_dist)

        _max = max(torch.max(rg_dist), torch.max(badpg_dist))
        rg_dist = rg_dist / _max
        badpg_dist = badpg_dist / _max
        relubadpg_dist = relubadpg_dist / _max

        writer.add_images("xy_dist", torch.stack((dm_dist, rg_dist, badpg_dist, relubadpg_dist), dim=0))


def get_hash(opts):
    return hashlib.md5(str({
        key: value
        for (key, value) in sorted(opts.items())
        if key[:3] == "mod"
    }).encode("utf8")).hexdigest().upper()[:8]
