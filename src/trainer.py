from collections import defaultdict, OrderedDict
import pandas as pd
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from enum import Enum
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import TaskDefinedBatch
from .model import MorphologyClassifier
from typing import Callable
from pathlib import Path


class ScheduleStrategy(Enum):
    EPOCH = "epoch"
    BATCH = "batch"


class LifecycleMode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _update_cur_metrics(cur_metrics: dict[str, float], metrics_to_add: dict[str, float], suffix: str) -> None:
    for k, v in metrics_to_add.items():
        if k in cur_metrics:
            raise ValueError(f"{k} already defined")
        cur_metrics[k + "_" + suffix] = v


class MultitaskTrainer:

    def __init__(
        self,

        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,

        schedule_strategy: ScheduleStrategy = ScheduleStrategy.EPOCH,
    ):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_strategy = schedule_strategy

    def _train_batch_morphology(
        self,
        model: MorphologyClassifier,
        batch: TaskDefinedBatch,
        mode: LifecycleMode,
        device: torch.device | str = "cpu",
    ) -> dict[str, float]:

        batch.to(device)
        if mode == LifecycleMode.TRAIN:
            self.optimizer.zero_grad()
            outputs = model(batch)
        else:
            with torch.no_grad():
                outputs = model(batch)

        # per_category_logits = outputs['per_category_logits']
        per_category_losses = outputs['per_category_losses']

        loss = torch.stack(list(per_category_losses.values())).sum()

        if mode == LifecycleMode.TRAIN:
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.schedule_strategy == ScheduleStrategy.BATCH and self.scheduler is not None:
                self.scheduler.step()

        result = {
            "loss": loss.detach().item()
        }

        for k, val in per_category_losses.items():
            result[k] = float(val.detach().item())

        return result

    def _train_batch_lemmatization(
            self,
            model: MorphologyClassifier,
            batch: TaskDefinedBatch,
            mode: LifecycleMode,
            device: torch.device | str = "cpu",
        ) -> dict[str, float]: raise NotImplementedError

    def _process_batch(
        self,
        model: MorphologyClassifier,
        batch: TaskDefinedBatch,
        mode: LifecycleMode,
        device: torch.device | str = "cpu",
    ) -> dict[str, float]:
        task_name = batch.task_name
        if task_name == "pos+morphology":
            result = self._train_batch_morphology(model=model, batch=batch, device=device, mode=mode)
        elif task_name == "lemmatization":
            result = self._train_batch_lemmatization(model=model, batch=batch, device=device, mode=mode)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        result = {k + "_" + task_name: val for k, val in result.items()}
        return result

    def _process_epoch(
        self,
        model: MorphologyClassifier,
        iterator: torch.utils.data.DataLoader[TaskDefinedBatch],
        estimated_iter_size: int,
        mode: LifecycleMode,
        device:  torch.device | str = "cpu",
    ) -> dict[str, float]:
        if mode == LifecycleMode.TRAIN:
            model.train()
        else:
            model.eval()

        all_results = defaultdict(list)

        for batch in tqdm(iterator, total=estimated_iter_size):

            train_batch_result = self._process_batch(model=model, batch=batch, device=device, mode=mode)
            for k, val in train_batch_result.items():
                all_results[k].append(val)

        if mode == LifecycleMode.TRAIN:
            if self.schedule_strategy == ScheduleStrategy.EPOCH and self.scheduler is not None:
                self.scheduler.step()

        avg_results = {k: float(np.mean(val)) for k, val in all_results.items()}

        return avg_results

    def _run_training_loop(
        self,

        model: MorphologyClassifier,
        device:  torch.device | str,

        train_dataloader: torch.utils.data.DataLoader[TaskDefinedBatch],
        estimated_train_size: int,
        val_dataloader: torch.utils.data.DataLoader[TaskDefinedBatch],
        estimated_val_size: int,
        n_epochs: int,
        cpt_dir: str | Path,

        main_metric: str = "loss",
        greater_is_better: bool = False,
        max_patience: int = 3,
        unfreeze_backbone_after: int | None = None,
    ) -> list[dict[str, float]]:

        if isinstance(cpt_dir, str):
            cpt_dir = Path(cpt_dir)
        cpt_dir.mkdir(exist_ok=True)

        patience = 0
        best_val_metric = 0 if greater_is_better else float("inf")
        all_metrics: list[dict[str, float]] = []

        # TODO: make this work properly
        main_metric = main_metric + "_pos+morphology"

        for epoch in range(1, n_epochs + 1):

            print(f"Epoch: {epoch}/{n_epochs}")

            cur_metrics: dict[str, float] = OrderedDict()

            train_metrics = self._process_epoch(
                model=model,
                iterator=train_dataloader,
                device=device,
                estimated_iter_size=estimated_train_size,
                mode=LifecycleMode.TRAIN
            )

            _update_cur_metrics(cur_metrics, train_metrics, "train")

            val_metrics = self._process_epoch(
                model=model,
                iterator=val_dataloader,
                device=device,
                estimated_iter_size=estimated_val_size,
                mode=LifecycleMode.VALIDATION
            )

            _update_cur_metrics(cur_metrics, val_metrics, "val")

            all_metrics.append(cur_metrics)

            if greater_is_better:
                checkpoint_is_better = val_metrics[main_metric] > best_val_metric
            else:
                checkpoint_is_better = val_metrics[main_metric] < best_val_metric

            if checkpoint_is_better:

                improvement = abs(val_metrics[main_metric] - best_val_metric)
                best_val_metric = val_metrics[main_metric]

                cpt_name = f"cpt_{epoch}"
                cur_cpt_dir = cpt_dir / cpt_name
                cur_cpt_dir.mkdir()
                model.save(cur_cpt_dir / "state_dict.pt")
                print(f"Save model: {main_metric}={best_val_metric: .4f} (improvement {improvement: .4f})")

                patience = 0

            else:
                patience += 1
                if patience >= max_patience:
                    break

            if unfreeze_backbone_after is not None and epoch >= unfreeze_backbone_after:
                print("Unfreezing model")
                unfreeze_backbone_after = None
                model.train_backbone(True)

        return all_metrics

    def train(
        self,

        model: MorphologyClassifier,
        device:  torch.device | str,
        train_dataset: torch.utils.data.Dataset[TaskDefinedBatch],
        estimated_train_size: int,
        val_dataset: torch.utils.data.Dataset[TaskDefinedBatch],
        estimated_val_size: int,
        collate_fn: Callable[[list[TaskDefinedBatch]], TaskDefinedBatch],
        worker_init_fn: Callable[[int], None],

        cpt_dir: str | Path,

        n_epochs: int,
        batch_size: int,

        main_metric: str = "loss",
        greater_is_better: bool = False,
        max_patience: int = 3,
        unfreeze_backbone_after: int | None = None,
    ) -> pd.DataFrame:

        model.to(device)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            num_workers=4,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            num_workers=4,
        )

        all_metrics = self._run_training_loop(
            model=model,
            device=device,

            train_dataloader=train_dataloader,
            estimated_train_size=math.ceil(estimated_train_size / batch_size),
            val_dataloader=val_dataloader,
            estimated_val_size=math.ceil(estimated_val_size / batch_size),
            n_epochs=n_epochs,
            cpt_dir=cpt_dir,

            main_metric=main_metric,
            greater_is_better=greater_is_better,
            max_patience=max_patience,

            unfreeze_backbone_after=unfreeze_backbone_after,
        )

        training_log = pd.DataFrame(all_metrics)
        return training_log
