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
from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore


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
        cur_metrics[k + "::" + suffix] = v


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

        # keyed by (task_name, category) -> [list_of_preds, list_of_golds, n_classes]
        self._metric_buffer: dict[tuple[str, str], dict[str, list[np.typing.ArrayLike]]] = {}

    def _reset_metric_buffer(self) -> None:
        self._metric_buffer = {}

    @torch.no_grad()
    def _accumulate_metrics(
        self,
        task_name: str,
        per_category_logits: dict[str, torch.Tensor],
        labels: torch.Tensor,
        names_order: tuple[str, ...],
    ) -> None:
        for i, category in enumerate(names_order):
            cur_labels = labels[:, :, i]  # [bs, seqlen]
            cur_logits = per_category_logits[category]  # [bs, seqlen, nclasses]

            _, _, nclasses = cur_logits.shape
            logits_flat = cur_logits.reshape(-1, nclasses)  # [bs*seqlen, nclasses]
            labels_flat = cur_labels.reshape(-1)  # [bs*seqlen]

            # Binary head: single logit -> threshold at 0
            if nclasses == 1:
                preds = (logits_flat.squeeze(-1) > 0).long()
            else:
                preds = logits_flat.argmax(dim=-1)

            valid = labels_flat != -100
            preds_np = preds[valid].cpu().numpy()
            labels_np = labels_flat[valid].cpu().numpy()

            key = (task_name, category)
            if key not in self._metric_buffer:
                self._metric_buffer[key] = {
                    "preds": [],
                    "labels": [],
                }
            self._metric_buffer[key]["preds"].append(preds_np)
            self._metric_buffer[key]["labels"].append(labels_np)

    def _compute_metrics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for (task_name, category), buf in self._metric_buffer.items():
            preds = np.concatenate(buf["preds"])
            labels = np.concatenate(buf["labels"])
            prefix = f"{task_name}::{category}"

            for average in ("micro", "macro", "weighted"):
                result[f"{prefix}::{average}_precision"] = precision_score(labels, preds, average=average, zero_division=0)
                result[f"{prefix}::{average}_recall"] = recall_score(labels, preds, average=average, zero_division=0)
                result[f"{prefix}::{average}_f1"] = f1_score(labels, preds, average=average, zero_division=0)

        return result

    def _train_batch_morphology(
        self,
        model: MorphologyClassifier,
        batch: TaskDefinedBatch,
        mode: LifecycleMode,
        device: torch.device | str = "cpu",
    ) -> dict[str, float]:

        batch.to(device)

        # --- forward ---
        if mode == LifecycleMode.TRAIN:
            self.optimizer.zero_grad()
            outputs = model(batch)
        else:
            with torch.no_grad():
                outputs = model(batch)

        loss = outputs["loss"]

        # --- backward ---
        if mode == LifecycleMode.TRAIN:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.schedule_strategy == ScheduleStrategy.BATCH and self.scheduler is not None:
                self.scheduler.step()

        # --- update metric containers ---
        if mode != LifecycleMode.TRAIN:
            per_category_logits = outputs["per_category_logits"]
            labels = batch["labels"]  # [bs, seqlen, n_categories]
            self._accumulate_metrics(
                batch.task_name,
                per_category_logits,
                labels,
                model.names_order
            )

        # --- loss report ---
        result = {"loss": loss.detach().item()}
        for k, val in outputs['per_category_losses'].items():
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
        result = {task_name + "::" + (k + "::loss" if k != "loss" else k): val for k, val in result.items()}
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

        if mode != LifecycleMode.TRAIN:
            for k, val in self._compute_metrics().items():
                avg_results[k] = val
            self._reset_metric_buffer()

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
        cpt_dir.mkdir()

        patience = 0
        best_val_metric = 0 if greater_is_better else float("inf")
        all_metrics: list[dict[str, float]] = []

        # TODO: make this work properly
        main_metric = "pos+morphology::" + main_metric

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
