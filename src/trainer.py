from collections import defaultdict, OrderedDict
import pandas as pd
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Literal
from enum import Enum
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class ScheduleStrategy(Enum):
    EPOCH = "epoch"
    BATCH = "batch"

class LifecycleMode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

def _update_cur_metrics(cur_metrics, metrics_to_add, suffix):
    for k, v in metrics_to_add.items():
        if k in cur_metrics: raise ValueError(f"{k} already defined")
        cur_metrics[k + "_" + suffix] = v

class MultitaskTrainer:

    def __init__(
        self,

        optimizer: Optimizer,
        scheduler: LRScheduler | None=None,

        schedule_strategy: ScheduleStrategy = ScheduleStrategy.EPOCH,
    ):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_strategy = schedule_strategy


    def _train_batch_morphology(
        self,
        model,
        batch,
        mode: LifecycleMode,
        device="cpu",
    ):
        # get predictions
        batch.to(device)
        if mode == LifecycleMode.TRAIN:
            self.optimizer.zero_grad()
            outputs = model(batch)
        else:
            with torch.no_grad():
                outputs = model(batch)

        per_category_logits = outputs['per_category_logits']
        per_category_losses = outputs['per_category_losses']

        loss = torch.stack(list(per_category_losses.values())).mean()

        if mode == LifecycleMode.TRAIN:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.schedule_strategy == "batch" and self.scheduler is not None:
                self.scheduler.step()

        result = {
            "loss": loss.detach().item()
        }

        return result

    def _train_batch_lemmatization(
            self,
            model,
            batch,
            mode: LifecycleMode,
            device="cpu",
        ): raise NotImplementedError

    def _process_batch(   
        self,
        model,
        batch,
        mode: LifecycleMode,
        device="cpu",
    ):
        task_name = batch.task_name
        if task_name == "pos+morphology":
            result =  self._train_batch_morphology(model=model, batch=batch, device=device, mode=mode)
        elif task_name == "lemmatization":
            result =  self._train_batch_lemmatization(model=model, batch=batch, device=device, mode=mode)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        result = {k + "_" + task_name: val for k, val in result.items()}
        return result

    def _process_epoch(
        self,
        model,
        iterator,
        estimated_iter_size,
        mode: LifecycleMode,
        device="cpu",
    ):
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
            if self.schedule_strategy == "epoch" and self.scheduler is not None:
                self.scheduler.step()

        avg_results = {k: np.mean(val) for k, val in all_results.items()}

        return avg_results

    def _run_training_loop(
        self,

        model,
        device,

        train_dataloader,
        estimated_train_size,
        val_dataloader,
        estimated_val_size,
        n_epochs,

        main_metric="loss",
        greater_is_better=False,
        max_patience=3,
    ):

        patience = 0
        best_val_metric = 0 if greater_is_better else float("inf")
        all_metrics = []

        # # TODO: make this work properly
        main_metric = main_metric + "_pos+morphology"

        for epoch in range(1, n_epochs + 1):

            print(f"Epoch: {epoch}/{n_epochs}")

            cur_metrics = OrderedDict()

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
            print(cur_metrics)
            if greater_is_better:
                cmp_fn = lambda new, old: new > old
            else:
                cmp_fn = lambda new, old: new < old

            checkpoint_is_better = cmp_fn(val_metrics[main_metric], best_val_metric)

            if checkpoint_is_better:

                improvement = abs(val_metrics[main_metric] - best_val_metric)
                best_val_metric = val_metrics[main_metric]

                print(f"Save model: {main_metric}={best_val_metric} (improvement {improvement})")

                # TODO: make this normal
                torch.save(model, "checkpoints/model.pt")
                patience = 0

            else:
                patience += 1
                if patience >= max_patience:
                    break

            # TODO: make this normal
            training_log = pd.DataFrame(all_metrics)
            training_log.to_csv("training_log.csv")

        return training_log
    

    def train(
        self,

        model,
        device,
        train_dataset,
        estimated_train_size,
        val_dataset,
        estimated_val_size,
        collate_fn,

        n_epochs: int,
        batch_size: int,

        main_metric="loss",
        greater_is_better=False,
        max_patience=3,
    ):

        model.to(device)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        return self._run_training_loop(
            model=model,
            device=device,

            train_dataloader=train_dataloader,
            estimated_train_size=estimated_train_size // batch_size,
            val_dataloader=val_dataloader,
            estimated_val_size=estimated_val_size // batch_size,
            n_epochs=n_epochs,

            main_metric=main_metric,
            greater_is_better=greater_is_better,
            max_patience=max_patience,
        )
