import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from src.categories import names_order, name2mapping_from_id
from src.model import MorphologyClassifier
from src.utils import seed_everything, worker_init_fn, save_args
from src.trainer import MultitaskTrainer, ScheduleStrategy
from src.collator import PosAndMorphologyCollator
from src.dataset import *
from src.data_utils import get_train_dev_test_paths
from torch.optim import AdamW
from argparse import ArgumentParser


def main():

    parser = ArgumentParser()

    # --- data & paths ---
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--checkpoint_dir", type=str,)

    # --- model / tokenizer ---
    parser.add_argument("--encoder_id", type=str,)
    parser.add_argument("--decoder_id", type=str,)

    # --- optimization / training ---
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_backbone_after", type=int, default=3)
    parser.add_argument("--main_metric", type=str, default="pos+morphology::loss")
    parser.add_argument("--seed", type=int, default=42)

    # --- sizing / runtime ---
    parser.add_argument("--estimated_train_size", type=int, default=136629)
    parser.add_argument("--estimated_val_size", type=int, default=23582)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    seed_everything(seed=args.seed)

    splits = get_train_dev_test_paths(args.data_path)

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_id)

    train_dataset = PosAndMorphologyDataset(splits["train"], encoder_tokenizer, decoder_tokenizer, debug_fit=False)
    val_dataset = PosAndMorphologyDataset(splits["dev"], encoder_tokenizer, decoder_tokenizer, debug_fit=False)

    collator = PosAndMorphologyCollator(encoder_tokenizer.pad_token_type_id, -100)

    model = MorphologyClassifier(names_order, name2mapping_from_id, args.encoder_id, args.decoder_id)
    model.train_backbone(False)  # freeze backbone for the beginning


    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_training_steps = args.estimated_train_size // args.batch_size * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
        num_training_steps=num_training_steps,
    )
    trainer = MultitaskTrainer(optimizer, scheduler, schedule_strategy=ScheduleStrategy.BATCH)

    training_log = trainer.train(
        model=model,
        device=args.device,
        train_dataset=train_dataset,
        estimated_train_size=args.estimated_train_size,
        val_dataset=val_dataset,
        estimated_val_size=args.estimated_val_size,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        cpt_dir=args.checkpoint_dir,
        unfreeze_backbone_after=args.unfreeze_backbone_after,
        main_metric=args.main_metric,
    )

    training_log.to_csv(args.checkpoint_dir + "/training_log.csv", sep="\t")
    save_args(args, save_path=args.checkpoint_dir + "/args.json")

if __name__ == "__main__":
    main()
