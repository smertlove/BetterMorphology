from pathlib import Path
from typing import TypedDict


class SubsetAndPath(TypedDict):
    subset: str
    path: Path


def _iter_dataset_dir(
    datadir: Path | str,
    ignoredir_pref="__",
    valid_subsets={"train", "test", "dev"},
):

    if isinstance(datadir, str):
        datadir = Path(datadir)
    assert datadir.exists()

    for dir_ in sorted(datadir.iterdir(), key=lambda dir_: dir_.name):

        # Ignore root files and dirs starting with ignoredir_pref
        if not dir_.is_dir() or dir_.name.startswith(ignoredir_pref):
            continue

        # Delete all splits with weird naming
        children = [
            subdir_ for subdir_ in dir_.iterdir()
            if subdir_.name in valid_subsets and subdir_.is_dir()
        ]

        for child in children:
            yield dir_.name, child.name, child


def get_train_dev_test_paths(datadir: Path | str,):
    it = _iter_dataset_dir(datadir)

    splits: dict[str, list[SubsetAndPath]] = {
        "train": [],
        "test": [],
        "dev": [],
    }

    for subset, split, dir_ in it:
        for file in dir_.iterdir():
            if file.suffix != ".conllu":
                raise ValueError(f"{file} extension '{file.suffix}' != '.conllu'")
            splits[split].append({"subset": subset, "path": file})

    return splits
