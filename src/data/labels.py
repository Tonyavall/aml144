from pathlib import Path


def build_class_to_idx(train_dir):
    # map class-folder name to integer index, using int(name) for integer-like names
    # ignore hidden/dot directories (e.g. tooling caches that may appear in the data dir)
    names = [
        p.name
        for p in sorted(Path(train_dir).iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]

    if names and all(n.lstrip("-").isdigit() for n in names):
        return {name: int(name) for name in names}
    else:
        names = sorted(names)
    return {name: i for i, name in enumerate(names)}


def idx_to_class(class_to_idx):
    return {i: name for name, i in class_to_idx.items()}
