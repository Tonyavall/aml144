from pathlib import Path

from torchvision import transforms as T


def list_train_images(train_dir, class_to_idx):
    # return (paths, labels) for every training image, ordered by class then filename
    paths, labels = [], []

    for name, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
        for img in sorted((Path(train_dir) / name).iterdir()):
            paths.append(img)
            labels.append(idx)

    return paths, labels


def list_all_test_images(test_dir):
    # every test image is a scored row; sort numerically by integer filename stem
    paths = sorted(Path(test_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    ids = [p.name for p in paths]

    return ids, paths


def build_transform(img_size, mean, std, view):
    # deterministic per-view transforms so tta is reproducible
    bicubic = T.InterpolationMode.BICUBIC

    if view in ("scale_up", "scale_up_hflip"):
        up = img_size + 32
        resize = [T.Resize(up, interpolation=bicubic), T.CenterCrop(img_size)]
    else:
        resize = [T.Resize(img_size, interpolation=bicubic), T.CenterCrop(img_size)]

    flip = (
        [T.RandomHorizontalFlip(p=1.0)] if view in ("hflip", "scale_up_hflip") else []
    )
    tail = [T.ToTensor(), T.Normalize(mean, std)]

    return T.Compose(resize + flip + tail)
