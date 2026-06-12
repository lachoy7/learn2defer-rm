"""Build PyTorch DataLoaders for router training."""

from torch.utils.data import DataLoader

from dataset.router_dataset import MODEL_ORDER_DEFAULT, RouterDataset, collate_router
from utils.paths import DEV_FEATURES, TEST_RMBENCH_FEATURES, TRAIN_FEATURES

_LOADERS: tuple[DataLoader, DataLoader, DataLoader] | None = None


def build_router_dataloaders(
    stage1_index: int = 0,
    use_swapped_pd: bool = False,
    train_batch_size: int = 256,
    dev_batch_size: int = 128,
    test_batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(model_order=MODEL_ORDER_DEFAULT, stage1_index=stage1_index, use_swapped_pd=use_swapped_pd)
    train_loader = DataLoader(
        RouterDataset(str(TRAIN_FEATURES), split="train", **common),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_router,
    )
    dev_loader = DataLoader(
        RouterDataset(str(DEV_FEATURES), split="dev", **common),
        batch_size=dev_batch_size,
        shuffle=False,
        collate_fn=collate_router,
    )
    test_loader = DataLoader(
        RouterDataset(str(TEST_RMBENCH_FEATURES), split="test_rmbench", **common),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_router,
    )
    return train_loader, dev_loader, test_loader


def get_router_dataloaders(use_swapped_pd: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    global _LOADERS
    if _LOADERS is None:
        print("Loading router datasets...")
        _LOADERS = build_router_dataloaders(use_swapped_pd=use_swapped_pd)
        train_loader, dev_loader, test_loader = _LOADERS
        print(f"   Train : {len(train_loader.dataset):,} samples")
        print(f"   Dev   : {len(dev_loader.dataset):,} samples")
        print(f"   Test-RM: {len(test_loader.dataset):,} samples")
    return _LOADERS
