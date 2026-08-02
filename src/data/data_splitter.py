import random
from pathlib import Path
from src.data.dataset import SyntheticDocumentDataset

def get_synthetic_splits(clean_scans_dir, backgrounds_dir, image_size=(256, 256), seed=42, num_eval_samples=50, train_samples_per_epoch=4000):
    clean_scans = sorted([str(p) for p in Path(clean_scans_dir).glob("*.jpg")])
    
    random.seed(seed)
    random.shuffle(clean_scans)
    
    total_scans = len(clean_scans)
    train_idx = int(0.8 * total_scans)
    val_idx = int(0.9 * total_scans)
    
    train_scans = clean_scans[:train_idx]
    val_scans = clean_scans[train_idx:val_idx]
    test_scans = clean_scans[val_idx:]
    
    print(f"Data Split -> Train: {len(train_scans)} | Val: {len(val_scans)} | Test: {len(test_scans)} source scans.")
    
    train_dataset = SyntheticDocumentDataset(
        clean_scans_paths=train_scans, 
        backgrounds_dir=backgrounds_dir, 
        image_size=image_size, 
        use_degradation=True, 
        freeze_data=False,
        num_samples=train_samples_per_epoch
    )
    
    val_dataset = SyntheticDocumentDataset(
        clean_scans_paths=val_scans, 
        backgrounds_dir=backgrounds_dir, 
        image_size=image_size, 
        use_degradation=True, 
        seed=seed,
        freeze_data=True,
        num_samples=num_eval_samples
    )
    
    test_dataset = SyntheticDocumentDataset(
        clean_scans_paths=test_scans, 
        backgrounds_dir=backgrounds_dir, 
        image_size=image_size, 
        use_degradation=True, 
        seed=seed + 1,
        freeze_data=True,
        num_samples=num_eval_samples
    )
    
    return train_dataset, val_dataset, test_dataset