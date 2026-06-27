"""
train_local.py

A unified script to download skin cancer datasets from Kaggle, preprocess and map
them to target classes, and train/fine-tune a PyTorch EfficientNet classifier.
"""

import os
import sys
import json
import random
import logging
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load local environment variables
load_dotenv()

# Target classes matching algorithms/cnn_classifier.py
CLASS_NAMES = [
    "melanoma",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "melanocytic_nevi",
    "benign_keratosis",
    "vascular_lesion",
    "petechiae_purpura",
    "leukemia_cutis",
    "ecchymosis",
    "normal_skin"
]
NUM_CLASSES = len(CLASS_NAMES)

DATASETS = {
    "ham10000": "kmader/skin-cancer-mnist-ham10000",
    "skin_cancer_classification": "murtozalikhon/skin-cancer-classification",
    "dermnet": "shubhamgoel27/dermnet",
    "skin_diseases": "ismailpromus/skin-diseases-image-dataset",
    "isic_2019": "andrewmvd/isic-2019"
}

def authenticate_kaggle():
    """Ensure Kaggle API token is available."""
    # Check .env file explicitly
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.startswith("KAGGLE_USERNAME="):
                    os.environ["KAGGLE_USERNAME"] = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("KAGGLE_KEY="):
                    os.environ["KAGGLE_KEY"] = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("KAGGLE_API_TOKEN="):
                    os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip().strip('"')

    home_kaggle = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(home_kaggle):
        logger.info("Kaggle API token found at ~/.kaggle/kaggle.json")
        return True
    
    if os.getenv("KAGGLE_API_TOKEN"):
        logger.info("Kaggle API token found in environment variables/dotenv.")
        return True

    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        logger.info("Kaggle credentials found in environment variables/dotenv.")
        return True
        
    logger.warning("Kaggle API credentials not found in standard paths or .env.")
    print("=" * 60)
    print("Kaggle API credentials are required to download datasets.")
    print("Please go to https://www.kaggle.com/settings, create a new API token,")
    print("and place it at ~/.kaggle/kaggle.json, or add KAGGLE_API_TOKEN to your .env file.")
    print("=" * 60)
    
    username = input("Enter your Kaggle Username (or press Enter to skip): ").strip()
    if username:
        key = input("Enter your Kaggle API Key / Token: ").strip()
        if key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = key
        else:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
        return True
    return False

def download_datasets(data_dir="data"):
    """Download and extract datasets from Kaggle."""
    os.makedirs(data_dir, exist_ok=True)
    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    
    for name, slug in DATASETS.items():
        dest_path = os.path.join(data_dir, name)
        # Quick check if folder exists and is not empty
        if os.path.exists(dest_path) and os.path.isdir(dest_path) and len(os.listdir(dest_path)) > 0:
            logger.info("Dataset '%s' already exists at '%s'. Skipping download.", name, dest_path)
            continue
            
        logger.info("Downloading dataset '%s' (%s) to '%s'...", name, slug, dest_path)
        os.makedirs(dest_path, exist_ok=True)
        try:
            api.dataset_download_files(slug, path=dest_path, unzip=True, quiet=False)
            logger.info("Finished downloading and extracting '%s'.", name)
        except Exception as e:
            logger.error("Failed to download '%s': %s", name, e)
            raise e

def map_datasets(data_dir="data"):
    """Parse each dataset folder and map images to target classes."""
    mapped_images = {cls: [] for cls in CLASS_NAMES}
    data_path = Path(data_dir)
    
    # ── 1. HAM10000 ───────────────────────────────────────────────────────────
    ham_path = data_path / "ham10000"
    if ham_path.exists():
        logger.info("Parsing HAM10000 dataset...")
        metadata_csv = list(ham_path.glob("**/*metadata.csv"))
        if metadata_csv:
            import pandas as pd
            df = pd.read_csv(metadata_csv[0])
            ham_map = {
                'nv': 'melanocytic_nevi',
                'mel': 'melanoma',
                'bkl': 'benign_keratosis',
                'bcc': 'basal_cell_carcinoma',
                'akiec': 'actinic_keratosis',
                'vasc': 'vascular_lesion'
            }
            # Find all image paths
            image_paths = {p.stem: p for p in ham_path.glob("**/*.jpg")}
            for _, row in df.iterrows():
                img_id = row['image_id']
                dx = row['dx']
                if dx in ham_map and img_id in image_paths:
                    target_cls = ham_map[dx]
                    mapped_images[target_cls].append(str(image_paths[img_id]))
        else:
            logger.warning("HAM10000 metadata CSV not found.")

    # ── 2. Skin Cancer Classification ─────────────────────────────────────────
    scc_path = data_path / "skin_cancer_classification"
    if scc_path.exists():
        logger.info("Parsing Skin Cancer Classification dataset...")
        scc_map = {
            'akiec': 'actinic_keratosis',
            'bcc': 'basal_cell_carcinoma',
            'bkl': 'benign_keratosis',
            'mel': 'melanoma',
            'nv': 'melanocytic_nevi',
            'vasc': 'vascular_lesion'
        }
        for img_path in scc_path.glob("**/*.jpg"):
            parent_name = img_path.parent.name.lower()
            if parent_name in scc_map:
                target_cls = scc_map[parent_name]
                mapped_images[target_cls].append(str(img_path))

    # ── 3. DermNet ────────────────────────────────────────────────────────────
    derm_path = data_path / "dermnet"
    if derm_path.exists():
        logger.info("Parsing DermNet dataset...")
        for img_path in derm_path.glob("**/*.jpg"):
            path_str = str(img_path).lower()
            if "melanoma" in path_str:
                mapped_images["melanoma"].append(str(img_path))
            elif "actinic" in path_str:
                mapped_images["actinic_keratosis"].append(str(img_path))
            elif "basal cell" in path_str or "basal-cell" in path_str:
                mapped_images["basal_cell_carcinoma"].append(str(img_path))

    # ── 4. Skin Diseases Image Dataset ────────────────────────────────────────
    sd_path = data_path / "skin_diseases"
    if sd_path.exists():
        logger.info("Parsing Skin Diseases Image Dataset...")
        sd_map = {
            'melanoma': 'melanoma',
            'basal cell': 'basal_cell_carcinoma',
            'nevi': 'melanocytic_nevi',
            'nevus': 'melanocytic_nevi',
            'benign keratosis': 'benign_keratosis',
        }
        for img_path in sd_path.glob("**/*.jpg"):
            parent_name = img_path.parent.name.lower()
            for key, target_cls in sd_map.items():
                if key in parent_name:
                    mapped_images[target_cls].append(str(img_path))
                    break

    # ── 5. ISIC 2019 ──────────────────────────────────────────────────────────
    isic_path = data_path / "isic_2019"
    if isic_path.exists():
        logger.info("Parsing ISIC 2019 dataset...")
        csv_files = list(isic_path.glob("**/*GroundTruth.csv"))
        if csv_files:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            isic_cols = {
                'MEL': 'melanoma',
                'NV': 'melanocytic_nevi',
                'BCC': 'basal_cell_carcinoma',
                'AK': 'actinic_keratosis',
                'BKL': 'benign_keratosis',
                'VASC': 'vascular_lesion'
            }
            # Search for image folders
            img_folders = list(isic_path.glob("**/ISIC_2019_Training_Input"))
            if img_folders:
                img_dir = img_folders[0]
                for _, row in df.iterrows():
                    img_name = row['image']
                    img_file = img_dir / f"{img_name}.jpg"
                    if img_file.exists():
                        for col, target_cls in isic_cols.items():
                            if col in row and row[col] == 1.0:
                                mapped_images[target_cls].append(str(img_file))
                                break
            else:
                logger.warning("ISIC 2019 training input directory not found.")
        else:
            logger.warning("ISIC 2019 ground truth CSV not found.")

    # Print summary
    total_imgs = 0
    for cls, paths in mapped_images.items():
        # Clean paths (make sure they are unique files)
        unique_paths = list(set(paths))
        mapped_images[cls] = unique_paths
        logger.info("Class '%s': %d images mapped.", cls, len(unique_paths))
        total_imgs += len(unique_paths)
    logger.info("Total mapped images across all datasets: %d", total_imgs)
    
    return mapped_images

class SkinLesionDataset(Dataset):
    def __init__(self, class_to_images, transform=None):
        self.samples = []
        self.transform = transform
        
        class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
        for cls, paths in class_to_images.items():
            idx = class_to_idx[cls]
            for path in paths:
                self.samples.append((path, idx))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback if image fails to load (create a dummy image)
            img = Image.new("RGB", (300, 300), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
        return img, label

def train_model(mapped_images, model_arch="efficientnet_b3", epochs=5, batch_size=16, lr=1e-3, max_images_per_class=200, resume=False):
    """Downsample data and run the PyTorch training pipeline."""
    # 1. Downsample dataset
    random.seed(42)
    downsampled = {}
    for cls, paths in mapped_images.items():
        if not paths:
            downsampled[cls] = []
            continue
        if len(paths) > max_images_per_class:
            downsampled[cls] = random.sample(paths, max_images_per_class)
        else:
            downsampled[cls] = list(paths)
        logger.info("Downsampled class '%s' to %d images.", cls, len(downsampled[cls]))

    # Check if we have any data to train on
    total_samples = sum(len(paths) for paths in downsampled.values())
    if total_samples == 0:
        logger.error("No training samples available. Please download and extract datasets first.")
        return

    # 2. Transforms (EfficientNet-B3 uses 300x300, normalized)
    train_transform = T.Compose([
        T.Resize((300, 300)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Split dataset
    full_dataset = SkinLesionDataset(downsampled, transform=train_transform)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    # Apply val transform to validation set
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Dataset summary: %d train samples, %d validation samples.", train_size, val_size)

    # 4. Load model
    import timm
    logger.info("Initializing %s...", model_arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)
    
    model = timm.create_model(model_arch, pretrained=True, num_classes=NUM_CLASSES)
    
    start_epoch = 1
    best_acc = 0.0
    best_model_path = Path("models/dermai_best.pt")
    best_model_path.parent.mkdir(exist_ok=True)
    
    if resume and best_model_path.exists():
        logger.info("Loading existing model checkpoint from %s to resume training...", best_model_path)
        try:
            ckpt = torch.load(best_model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "classifier.bias" in ckpt:
                model.load_state_dict(ckpt)
                logger.info("Successfully loaded model state dict.")
            else:
                logger.warning("Checkpoint found but did not match expected state dict format. Starting from pretrained.")
        except Exception as e:
            logger.error("Failed to load checkpoint to resume: %s", e)
            
        # Try to read starting epoch and best accuracy from model_meta.json
        meta_path = Path("models/model_meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                start_epoch = meta.get("epochs_completed", 0) + 1
                best_acc = meta.get("best_val_acc", 0.0)
                logger.info("Resuming from epoch %d with best validation accuracy %.4f", start_epoch, best_acc)
            except Exception as e:
                logger.warning("Could not read metadata to resume epoch counter: %s", e)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        logger.info("--- Epoch %d / %d ---", epoch, epochs)
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)
            
        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        logger.info("Epoch %d Results: Train Loss = %.4f, Train Acc = %.4f | Val Loss = %.4f, Val Acc = %.4f",
                    epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)
        
        # Save best checkpoint state dict
        if epoch_val_acc >= best_acc:
            best_acc = epoch_val_acc
            logger.info("New best validation accuracy: %.4f. Saving model state dict...", best_acc)
            torch.save(model.state_dict(), str(best_model_path))
            
            # Save meta info
            meta = {
                "model_arch": model_arch,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "class_descriptions": {
                    "melanoma": "Malignant melanoma — high-risk skin cancer. Urgent referral required.",
                    "basal_cell_carcinoma": "Basal cell carcinoma — most common skin cancer. Dermatologist visit within 2 weeks.",
                    "actinic_keratosis": "Actinic keratosis — pre-cancerous lesion from UV damage.",
                    "melanocytic_nevi": "Melanocytic nevus (common mole) — benign.",
                    "benign_keratosis": "Benign keratosis-like lesion — benign.",
                    "vascular_lesion": "Vascular lesion — usually benign.",
                    "petechiae_purpura": "Petechiae/Purpura — minor hemorrhage. Monitor.",
                    "leukemia_cutis": "Leukemia cutis — leukemia infiltration. Urgent referral.",
                    "ecchymosis": "Ecchymosis — bruising or bleeding under the skin.",
                    "normal_skin": "Normal skin — no active lesions detected."
                },
                "checkpoint_total_outputs": NUM_CLASSES,
                "active_trained_outputs": NUM_CLASSES,
                "img_size": 300,
                "best_val_acc": round(best_acc, 4),
                "epochs_completed": epoch,
                "note": f"Locally fine-tuned {model_arch} classifier on Kaggle datasets.",
                "high_risk_classes": ["melanoma", "basal_cell_carcinoma", "leukemia_cutis"],
                "disclaimer": "Educational tool only. Not a medical device."
            }
            with open("models/model_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    logger.info("Training complete! Best validation accuracy: %.4f", best_acc)

def main():
    # Parse CLI arguments if run directly
    import argparse
    parser = argparse.ArgumentParser(description="Download Kaggle datasets and train a skin classifier model.")
    parser.add_argument("--download-only", action="store_true", help="Only download and extract datasets, do not train.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max-images", type=int, default=200, help="Max training images per class for downsampling.")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Timm model architecture to use (e.g. efficientnet_b3).")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory where datasets are stored.")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing models/dermai_best.pt")
    args = parser.parse_args()

    # 1. Setup Auth & Download Datasets
    # We always authenticate and call download_datasets if any dataset is missing,
    # as download_datasets internally checks each specific dataset and only downloads the missing ones.
    datasets_missing = False
    if os.path.exists(args.data_dir) and os.path.isdir(args.data_dir):
        for name in DATASETS.keys():
            dest_path = os.path.join(args.data_dir, name)
            if not (os.path.exists(dest_path) and os.path.isdir(dest_path) and len(os.listdir(dest_path)) > 0):
                datasets_missing = True
                break
    else:
        datasets_missing = True
            
    if not datasets_missing and not args.download_only:
        logger.info("All datasets are already present locally. Skipping Kaggle credentials check and download.")
    else:
        if not authenticate_kaggle():
            logger.error("Authentication failed. Cannot download datasets.")
            sys.exit(1)
        try:
            download_datasets(args.data_dir)
        except Exception as e:
            logger.error("Error during dataset download: %s", e)
            sys.exit(1)

    if args.download_only:
        logger.info("Download completed successfully. Skipping training as requested.")
        sys.exit(0)

    # 3. Class Mapping & Parse
    mapped_images = map_datasets(args.data_dir)

    # 4. Train Model
    train_model(
        mapped_images,
        model_arch=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_images_per_class=args.max_images,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
