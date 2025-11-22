

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

class NonastredaDatasetCV(Dataset):
    """
    Nonastreda dataset with tool-based cross-validation support
    Dataset has 10 tools (T1-T10), we do 10-fold CV where:
    - Each fold: train on 9 tools, test on 1 tool
    """
    def __init__(self, root_dir, transform=None, test_tool=None, mode="train"):
        """
        root_dir: root folder of Nonastreda dataset
        transform: torchvision transforms
        test_tool: tool number to use for testing (1-10), None means use all
        mode: "train" or "test"
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.test_tool = test_tool
        
        # Load BOTH classification and regression labels
        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))
        self.labels_reg_df = pd.read_csv(os.path.join(root_dir, "labels_reg.csv"))
        
        # Merge on 'id'
        self.data_df = pd.merge(self.labels_df, self.labels_reg_df, on='id')
        
        # Extract tool number from ID (e.g., "T1R2B3" -> tool 1)
        self.data_df['tool'] = self.data_df['id'].str.extract(r'T(\d+)').astype(int)
        
        # Split based on test_tool
        if test_tool is not None:
            if mode == "train":
                # Train on all tools EXCEPT test_tool
                self.data_df = self.data_df[self.data_df['tool'] != test_tool]
            else:  # mode == "test"
                # Test on ONLY test_tool
                self.data_df = self.data_df[self.data_df['tool'] == test_tool]
        
        self.sample_ids = self.data_df['id'].tolist()
        
        print(f"Loaded {len(self.sample_ids)} samples")
        if test_tool is not None:
            if mode == "train":
                train_tools = sorted(self.data_df['tool'].unique())
                print(f"   Mode: {mode} | Using tools: {train_tools} | Excluded: T{test_tool}")
            else:
                print(f"   Mode: {mode} | Testing on tool: T{test_tool}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def load_image_any_ext(self, folder, sample_id):
        """Load image from folder with any of .png, .jpg, .jpeg"""
        folder_path = os.path.join(self.root_dir, folder)
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Try common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = os.path.join(folder_path, f"{sample_id}{ext}")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img
        
        # If not found, list available files for debugging
        available_files = [f for f in os.listdir(folder_path) 
                          if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))]
        raise FileNotFoundError(
            f"No image found for ID '{sample_id}' in '{folder_path}'\n"
            f"First 5 available files: {available_files[:5]}"
        )
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load images
        chip = self.load_image_any_ext("chip", sample_id)
        tool = self.load_image_any_ext("tool", sample_id)
        work = self.load_image_any_ext("work", sample_id)
        
        # Load scalograms (convert to grayscale)
        scalos = []
        for axis in ['x', 'y', 'z']:
            scalo = self.load_image_any_ext(os.path.join("scal", axis), sample_id)
            # scalo is already transformed (tensor with shape [3, H, W])
            # Average across RGB channels to get grayscale
            scalo_gray = scalo.mean(dim=0)  # [H, W]
            scalos.append(scalo_gray)
        scalos = torch.stack(scalos, dim=0)  # [3, H, W]
        
        # Load spectrograms (convert to grayscale)
        specs = []
        for axis in ['x', 'y', 'z']:
            spec = self.load_image_any_ext(os.path.join("spec", axis), sample_id)
            # spec is already transformed (tensor with shape [3, H, W])
            # Average across RGB channels to get grayscale
            spec_gray = spec.mean(dim=0)  # [H, W]
            specs.append(spec_gray)
        specs = torch.stack(specs, dim=0)  # [3, H, W]
        
        # Load labels
        row = self.data_df[self.data_df['id'] == sample_id].iloc[0]
        
        # CRITICAL FIX: Convert 1-indexed labels [1,2,3] to 0-indexed [0,1,2]
        class_label = torch.tensor(row['tool_label'] - 1, dtype=torch.long)
        
        regression_targets = torch.tensor(
            [row['flank_wear'], row['gaps'], row['overhang']], 
            dtype=torch.float32
        )
        
        # Clean regression targets (handle NaN, inf, extreme values)
        regression_targets = torch.nan_to_num(
            regression_targets, 
            nan=0.0, 
            posinf=500.0, 
            neginf=-500.0
        )
        regression_targets = torch.clamp(regression_targets, min=-500, max=500)
        
        return chip, tool, work, scalos, specs, class_label, regression_targets


# ============================================================================
# HELPER FUNCTIONS FOR CROSS-VALIDATION
# ============================================================================

def create_cv_dataloaders(root_dir, test_tool, batch_size, train_transform, val_transform):
    """
    Create train and test dataloaders for a specific fold
    
    Args:
        root_dir: path to Nonastreda dataset
        test_tool: tool number (1-10) to use for testing
        batch_size: batch size for dataloaders
        train_transform: transforms for training
        val_transform: transforms for validation/testing
    
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = NonastredaDatasetCV(
        root_dir, 
        transform=train_transform, 
        test_tool=test_tool,
        mode="train"
    )
    
    test_dataset = NonastredaDatasetCV(
        root_dir, 
        transform=val_transform, 
        test_tool=test_tool,
        mode="test"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


# ============================================================================
# MAIN K-FOLD CV FUNCTION
# ============================================================================

def run_kfold_cv(config, train_transform, val_transform, num_folds=None):
    """
    Run K-fold cross-validation (configurable number of folds)
    
    Args:
        config: configuration object with all hyperparameters
        train_transform: transforms for training
        val_transform: transforms for validation
        num_folds: number of folds to run (default: uses config.num_folds)
    
    Returns:
        results_per_fold: list of results dictionaries for each fold
    """
    from complete_multitask_standalone import (
        MultiTaskMultimodalModel,
        MultiTaskLoss,
        train_epoch_multitask,
        evaluate_multitask
    )
    import torch
    
    if num_folds is None:
        num_folds = getattr(config, 'num_folds', 10)
    
    results_per_fold = []
    
    print("\n" + "="*70)
    print(f"  ðŸ”„ {num_folds}-FOLD CROSS-VALIDATION")
    print("="*70)
    print(f"  Training on 9 tools, testing on 1 tool (rotating)")
    print(f"  Testing on tools: T1 to T{num_folds}")
    print("="*70)
    
    for fold in range(1, num_folds + 1):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold}/{num_folds} - Testing on Tool T{fold}")
        print(f"{'='*70}")
        
        # Create dataloaders for this fold
        train_loader, test_loader = create_cv_dataloaders(
            config.nonastreda_dir,
            test_tool=fold,
            batch_size=config.batch_size,
            train_transform=train_transform,
            val_transform=val_transform
        )
        
        # Create fresh model for this fold
        model = MultiTaskMultimodalModel(
            embedding_dim=config.embedding_dim,
            fusion_type=config.fusion_type,
            num_classes=config.num_classes,
            num_regression_targets=config.num_reg_targets
        ).to(config.device)
        
        # Setup training
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs
        )
        criterion = MultiTaskLoss(
            alpha_class=config.alpha_class,
            alpha_reg=config.alpha_reg,
            alpha_recon=config.alpha_recon
        )
        
        # Training loop for this fold
        best_accuracy = 0
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Train
            train_loss, _ = train_epoch_multitask(
                model, train_loader, optimizer, criterion, config.device
            )
            
            # Evaluate
            test_results = evaluate_multitask(
                model, test_loader, criterion, config.device
            )
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{config.epochs}: "
                      f"Loss={train_loss:.4f}, "
                      f"Acc={test_results['classification']['accuracy']:.4f}, "
                      f"F1={test_results['classification']['f1']:.4f}, "
                      f"MAE={test_results['regression']['mae']:.2f}")
            
            # Save best model for this fold
            if test_results['classification']['accuracy'] > best_accuracy:
                best_accuracy = test_results['classification']['accuracy']
                best_results = test_results
                patience_counter = 0
                
                # Save checkpoint
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'results': test_results
                }, f'{config.checkpoint_dir}/fold_{fold}_best.pth')
            else:
                patience_counter += 1
            
            scheduler.step()
            
            # Early stopping
            if patience_counter >= config.patience:
                print(f"   â¹ï¸  Early stopping at epoch {epoch+1}")
                break
        
        # Store results for this fold
        results_per_fold.append({
            'fold': fold,
            'test_tool': fold,
            'best_accuracy': best_results['classification']['accuracy'],
            'best_f1': best_results['classification']['f1'],
            'mae': best_results['regression']['mae'],
            'rmse': best_results['regression']['rmse'],
            'full_results': best_results
        })
        
        print(f"\n   âœ… Fold {fold} Complete:")
        print(f"      Accuracy: {best_results['classification']['accuracy']:.4f}")
        print(f"      F1 Score: {best_results['classification']['f1']:.4f}")
        print(f"      MAE:      {best_results['regression']['mae']:.2f} Âµm")
        print(f"      RMSE:     {best_results['regression']['rmse']:.2f} Âµm")
        
        # Clear GPU memory
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results_per_fold