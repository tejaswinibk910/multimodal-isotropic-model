# ============================================================================
# COMPLETE STANDALONE MULTI-TASK MODEL
# Replace complete_training_code.py with this file
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. ENCODER MODULES
# ============================================================================

class ImageEncoder(nn.Module):
    """ResNet-based image encoder"""
    def __init__(self, pretrained=True, embedding_dim=256):
        super().__init__()
        import torchvision.models as models
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embedded = self.projection(features)
        return embedded


class TimeSeriesEncoder(nn.Module):
    """2D CNN encoder for time-series data (scalograms/spectrograms)"""
    def __init__(self, in_channels=3, embedding_dim=256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        embedded = self.projection(features)
        return embedded


# ============================================================================
# 2. FUSION MODULES
# ============================================================================

class AttentionFusion(nn.Module):
    """Attention-based multimodal fusion"""
    def __init__(self, embedding_dim=256, num_modalities=5):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(embedding_dim // 4, 1)
        )
    
    def forward(self, modality_features):
        stacked = torch.stack(modality_features, dim=1)
        attention_scores = self.attention(stacked)
        attention_weights = F.softmax(attention_scores, dim=1)
        fused = (stacked * attention_weights).sum(dim=1)
        return fused, attention_weights


class TransformerFusion(nn.Module):
    """Transformer-based cross-modal fusion"""
    def __init__(self, embedding_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, modality_features):
        stacked = torch.stack(modality_features, dim=1)
        transformed = self.transformer(stacked)
        fused = transformed.mean(dim=1)
        fused = self.output_proj(fused)
        return fused, None


class ConcatFusion(nn.Module):
    """Simple concatenation fusion"""
    def __init__(self, embedding_dim=256, num_modalities=5):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, modality_features):
        concatenated = torch.cat(modality_features, dim=1)
        fused = self.projection(concatenated)
        return fused, None


# ============================================================================
# 3. MULTI-TASK MODEL
# ============================================================================

class MultiTaskMultimodalModel(nn.Module):
    """
    Multimodal model with three tasks:
    1. Classification: 3 classes (sharp, used, dulled)
    2. Regression: 3 targets (flank_wear, gaps, overhang)
    3. Anomaly Detection: reconstruction error
    """
    def __init__(self, embedding_dim=256, fusion_type='attention', 
                 num_classes=3, num_regression_targets=3):
        super().__init__()
        
        # Encoders
        self.chip_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.tool_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.work_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.scalo_encoder = TimeSeriesEncoder(in_channels=3, embedding_dim=embedding_dim)
        self.spec_encoder = TimeSeriesEncoder(in_channels=3, embedding_dim=embedding_dim)
        
        # Fusion
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(embedding_dim, num_modalities=5)
        elif fusion_type == 'transformer':
            self.fusion = TransformerFusion(embedding_dim)
        else:
            self.fusion = ConcatFusion(embedding_dim, num_modalities=5)
        
        # Task-specific heads
        # 1. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # 2. Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_regression_targets)
        )
        
        # 3. Anomaly detection head
        self.autoencoder_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.autoencoder_decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, chip, tool, work, scalos, specs):
        # Encode each modality
        chip_emb = self.chip_encoder(chip)
        tool_emb = self.tool_encoder(tool)
        work_emb = self.work_encoder(work)
        scalo_emb = self.scalo_encoder(scalos)
        spec_emb = self.spec_encoder(specs)
        
        # Fuse modalities
        modality_features = [chip_emb, tool_emb, work_emb, scalo_emb, spec_emb]
        fused, attention_weights = self.fusion(modality_features)
        
        # Task outputs
        class_logits = self.classifier(fused)
        regression_outputs = self.regressor(fused)
        
        latent = self.autoencoder_encoder(fused)
        reconstructed = self.autoencoder_decoder(latent)
        
        return {
            'fused': fused,
            'class_logits': class_logits,
            'regression': regression_outputs,
            'reconstructed': reconstructed,
            'latent': latent,
            'attention_weights': attention_weights,
            'modality_embeddings': {
                'chip': chip_emb,
                'tool': tool_emb,
                'work': work_emb,
                'scalo': scalo_emb,
                'spec': spec_emb
            }
        }


# ============================================================================
# 4. LOSS FUNCTION
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Combined loss for all three tasks"""
    def __init__(self, alpha_class=1.0, alpha_reg=1.0, alpha_recon=0.5):
        super().__init__()
        self.alpha_class = alpha_class
        self.alpha_reg = alpha_reg
        self.alpha_recon = alpha_recon
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(self, outputs, class_labels, regression_targets):
        loss_class = self.classification_loss(outputs['class_logits'], class_labels)
        loss_reg = self.regression_loss(outputs['regression'], regression_targets)
        loss_recon = self.reconstruction_loss(outputs['reconstructed'], outputs['fused'])
        
        total_loss = (
            self.alpha_class * loss_class + 
            self.alpha_reg * loss_reg + 
            self.alpha_recon * loss_recon
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'classification': loss_class.item(),
            'regression': loss_reg.item(),
            'reconstruction': loss_recon.item()
        }


# ============================================================================
# 5. TRAINING FUNCTION
# ============================================================================

def train_epoch_multitask(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'classification': 0, 'regression': 0, 'reconstruction': 0}
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (chip, tool, work, scalos, specs, class_labels, reg_targets) in enumerate(pbar):
        # Move to device
        chip = chip.to(device)
        tool = tool.to(device)
        work = work.to(device)
        scalos = scalos.to(device).float()
        specs = specs.to(device).float()
        class_labels = class_labels.to(device)
        reg_targets = reg_targets.to(device)
        
        # Forward pass
        outputs = model(chip, tool, work, scalos, specs)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, class_labels, reg_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{loss_dict["classification"]:.4f}',
            'reg': f'{loss_dict["regression"]:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


# ============================================================================
# 6. EVALUATION FUNCTION
# ============================================================================

def evaluate_multitask(model, dataloader, criterion, device):
    """Evaluate model on all tasks"""
    model.eval()
    total_loss = 0
    
    all_class_preds = []
    all_class_labels = []
    all_reg_preds = []
    all_reg_targets = []
    all_recon_errors = []
    
    with torch.no_grad():
        for chip, tool, work, scalos, specs, class_labels, reg_targets in tqdm(dataloader, desc="Evaluating"):
            chip = chip.to(device)
            tool = tool.to(device)
            work = work.to(device)
            scalos = scalos.to(device).float()
            specs = specs.to(device).float()
            class_labels = class_labels.to(device)
            reg_targets = reg_targets.to(device)
            
            # Forward pass
            outputs = model(chip, tool, work, scalos, specs)
            
            # Compute loss
            loss, _ = criterion(outputs, class_labels, reg_targets)
            total_loss += loss.item()
            
            # Collect predictions
            class_preds = torch.argmax(outputs['class_logits'], dim=1)
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_labels.extend(class_labels.cpu().numpy())
            
            all_reg_preds.extend(outputs['regression'].cpu().numpy())
            all_reg_targets.extend(reg_targets.cpu().numpy())
            
            recon_error = torch.mean((outputs['reconstructed'] - outputs['fused']) ** 2, dim=1)
            all_recon_errors.extend(recon_error.cpu().numpy())
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error
    
    # Classification metrics
    accuracy = accuracy_score(all_class_labels, all_class_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_class_labels, all_class_preds, average='weighted', zero_division=0
    )
    
    # Regression metrics
    all_reg_preds = np.array(all_reg_preds)
    all_reg_targets = np.array(all_reg_targets)
    mae = mean_absolute_error(all_reg_targets, all_reg_preds)
    rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
    
    results = {
        'loss': total_loss / len(dataloader),
        'classification': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'regression': {
            'mae': mae,
            'rmse': rmse
        },
        'anomaly_detection': {
            'mean_recon_error': np.mean(all_recon_errors),
            'std_recon_error': np.std(all_recon_errors)
        }
    }
    
    return results


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def visualize_attention_weights(model, dataloader, device, num_samples=5, save_path='attention_weights.png'):
    """Visualize attention weights"""
    import matplotlib.pyplot as plt
    
    model.eval()
    samples_visualized = 0
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for chip, tool, work, scalos, specs, labels, _ in dataloader:
            if samples_visualized >= num_samples:
                break
            
            chip = chip.to(device)
            tool = tool.to(device)
            work = work.to(device)
            scalos = scalos.to(device).float()
            specs = specs.to(device).float()
            
            outputs = model(chip, tool, work, scalos, specs)
            
            if outputs['attention_weights'] is not None:
                attention = outputs['attention_weights'][0].cpu().numpy().squeeze()
                
                modality_names = ['Chip', 'Tool', 'Work', 'Scalogram', 'Spectrogram']
                axes[samples_visualized].bar(modality_names, attention)
                axes[samples_visualized].set_ylabel('Attention Weight')
                axes[samples_visualized].set_title(f'Sample {samples_visualized+1} (Label: {labels[0].item()})')
                axes[samples_visualized].grid(True, alpha=0.3)
                
                samples_visualized += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved attention weights to {save_path}")
    plt.close()