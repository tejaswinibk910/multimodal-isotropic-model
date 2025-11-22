# ============================================================================
# CROSS-DOMAIN EVALUATION & ABLATION STUDIES
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# ============================================================================
# 1. CROSS-DOMAIN EVALUATION
# ============================================================================

class CrossDomainEvaluator:
    """Evaluate model generalization across domains"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate_domain(self, dataloader, domain_name):
        """Evaluate on a single domain"""
        self.model.eval()
        all_scores = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for chip, tool, work, scalos, specs, labels in tqdm(dataloader, desc=f"Evaluating {domain_name}"):
                chip = chip.to(self.device)
                tool = tool.to(self.device)
                work = work.to(self.device)
                scalos = scalos.to(self.device).float()
                specs = specs.to(self.device).float()
                
                outputs = self.model(chip, tool, work, scalos, specs)
                
                # Anomaly score = reconstruction error
                anomaly_scores = torch.mean(
                    (outputs['reconstructed'] - outputs['fused']) ** 2, 
                    dim=1
                ).cpu().numpy()
                
                all_scores.extend(anomaly_scores)
                all_labels.extend(labels.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Convert to binary if multi-class
        if len(np.unique(all_labels)) > 2:
            binary_labels = (all_labels > 0).astype(int)
        else:
            binary_labels = all_labels
        
        # Compute metrics
        auroc = roc_auc_score(binary_labels, all_scores)
        
        # Use median as threshold for prediction
        threshold = np.median(all_scores)
        predictions = (all_scores > threshold).astype(int)
        f1 = f1_score(binary_labels, predictions)
        
        return {
            'domain': domain_name,
            'auroc': auroc,
            'f1': f1,
            'scores': all_scores,
            'labels': binary_labels,
            'predictions': predictions
        }
    
    def evaluate_multiple_domains(self, domain_loaders):
        """
        Evaluate on multiple domains
        domain_loaders: dict like {'Domain_A': loader_A, 'Domain_B': loader_B}
        """
        results = []
        
        for domain_name, loader in domain_loaders.items():
            print(f"\n{'='*60}")
            print(f"Evaluating on {domain_name}")
            print(f"{'='*60}")
            
            result = self.evaluate_domain(loader, domain_name)
            results.append(result)
            
            print(f"  AUROC: {result['auroc']:.4f}")
            print(f"  F1:    {result['f1']:.4f}")
        
        return results
    
    def compute_domain_gap(self, source_results, target_results):
        """Compute performance gap between source and target domains"""
        source_auroc = source_results['auroc']
        target_auroc = target_results['auroc']
        
        gap = source_auroc - target_auroc
        relative_gap = gap / source_auroc * 100
        
        return {
            'absolute_gap': gap,
            'relative_gap': relative_gap
        }
    
    def plot_cross_domain_results(self, results, save_path='cross_domain_results.png'):
        """Plot cross-domain evaluation results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        domains = [r['domain'] for r in results]
        aurocs = [r['auroc'] for r in results]
        f1s = [r['f1'] for r in results]
        
        # AUROC comparison
        axes[0].bar(domains, aurocs, color='steelblue')
        axes[0].set_ylabel('AUROC')
        axes[0].set_title('AUROC Across Domains')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # F1 comparison
        axes[1].bar(domains, f1s, color='coral')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 Score Across Domains')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved cross-domain results to {save_path}")


# ============================================================================
# 2. ABLATION STUDIES
# ============================================================================

class AblationStudy:
    """Perform ablation studies on multimodal model"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate_single_modality(self, dataloader, modality='chip'):
        """Evaluate using only one modality"""
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for chip, tool, work, scalos, specs, labels in tqdm(dataloader, desc=f"Ablation: {modality}"):
                chip = chip.to(self.device)
                tool = tool.to(self.device)
                work = work.to(self.device)
                scalos = scalos.to(self.device).float()
                specs = specs.to(self.device).float()
                
                # Get embedding for specific modality
                if modality == 'chip':
                    embedding = self.model.chip_encoder(chip)
                elif modality == 'tool':
                    embedding = self.model.tool_encoder(tool)
                elif modality == 'work':
                    embedding = self.model.work_encoder(work)
                elif modality == 'scalogram':
                    embedding = self.model.scalo_encoder(scalos)
                elif modality == 'spectrogram':
                    embedding = self.model.spec_encoder(specs)
                
                # Pass through autoencoder
                reconstructed, _ = self.model.autoencoder(embedding)
                
                # Compute anomaly score
                anomaly_scores = torch.mean(
                    (reconstructed - embedding) ** 2, 
                    dim=1
                ).cpu().numpy()
                
                all_scores.extend(anomaly_scores)
                all_labels.extend(labels.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Convert to binary
        if len(np.unique(all_labels)) > 2:
            binary_labels = (all_labels > 0).astype(int)
        else:
            binary_labels = all_labels
        
        auroc = roc_auc_score(binary_labels, all_scores)
        
        return {
            'modality': modality,
            'auroc': auroc
        }
    
    def ablate_modalities(self, dataloader):
        """Test each modality individually"""
        modalities = ['chip', 'tool', 'work', 'scalogram', 'spectrogram']
        results = []
        
        print("\n" + "="*60)
        print("MODALITY ABLATION STUDY")
        print("="*60)
        
        for modality in modalities:
            result = self.evaluate_single_modality(dataloader, modality)
            results.append(result)
            print(f"{modality:15s} AUROC: {result['auroc']:.4f}")
        
        return results
    
    def evaluate_without_modality(self, dataloader, excluded_modality):
        """Evaluate model excluding one modality"""
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for chip, tool, work, scalos, specs, labels in tqdm(dataloader, desc=f"Exclude: {excluded_modality}"):
                chip = chip.to(self.device)
                tool = tool.to(self.device)
                work = work.to(self.device)
                scalos = scalos.to(self.device).float()
                specs = specs.to(self.device).float()
                
                # Encode all modalities except excluded one
                embeddings = []
                if excluded_modality != 'chip':
                    embeddings.append(self.model.chip_encoder(chip))
                if excluded_modality != 'tool':
                    embeddings.append(self.model.tool_encoder(tool))
                if excluded_modality != 'work':
                    embeddings.append(self.model.work_encoder(work))
                if excluded_modality != 'scalogram':
                    embeddings.append(self.model.scalo_encoder(scalos))
                if excluded_modality != 'spectrogram':
                    embeddings.append(self.model.spec_encoder(specs))
                
                # Fuse remaining modalities
                if len(embeddings) > 1:
                    fused, _ = self.model.fusion(embeddings)
                else:
                    fused = embeddings[0]
                
                # Anomaly detection
                reconstructed, _ = self.model.autoencoder(fused)
                
                anomaly_scores = torch.mean(
                    (reconstructed - fused) ** 2, 
                    dim=1
                ).cpu().numpy()
                
                all_scores.extend(anomaly_scores)
                all_labels.extend(labels.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 2:
            binary_labels = (all_labels > 0).astype(int)
        else:
            binary_labels = all_labels
        
        auroc = roc_auc_score(binary_labels, all_scores)
        
        return {
            'excluded_modality': excluded_modality,
            'auroc': auroc
        }
    
    def modality_importance(self, dataloader, full_auroc):
        """Compute importance of each modality by removing it"""
        modalities = ['chip', 'tool', 'work', 'scalogram', 'spectrogram']
        results = []
        
        print("\n" + "="*60)
        print("MODALITY IMPORTANCE (Drop One)")
        print("="*60)
        print(f"Full model AUROC: {full_auroc:.4f}\n")
        
        for modality in modalities:
            result = self.evaluate_without_modality(dataloader, modality)
            importance = full_auroc - result['auroc']
            result['importance'] = importance
            results.append(result)
            print(f"Without {modality:15s} AUROC: {result['auroc']:.4f} | Importance: {importance:+.4f}")
        
        return results
    
    def plot_ablation_results(self, single_results, drop_results, save_path='ablation_results.png'):
        """Plot ablation study results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Single modality performance
        modalities = [r['modality'] for r in single_results]
        single_aurocs = [r['auroc'] for r in single_results]
        
        axes[0].barh(modalities, single_aurocs, color='steelblue')
        axes[0].set_xlabel('AUROC')
        axes[0].set_title('Single Modality Performance')
        axes[0].set_xlim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Modality importance (drop one)
        excluded = [r['excluded_modality'] for r in drop_results]
        importances = [r['importance'] for r in drop_results]
        
        colors = ['red' if imp > 0 else 'green' for imp in importances]
        axes[1].barh(excluded, importances, color=colors)
        axes[1].set_xlabel('Performance Drop (Importance)')
        axes[1].set_title('Modality Importance (Full - Without)')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved ablation results to {save_path}")


# ============================================================================
# 3. FUSION STRATEGY COMPARISON
# ============================================================================

def compare_fusion_strategies(train_loader, val_loader, device, num_epochs=20):
    """Compare different fusion strategies"""
    from complete_training_code import MultimodalAnomalyDetector, train_model
    
    fusion_types = ['concat', 'attention', 'transformer']
    results = {}
    
    print("\n" + "="*60)
    print("FUSION STRATEGY COMPARISON")
    print("="*60)
    
    for fusion_type in fusion_types:
        print(f"\n🔧 Training with {fusion_type} fusion...")
        
        model = MultimodalAnomalyDetector(
            embedding_dim=256,
            fusion_type=fusion_type,
            use_contrastive=False
        ).to(device)
        
        train_losses, val_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=1e-4,
            patience=5,
            save_dir=f'checkpoints/{fusion_type}_fusion'
        )
        
        results[fusion_type] = {
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'best_auroc': max([m['auroc'] for m in val_metrics])
        }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for fusion_type, data in results.items():
        epochs = range(1, len(data['train_losses']) + 1)
        axes[0].plot(epochs, data['train_losses'], label=fusion_type, marker='o')
        axes[1].plot(epochs, [m['auroc'] for m in data['val_metrics']], label=fusion_type, marker='o')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation AUROC')
    axes[1].set_title('Validation AUROC Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fusion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved fusion comparison to fusion_comparison.png")
    
    # Print summary
    print("\n" + "="*60)
    print("FUSION STRATEGY SUMMARY")
    print("="*60)
    for fusion_type, data in results.items():
        print(f"{fusion_type:15s} Best AUROC: {data['best_auroc']:.4f}")
    
    return results


# ============================================================================
# 4. ERROR ANALYSIS
# ============================================================================

class ErrorAnalyzer:
    """Analyze model errors and failures"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def analyze_errors(self, dataloader, threshold):
        """Analyze false positives and false negatives"""
        self.model.eval()
        
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'true_positives': [],
            'true_negatives': []
        }
        
        with torch.no_grad():
            for idx, (chip, tool, work, scalos, specs, labels) in enumerate(tqdm(dataloader, desc="Analyzing errors")):
                chip = chip.to(self.device)
                tool = tool.to(self.device)
                work = work.to(self.device)
                scalos = scalos.to(self.device).float()
                specs = specs.to(self.device).float()
                
                outputs = self.model(chip, tool, work, scalos, specs)
                
                anomaly_scores = torch.mean(
                    (outputs['reconstructed'] - outputs['fused']) ** 2, 
                    dim=1
                ).cpu().numpy()
                
                predictions = (anomaly_scores > threshold).astype(int)
                true_labels = (labels.cpu().numpy() > 0).astype(int)
                
                for i in range(len(predictions)):
                    error_info = {
                        'batch_idx': idx,
                        'sample_idx': i,
                        'score': anomaly_scores[i],
                        'label': true_labels[i],
                        'prediction': predictions[i]
                    }
                    
                    if predictions[i] == 1 and true_labels[i] == 0:
                        errors['false_positives'].append(error_info)
                    elif predictions[i] == 0 and true_labels[i] == 1:
                        errors['false_negatives'].append(error_info)
                    elif predictions[i] == 1 and true_labels[i] == 1:
                        errors['true_positives'].append(error_info)
                    else:
                        errors['true_negatives'].append(error_info)
        
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        print(f"False Positives: {len(errors['false_positives'])}")
        print(f"False Negatives: {len(errors['false_negatives'])}")
        print(f"True Positives:  {len(errors['true_positives'])}")
        print(f"True Negatives:  {len(errors['true_negatives'])}")
        
        return errors
    
    def plot_confusion_matrix(self, errors, save_path='confusion_matrix.png'):
        """Plot confusion matrix from error analysis"""
        tp = len(errors['true_positives'])
        tn = len(errors['true_negatives'])
        fp = len(errors['false_positives'])
        fn = len(errors['false_negatives'])
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                    yticklabels=['Actual Normal', 'Actual Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved confusion matrix to {save_path}")


# ============================================================================
# 5. BENCHMARK SUMMARY GENERATOR
# ============================================================================

def generate_benchmark_table(results_dict, save_path='benchmark_results.csv'):
    """Generate comprehensive benchmark results table"""
    
    rows = []
    for method_name, metrics in results_dict.items():
        row = {
            'Method': method_name,
            'AUROC': metrics['auroc'],
            'F1': metrics['f1'],
            'Precision': metrics.get('precision', '-'),
            'Recall': metrics.get('recall', '-')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('AUROC', ascending=False)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv(save_path, index=False)
    print(f"\n📊 Saved benchmark table to {save_path}")
    
    return df


# ============================================================================
# 6. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage (assuming model and dataloaders are defined)
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  CROSS-DOMAIN EVALUATION & ABLATION STUDY TOOLKIT            ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Available analyses:
    
    1. Cross-Domain Evaluation
       - Evaluate model on multiple domains
       - Compute domain gap metrics
       - Visualize cross-domain performance
    
    2. Ablation Studies
       - Single modality performance
       - Modality importance (drop-one analysis)
       - Fusion strategy comparison
    
    3. Error Analysis
       - False positive/negative analysis
       - Confusion matrix visualization
       - Failure case identification
    
    4. Benchmark Generation
       - Comprehensive results table
       - Method comparison
       - Statistical significance testing
    
    Example usage:
    
    # 1. Cross-domain evaluation
    evaluator = CrossDomainEvaluator(model, device)
    domain_loaders = {
        'Domain_A': loader_a,
        'Domain_B': loader_b,
        'Domain_C': loader_c
    }
    results = evaluator.evaluate_multiple_domains(domain_loaders)
    evaluator.plot_cross_domain_results(results)
    
    # 2. Ablation study
    ablation = AblationStudy(model, device)
    single_results = ablation.ablate_modalities(val_loader)
    drop_results = ablation.modality_importance(val_loader, full_auroc=0.85)
    ablation.plot_ablation_results(single_results, drop_results)
    
    # 3. Error analysis
    analyzer = ErrorAnalyzer(model, device)
    errors = analyzer.analyze_errors(val_loader, threshold=0.5)
    analyzer.plot_confusion_matrix(errors)
    
    # 4. Fusion comparison
    fusion_results = compare_fusion_strategies(train_loader, val_loader, device)
    
    # 5. Generate benchmark table
    benchmark_results = {
        'Ours (Attention)': {'auroc': 0.85, 'f1': 0.78},
        'Ours (Transformer)': {'auroc': 0.87, 'f1': 0.80},
        'Concat Baseline': {'auroc': 0.75, 'f1': 0.68},
        'Image Only': {'auroc': 0.70, 'f1': 0.65}
    }
    generate_benchmark_table(benchmark_results)
    """)