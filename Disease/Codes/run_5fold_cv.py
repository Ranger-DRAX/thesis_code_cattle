"""
Option C - 5-Fold Cross-Validation
====================================
Train multi-task model on all 5 folds with best hyperparameters
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from option_c import train_option_c, RESULTS_DIR

def run_5fold_cv():
    """
    Run 5-fold cross-validation with best hyperparameters
    """
    
    # Load best configuration
    tuning_dir = RESULTS_DIR / "hyperparameter_tuning"
    
    try:
        with open(tuning_dir / "best_config.json", 'r') as f:
            best_hp = json.load(f)
        
        print(f"\n{'='*80}")
        print("LOADED BEST HYPERPARAMETERS")
        print(f"{'='*80}")
        print(f"Backbone LR: {best_hp['backbone_lr']}")
        print(f"Head LR: {best_hp['head_lr']}")
        print(f"Lambda: {best_hp['lambda_severity']}")
        print(f"Tuning Combined Score: {best_hp['combined_score']:.4f}")
        print(f"{'='*80}\n")
        
    except FileNotFoundError:
        print("Best config not found. Using default hyperparameters.")
        best_hp = {
            'backbone_lr': 5e-5,
            'head_lr': 1e-3,
            'lambda_severity': 1.0
        }
    
    # Configuration with best hyperparameters
    config = {
        'batch_size': 32,
        'warmup_epochs': 5,
        'max_epochs': 25,
        'backbone_lr': best_hp['backbone_lr'],
        'head_lr': best_hp['head_lr'],
        'lambda_severity': best_hp['lambda_severity'],
        'weight_decay': 1e-4,
        'dropout': 0.25,
        'patience': 5
    }
    
    # Results storage
    cv_results = []
    
    print(f"\n{'='*80}")
    print("OPTION C - 5-FOLD CROSS-VALIDATION")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    
    for fold_idx in range(5):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*80}")
        
        # Create save directory
        save_dir = RESULTS_DIR / "5fold_cv" / f"fold_{fold_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Train
            start_time = time.time()
            results = train_option_c(fold_idx, config, save_dir)
            elapsed_time = time.time() - start_time
            
            # Extract metrics
            fold_result = {
                'fold': fold_idx,
                'disease_accuracy': results['disease']['accuracy'],
                'disease_f1': results['disease']['f1_macro'],
                'severity_accuracy': results['severity']['accuracy'],
                'severity_f1': results['severity']['f1_macro'],
                'hierarchical_accuracy': results['hierarchical_accuracy'],
                'best_epoch': results['training']['best_epoch'],
                'best_combined_f1': results['training']['best_combined_f1'],
                'training_time_minutes': elapsed_time / 60
            }
            
            cv_results.append(fold_result)
            
            print(f"\nFold {fold_idx} Results:")
            print(f"  Disease Acc: {fold_result['disease_accuracy']:.4f}, F1: {fold_result['disease_f1']:.4f}")
            print(f"  Severity Acc: {fold_result['severity_accuracy']:.4f}, F1: {fold_result['severity_f1']:.4f}")
            print(f"  Hierarchical Acc: {fold_result['hierarchical_accuracy']:.4f}")
            print(f"  Training time: {elapsed_time/60:.2f} minutes")
            
            # Save fold summary
            with open(save_dir / "fold_summary.json", 'w') as f:
                json.dump(fold_result, f, indent=2)
            
        except Exception as e:
            print(f"\nError in fold {fold_idx}: {str(e)}")
            cv_results.append({
                'fold': fold_idx,
                'error': str(e)
            })
    
    total_elapsed_time = time.time() - total_start_time
    
    # Compute statistics
    valid_results = [r for r in cv_results if 'error' not in r]
    
    if len(valid_results) > 0:
        metrics = [
            'disease_accuracy', 'disease_f1',
            'severity_accuracy', 'severity_f1',
            'hierarchical_accuracy', 'best_epoch', 'best_combined_f1',
            'training_time_minutes'
        ]
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in valid_results]
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Print summary
        print(f"\n{'='*80}")
        print("5-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}\n")
        
        print("Disease Classification:")
        print(f"  Accuracy:  {stats['disease_accuracy']['mean']:.4f} ± {stats['disease_accuracy']['std']:.4f}")
        print(f"  Macro-F1:  {stats['disease_f1']['mean']:.4f} ± {stats['disease_f1']['std']:.4f}")
        
        print("\nSeverity Classification (diseased only):")
        print(f"  Accuracy:  {stats['severity_accuracy']['mean']:.4f} ± {stats['severity_accuracy']['std']:.4f}")
        print(f"  Macro-F1:  {stats['severity_f1']['mean']:.4f} ± {stats['severity_f1']['std']:.4f}")
        
        print(f"\nHierarchical Accuracy: {stats['hierarchical_accuracy']['mean']:.4f} ± {stats['hierarchical_accuracy']['std']:.4f}")
        
        print("\nTraining Statistics:")
        print(f"  Best epoch: {stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}")
        print(f"  Best combined F1: {stats['best_combined_f1']['mean']:.4f} ± {stats['best_combined_f1']['std']:.4f}")
        print(f"  Avg training time per fold: {stats['training_time_minutes']['mean']:.1f} ± {stats['training_time_minutes']['std']:.1f} minutes")
        print(f"  Total training time: {total_elapsed_time/60:.1f} minutes")
        
        print(f"\n{'='*80}\n")
        
        # Save results
        cv_dir = RESULTS_DIR / "5fold_cv"
        
        with open(cv_dir / "cv_results.json", 'w') as f:
            json.dump({
                'folds': cv_results,
                'statistics': stats,
                'config': config,
                'total_training_time_minutes': total_elapsed_time / 60
            }, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(valid_results)
        df.to_csv(cv_dir / "cv_summary.csv", index=False)
        
        print(f"Results saved to: {cv_dir}")
        
        return stats
    
    else:
        print("\nNo valid results from cross-validation!")
        return None


if __name__ == "__main__":
    stats = run_5fold_cv()
