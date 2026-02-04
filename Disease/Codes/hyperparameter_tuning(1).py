"""
Option C - Hyperparameter Tuning
==================================
Tune on fold-0:
- backbone_lr {5e-5, 1e-4}
- head_lr {5e-4, 1e-3}
- λ (lambda_severity) {0.5, 1.0, 2.0}

Total: 12 configurations
Selection: Disease Macro-F1 + Severity Macro-F1
"""

import os
import sys
import json
import time
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))

from option_c import train_option_c, RESULTS_DIR

def hyperparameter_tuning():
    """
    Hyperparameter tuning for Option C on fold-0
    """
    
    # Hyperparameter search space
    backbone_lrs = [5e-5, 1e-4]
    head_lrs = [5e-4, 1e-3]
    lambdas = [0.5, 1.0, 2.0]
    
    # Base configuration
    base_config = {
        'batch_size': 32,
        'warmup_epochs': 5,
        'max_epochs': 25,
        'weight_decay': 1e-4,
        'dropout': 0.25,
        'patience': 5
    }
    
    # Results storage
    tuning_results = []
    
    fold_idx = 0  # Tune on fold-0
    
    print(f"\n{'='*80}")
    print("OPTION C - HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Testing {len(backbone_lrs) * len(head_lrs) * len(lambdas)} configurations on fold-0")
    print(f"Backbone LRs: {backbone_lrs}")
    print(f"Head LRs: {head_lrs}")
    print(f"Lambda values: {lambdas}")
    print(f"{'='*80}\n")
    
    config_num = 0
    for backbone_lr, head_lr, lambda_sev in product(backbone_lrs, head_lrs, lambdas):
        config_num += 1
        
        print(f"\n{'='*80}")
        print(f"Configuration {config_num}/{len(backbone_lrs)*len(head_lrs)*len(lambdas)}")
        print(f"Backbone LR: {backbone_lr}, Head LR: {head_lr}, Lambda: {lambda_sev}")
        print(f"{'='*80}")
        
        # Update configuration
        config = base_config.copy()
        config['backbone_lr'] = backbone_lr
        config['head_lr'] = head_lr
        config['lambda_severity'] = lambda_sev
        
        # Create save directory
        config_name = f"backbone_{backbone_lr}_head_{head_lr}_lambda_{lambda_sev}"
        save_dir = RESULTS_DIR / "hyperparameter_tuning" / config_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Train
            start_time = time.time()
            results = train_option_c(fold_idx, config, save_dir)
            elapsed_time = time.time() - start_time
            
            # Extract metrics
            disease_f1 = results['disease']['f1_macro']
            severity_f1 = results['severity']['f1_macro']
            hierarchical_acc = results['hierarchical_accuracy']
            combined_score = disease_f1 + severity_f1
            
            # Store results
            tuning_results.append({
                'config_name': config_name,
                'backbone_lr': backbone_lr,
                'head_lr': head_lr,
                'lambda_severity': lambda_sev,
                'disease_f1': disease_f1,
                'severity_f1': severity_f1,
                'hierarchical_accuracy': hierarchical_acc,
                'combined_score': combined_score,
                'training_time_minutes': elapsed_time / 60
            })
            
            print(f"\nResults:")
            print(f"  Disease Macro-F1: {disease_f1:.4f}")
            print(f"  Severity Macro-F1: {severity_f1:.4f}")
            print(f"  Combined Score: {combined_score:.4f}")
            print(f"  Hierarchical Accuracy: {hierarchical_acc:.4f}")
            print(f"  Training time: {elapsed_time/60:.2f} minutes")
            
        except Exception as e:
            print(f"\nError in configuration {config_name}: {str(e)}")
            tuning_results.append({
                'config_name': config_name,
                'backbone_lr': backbone_lr,
                'head_lr': head_lr,
                'lambda_severity': lambda_sev,
                'error': str(e)
            })
    
    # Sort by combined score
    valid_results = [r for r in tuning_results if 'combined_score' in r]
    valid_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(valid_results, 1):
        print(f"{i}. {result['config_name']}")
        print(f"   Disease F1: {result['disease_f1']:.4f} | "
              f"Severity F1: {result['severity_f1']:.4f} | "
              f"Combined: {result['combined_score']:.4f}")
    
    # Best configuration
    best_config = valid_results[0]
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Config: {best_config['config_name']}")
    print(f"Backbone LR: {best_config['backbone_lr']}")
    print(f"Head LR: {best_config['head_lr']}")
    print(f"Lambda: {best_config['lambda_severity']}")
    print(f"Disease Macro-F1: {best_config['disease_f1']:.4f}")
    print(f"Severity Macro-F1: {best_config['severity_f1']:.4f}")
    print(f"Combined Score: {best_config['combined_score']:.4f}")
    print(f"Hierarchical Accuracy: {best_config['hierarchical_accuracy']:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    tuning_dir = RESULTS_DIR / "hyperparameter_tuning"
    
    with open(tuning_dir / "tuning_results.json", 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    with open(tuning_dir / "best_config.json", 'w') as f:
        json.dump({
            'backbone_lr': best_config['backbone_lr'],
            'head_lr': best_config['head_lr'],
            'lambda_severity': best_config['lambda_severity'],
            'disease_f1': best_config['disease_f1'],
            'severity_f1': best_config['severity_f1'],
            'combined_score': best_config['combined_score'],
            'hierarchical_accuracy': best_config['hierarchical_accuracy']
        }, f, indent=2)
    
    print(f"Results saved to: {tuning_dir}")
    
    return best_config


if __name__ == "__main__":
    best_config = hyperparameter_tuning()
