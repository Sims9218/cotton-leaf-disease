import optuna
import torch
from src.model import RobustAttentionGuidedEdgeViT

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    edge_mode = trial.suggest_categorical("edge_mode", ["sobel", "canny"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    model = RobustAttentionGuidedEdgeViT(num_classes=7, edge_mode=edge_mode)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    for epoch in range(3):
        val_acc = evaluate_validation_step()
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_val_acc = max(best_val_acc, val_acc)
        
    return best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    
    df = study.trials_dataframe()
    df.to_csv("hyperparameter_trials_impact.csv", index=False)
    print("Best trial parameters found:", study.best_params)
