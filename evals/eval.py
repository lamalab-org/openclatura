import os
import fire
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, r2_score, 
                             accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from structure_to_iupac.namer import name_smiles
from collections import defaultdict

load_dotenv()
client = OpenAI()

def get_cached_embeddings(text_list, dataset_name, suffix, model="text-embedding-3-large"):
    os.makedirs("embeds", exist_ok=True)
    cache_path = f"embeds/{dataset_name}_{suffix}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f: return pickle.load(f)
    
    print(f"Fetching {suffix} embeddings from OpenAI...")
    embeddings = []
    for i in range(0, len(text_list), 500):
        batch = text_list[i : i + 500]
        res = client.embeddings.create(input=batch, model=model)
        embeddings.extend([r.embedding for r in res.data])
    
    embeds_array = np.array(embeddings)
    with open(cache_path, "wb") as f: pickle.dump(embeds_array, f)
    return embeds_array

def get_scaffolds(smiles_list):
    groups = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        groups.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else "")
    return groups

def run_cv_eval(X, y, groups, label, mode, n_splits=3):
    """
    Performs Scaffold-Grouped Cross-Validation.
    Computes metrics within each fold, then returns the mean and std.
    """
    gkf = GroupKFold(n_splits=n_splits)
    # Store predictions for the final parity/confusion plot
    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))
    
    # List to store metrics calculated PER FOLD
    fold_results = defaultdict(list)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if mode == 'r':
            model = Ridge(alpha=1.0).fit(X_train, y_train)
            p = model.predict(X_test)
            all_preds[test_idx] = p

            # Metric computation WITHIN this fold
            fold_results['mae'].append(mean_absolute_error(y_test, p))
            fold_results['r2'].append(r2_score(y_test, p))
        else:
            model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            p = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
            all_preds[test_idx] = p
            all_probs[test_idx] = prob
            
            # Metric computation WITHIN this fold
            fold_results['acc'].append(accuracy_score(y_test, p))
            fold_results['f1'].append(f1_score(y_test, p))
            fold_results['auc'].append(roc_auc_score(y_test, prob))

    print(f"\n{label} Results ({n_splits}-fold CV):")
    for metric, values in fold_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  - {metric.upper():3}: {mean_val:.4f} ± {std_val:.4f}")
    
    return (all_preds, all_probs) if mode == 'c' else all_preds, fold_results

def plot_results(y_true, smiles_res, iupac_res, mode, output_path="comparison.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (preds_data, metrics), title in zip([ax1, ax2], [smiles_res, iupac_res], ["SMILES", "IUPAC"]):
        if mode == 'r':
            ax.scatter(y_true, preds_data, alpha=0.4, edgecolors='white', linewidth=0.5)
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', alpha=0.8)
            ax.set_title(f"{title} (Avg R²: {np.mean(metrics['r2']):.3f})")
            ax.set_xlabel("Experimental")
            ax.set_ylabel("Predicted")
        else:
            preds, _ = preds_data
            cm = confusion_matrix(y_true, preds)
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f"{title} (Avg AUC: {np.mean(metrics['auc']):.3f})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nVisual comparison saved to: {output_path}")

def main(csv_path, target_col, smiles_col="SMILES", mode='r'):
    dataset_name = os.path.basename(csv_path).split('.')[0]
    df = pd.read_csv(csv_path)[[smiles_col, target_col]].dropna()
    
    print(f"Dataset: {dataset_name} | Generating IUPAC names...")
    processed = []
    for smi, val in zip(df[smiles_col], df[target_col]):
        try:
            name = name_smiles(smi)
            if name: processed.append((smi, name, val))
        except: continue
    
    smiles, iupac, y = zip(*processed)
    y, groups = np.array(y), get_scaffolds(smiles)

    x_smi = get_cached_embeddings(list(smiles), dataset_name, "smiles")
    x_iup = get_cached_embeddings(list(iupac), dataset_name, "iupac")

    print("\n" + "="*40)
    s_preds, s_metrics = run_cv_eval(x_smi, y, groups, "SMILES", mode)
    i_preds, i_metrics = run_cv_eval(x_iup, y, groups, "IUPAC", mode)
    print("="*40)

    plot_results(y, (s_preds, s_metrics), (i_preds, i_metrics), mode)

if __name__ == "__main__":
    fire.Fire(main)