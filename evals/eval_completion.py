import os
import re
import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from structure_to_iupac.namer import name_smiles
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
client = OpenAI()

def ask_gpt(prompt, model="gpt-5.4-mini-2026-03-17"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a chemistry expert. Provide ONLY the numeric value. No units, no text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", content)
        return float(match.group()) if match else None
    except:
        return None

def process_row(row, smiles_col, target_col, property_name, template, few_shot_smi, few_shot_iup):
    smi = row[smiles_col]
    actual = row[target_col]
    
    try:
        iupac_name = name_smiles(smi)
    except:
        iupac_name = None
    if not iupac_name: return None

    # Prompt construction
    smi_context = "\n".join([f"Input: {ex['repr']} -> {property_name}: {ex['val']}" for ex in few_shot_smi])
    prompt_smi = f"Examples:\n{smi_context}\n\nTask: {template.format(prop=property_name, repr=smi)}"

    iup_context = "\n".join([f"Input: {ex['repr']} -> {property_name}: {ex['val']}" for ex in few_shot_iup])
    prompt_iup = f"Examples:\n{iup_context}\n\nTask: {template.format(prop=property_name, repr=iupac_name)}"

    pred_smi = ask_gpt(prompt_smi)
    pred_iup = ask_gpt(prompt_iup)

    if pred_smi is not None and pred_iup is not None:
        return {"actual": actual, "ae_smi": abs(actual - pred_smi), "ae_iup": abs(actual - pred_iup)}
    return None

def main(
    csv_path, 
    target_col, 
    property_name="melting point in Kelvin", 
    smiles_col="SMILES", 
    limit=55, 
    workers=15,
    template="The {prop} of the molecule {repr} is"
):
    # Load and clean
    df = pd.read_csv(csv_path).dropna(subset=[smiles_col, target_col])
    # BULLETPROOF: Ensure no duplicate molecules leak from context into test
    df = df.drop_duplicates(subset=[smiles_col])
    
    print(f"Loaded {len(df)} unique molecules. Preparing 5-shot context...")
    
    few_shot_smi, few_shot_iup = [], []
    idx = 0
    while len(few_shot_smi) < 5 and idx < len(df):
        s, v = df.iloc[idx][smiles_col], df.iloc[idx][target_col]
        try:
            name = name_smiles(s)
            if name:
                few_shot_smi.append({"repr": s, "val": v})
                few_shot_iup.append({"repr": name, "val": v})
        except: pass
        idx += 1

    eval_df = df.iloc[idx:].head(limit - 5)
    print(f"🚀 Processing {len(eval_df)} samples in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_row, row, smiles_col, target_col, property_name, template, few_shot_smi, few_shot_iup) 
            for _, row in eval_df.iterrows()
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res: results.append(res)

    res_df = pd.DataFrame(results)
    m_smi, m_iup = res_df['ae_smi'].mean(), res_df['ae_iup'].mean()

    # Plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(res_df['ae_smi'], fill=True, label=f'SMILES (MAE: {m_smi:.1f})', color='blue')
    sns.kdeplot(res_df['ae_iup'], fill=True, label=f'IUPAC (MAE: {m_iup:.1f})', color='orange')
    plt.title(f"MAE Distribution for {property_name} (5-shot)")
    plt.xlabel("Absolute Error")
    plt.legend()
    plt.savefig(f"mae_dist_tm.png", dpi=300)
    print(f"\nFinal MAE -> SMILES: {m_smi:.2f} | IUPAC: {m_iup:.2f}")

if __name__ == "__main__":
    fire.Fire(main)