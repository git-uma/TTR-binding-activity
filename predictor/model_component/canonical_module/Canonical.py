import sys
import argparse
import pandas as pd
from desalt import desalt
from neutralize import neutralize_charges
from rdkit_util import bringChargeToTail
import re
from rdkit import Chem

def simplify_charge(smiles):
    # 連続する＋や－を簡略化するためのパターン
    pattern = r'\[([^\[\]]*?)([\+\-]{2,})([^\[\]]*?)\]'
    while True:
        matched = re.search(pattern, smiles)
        if matched:
            atom_part = matched.group(1)
            charge_part = matched.group(2)
            remainder_part = matched.group(3)
            # 連続する＋や－の数を数える
            charge_count = len(charge_part)
            simplified_charge = f"{charge_part[0]}{charge_count}"
            new_smiles = f"[{atom_part}{simplified_charge}{remainder_part}]"
            smiles = smiles[:matched.start()] + new_smiles + smiles[matched.end():]
        else:
            break
    return smiles

def process_smiles(smiles):
    try:
        smiles = simplify_charge(smiles)
        smiles = bringChargeToTail(smiles)
        smiles = desalt(smiles)
        smiles = neutralize_charges(smiles)
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}", file=sys.stderr)
        return ""

def main():
    parser = argparse.ArgumentParser(description="Process SMILES from a CSV file.")
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file')
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    if 'SMILES_s' not in data.columns:
        print("The CSV file must contain a 'SMILES' column.")
        sys.exit(1)

    # Process each SMILES string in the CSV file
    data['Processed_SMILES'] = data['SMILES_s'].apply(process_smiles)

    # Save the processed data to a new CSV file
    output_csv_path = args.csv_path.replace('.csv', '_processed.csv')
    data.to_csv(output_csv_path, index=False)

    print(f"Processed SMILES have been saved to {output_csv_path}")

if __name__ == '__main__':
    main()
