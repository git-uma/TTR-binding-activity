import argparse
import pandas as pd
from rdkit import Chem


# ==============================================================================
# 外部引数
# ==============================================================================

parser = argparse.ArgumentParser(
    description="Remove invalid SMILES from a CSV file."
)

parser.add_argument(
    "--i",
    required=True,
    help="Input CSV path. The CSV must contain a SMILES column."
)

parser.add_argument(
    "--o",
    required=True,
    help="Output CSV path for valid SMILES."
)

parser.add_argument(
    "--invalid",
    default=None,
    help="Output CSV path for invalid SMILES. Default: same directory as --o with _invalid_smiles.csv suffix."
)

args = parser.parse_args()

input_csv_path = args.i
output_csv_path = args.o

if args.invalid is None:
    if output_csv_path.lower().endswith(".csv"):
        invalid_output_csv_path = output_csv_path[:-4] + "_invalid_smiles.csv"
    else:
        invalid_output_csv_path = output_csv_path + "_invalid_smiles.csv"
else:
    invalid_output_csv_path = args.invalid


# ==============================================================================
# CSV読み込み
# ==============================================================================

df = pd.read_csv(input_csv_path)

df.info()
print("")


# ==============================================================================
# SMILESカラム確認
# ==============================================================================

if "SMILES" not in df.columns:
    raise ValueError("'SMILES' カラムが見つかりません")


# ==============================================================================
# SMILES妥当性チェック
# ==============================================================================

def is_valid_smiles(smiles):
    """
    RDKitでMol化できるSMILESだけTrueにする。
    空欄、NaN、文字列でない値、無効SMILESはFalse。
    """
    if pd.isna(smiles):
        return False

    if not isinstance(smiles, str):
        return False

    smiles = smiles.strip()

    if smiles == "":
        return False

    mol = Chem.MolFromSmiles(smiles)

    return mol is not None


df["is_valid_smiles"] = df["SMILES"].apply(is_valid_smiles)


# ==============================================================================
# 無効SMILESの保存
# ==============================================================================

invalid_smiles_df = df[~df["is_valid_smiles"]].copy()

print(f"変換できなかったSMILESの数: {len(invalid_smiles_df)}")

if len(invalid_smiles_df) > 0:
    print(invalid_smiles_df[["SMILES"]].head(20))

invalid_smiles_df.to_csv(
    invalid_output_csv_path,
    index=False
)

print(f"無効なSMILESを保存しました: {invalid_output_csv_path}")


# ==============================================================================
# 有効SMILESのみ保存
# ==============================================================================

df_clean = df[df["is_valid_smiles"]].copy()

df_clean = df_clean.drop(columns=["is_valid_smiles"])

df_clean["SMILES"] = df_clean["SMILES"].astype(str).str.strip()

df_clean.to_csv(
    output_csv_path,
    index=False
)

print(f"\n有効なSMILESのみを保持したデータを保存しました: {output_csv_path}")
print(f"元データ数: {len(df)}")
print(f"有効SMILES数: {len(df_clean)}")
print(f"無効SMILES数: {len(invalid_smiles_df)}")