import pandas as pd
import pickle
import joblib
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import re
import sys
import datetime
from tqdm.notebook import tqdm
import warnings
from mordred import Calculator, descriptors

if not hasattr(np, 'float'):
    np.float = float

warnings.filterwarnings('ignore')

try:
    # canonical_module のインポート
    # この部分は、実際のファイルパスと構造に合わせて調整してください。
    # 例: canonical_module フォルダがスクリプトと同じ階層にある場合
    # sys.path.append(os.path.join(os.path.dirname(__file__), 'canonical_module'))
    from model_component.canonical_module.desalt import desalt
    from model_component.canonical_module.neutralize import neutralize_charges
    from model_component.canonical_module.rdkit_util import bringChargeToTail
    from rdkit import Chem
except ImportError as e:
    print(f"エラー: canonical_module のインポート中に問題が発生しました: {e}", file=sys.stderr)
    print("desalt.py, neutralize.py, rdkit_util.py が含まれる 'canonical_module' ディレクトリが正しく配置されているか、および RDKit がインストールされているか確認してください。", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# 設定
# ==============================================================================
base_dir = "model_component/ML_data"
lgbm_model_dir = "model_component/lgbm_model"
input_dir = "./"

# ------------------------------------------------------------------------------
# 外部引数
# input_smiles_csv_path だけ外部から指定できるようにする
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Predict TTR binding activity from an input SMILES CSV file."
)

parser.add_argument(
    "--i",
    default=os.path.join(input_dir, "PhytoHub.csv"),
    help="Path to input CSV file containing a SMILES column. Default: ./PhytoHub.csv"
)

args = parser.parse_args()

# 訓練データパス
csv_path_train_nc = os.path.join(base_dir, "tox24_t_mord.csv")
csv_path_train_c = os.path.join(base_dir, "tox24_t_c_mord.csv")

# 予測対象SMILESデータパス
input_smiles_csv_path = args.i

print(f"Input SMILES CSV: {input_smiles_csv_path}")

# モデルと記述子リストのパス
lgbm_model_path = os.path.join(lgbm_model_dir, "lgbm_model.pkl")
selected_descriptors_pkl_path = os.path.join(base_dir, "selected_descriptors.pkl")

# 出力ディレクトリ設定
main_output_dir = "result"
timestamp_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_full_dir = os.path.join(main_output_dir, timestamp_dir_name)

os.makedirs(output_full_dir, exist_ok=True)

# 出力ファイル名プレフィックス
output_csv_prefix = "predictions"
output_histogram_name = "predicted_y_histogram.png"
output_mordred_descriptors_name = "mordred_descriptors.csv"

# ==============================================================================
# IDカラムと目的変数カラムの設定
# ==============================================================================
train_identifier_columns = ['N'] # 訓練データ用IDカラム
predict_identifier_columns = ['Chemical Name'] # 予測データ用IDカラム
y_column = 'y' # 目的変数カラム

# AD判定ハイパーパラメータ
N_NEIGHBORS_AD = 5 # AD判定に使用する近傍点数 (k)

# ==============================================================================
# グローバル変数 (Mordred記述子名キャッシュ)
# ==============================================================================
_global_mordred_descriptor_names = None

def get_all_mordred_descriptor_names():
    """Mordredの全記述子名を取得し、キャッシュする関数"""
    global _global_mordred_descriptor_names
    if _global_mordred_descriptor_names is None:
        calc = Calculator(descriptors, ignore_3D=True)
        dummy_mol = Chem.MolFromSmiles("CCO")
        if dummy_mol is None:
            print("警告: ダミーSMILES 'CCO' のMol変換に失敗しました。", file=sys.stderr)
            return []

        try:
            dummy_df = calc.pandas([dummy_mol])
            _global_mordred_descriptor_names = dummy_df.columns.tolist()
        except Exception as e:
            print(f"警告: ダミー分子での記述子計算（calc.pandas）に失敗しました。エラー: {e}", file=sys.stderr)
            print("すべてのMordred記述子名を取得できませんでした。", file=sys.stderr)
            _global_mordred_descriptor_names = []
            
    return _global_mordred_descriptor_names

def calculate_mordred_user_style(df_input, smiles_col, prefix=None, identifier_column_for_return=None):
    """
    指定されたSMILESカラムからMordred記述子を計算する関数。
    計算エラーや非数値は0に置換される。
    """
    mols = [Chem.MolFromSmiles(sm) if isinstance(sm, str) and sm != "" else None 
            for sm in tqdm(df_input[smiles_col], desc=f"Converting {smiles_col} to Mol")]

    calc = Calculator(descriptors, ignore_3D=True)
    desc_df = calc.pandas(mols)

    # Mordredの計算結果を数値に変換し、エラーをNaN、NaNを0に変換
    desc_df = desc_df.applymap(lambda x: x if isinstance(x, (int, float, np.number, np.float64)) or pd.isna(x) else np.nan)
    desc_df = desc_df.fillna(0)

    if prefix:
        desc_df = desc_df.add_prefix(prefix)
            
    if identifier_column_for_return and identifier_column_for_return[0] in df_input.columns:
        result_df = pd.DataFrame(index=df_input.index)
        result_df[identifier_column_for_return[0]] = df_input[identifier_column_for_return[0]].copy()
        
        final_df = pd.concat([result_df, desc_df], axis=1)
        
        if not pd.api.types.is_numeric_dtype(final_df[identifier_column_for_return[0]]):
            final_df[identifier_column_for_return[0]] = final_df[identifier_column_for_return[0]].astype(str)

        return final_df
    else:
        return desc_df.reset_index(drop=True)

# ==============================================================================
# ユーティリティ関数 (SMILESカノニカル化)
# ==============================================================================
def simplify_charge(smiles):
    """SMILES文字列内の電荷表記を簡略化する（例: [N++] -> [N+2]）"""
    pattern = r'\[([^\[\]]*?)([\+\-]{2,})([^\[\]]*?)\]'
    while True:
        matched = re.search(pattern, smiles)
        if matched:
            atom_part = matched.group(1)
            charge_part = matched.group(2)
            remainder_part = matched.group(3)
            charge_count = len(charge_part)
            simplified_charge = f"{charge_part[0]}{charge_count}"
            new_smiles = f"[{atom_part}{simplified_charge}{remainder_part}]"
            smiles = smiles[:matched.start()] + new_smiles + smiles[matched.end():]
        else:
            break
    return smiles

def chain_canonical_functions(smiles):
    """複数のカノニカル化関数を連結して適用する"""
    if pd.isna(smiles) or smiles == "":
        return ""
    
    s = smiles
    
    # 簡略化された電荷表記を処理の前に適用
    s = simplify_charge(s)  
    
    s = bringChargeToTail(s)
    s = desalt(s)
    s = neutralize_charges(s)

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def process_smiles(smiles):
    """単一のSMILES文字列をカノニカル化するラッパー関数"""
    if pd.isna(smiles) or smiles == "":
        return ""
    return chain_canonical_functions(smiles)

# ==============================================================================
# 1. 訓練データ読み込みと結合
# ==============================================================================
print("--- 訓練データ読み込みを開始 ---")
try:
    df_train_nc = pd.read_csv(csv_path_train_nc)
    df_train_c = pd.read_csv(csv_path_train_c)
except FileNotFoundError as e:
    print(f"エラー: 訓練用のCSVファイルが見つかりません: {e.filename}", file=sys.stderr)
    sys.exit(1)

# ncデータフレームの処理
nc_cols_to_prefix_train = [col for col in df_train_nc.columns if col not in train_identifier_columns + [y_column, "SMILES"]]
df_train_nc_features_prefixed = df_train_nc[nc_cols_to_prefix_train].rename(columns={col: f'nc_{col}' for col in nc_cols_to_prefix_train})
df_train_nc_processed = pd.concat([df_train_nc[train_identifier_columns + [y_column]], df_train_nc_features_prefixed], axis=1)
if "SMILES" in df_train_nc.columns:
    df_train_nc_processed['SMILES'] = df_train_nc['SMILES']

# cデータフレームの処理
c_cols_to_prefix_train = [col for col in df_train_c.columns if col not in train_identifier_columns + ["SMILES"]]
df_train_c_features_prefixed = df_train_c[c_cols_to_prefix_train].rename(columns={col: f'c_{col}' for col in c_cols_to_prefix_train})
df_train_c_processed = pd.concat([df_train_c[train_identifier_columns], df_train_c_features_prefixed], axis=1)
if "SMILES" in df_train_c.columns:
    df_train_c_processed.rename(columns={'SMILES': 'SMILES_c_original'}, inplace=True)  

# データフレームの結合
df_train_combined = pd.merge(df_train_nc_processed, df_train_c_processed, on=train_identifier_columns, how='inner', suffixes=('_nc', '_c'))

# SMILESカラムの統合
if 'SMILES_nc' in df_train_combined.columns and 'SMILES_c_original' in df_train_combined.columns:
    df_train_combined.drop(columns=['SMILES_c_original'], inplace=True)  
    df_train_combined.rename(columns={'SMILES_nc': 'SMILES'}, inplace=True)
elif 'SMILES_nc' in df_train_combined.columns:
    df_train_combined.rename(columns={'SMILES_nc': 'SMILES'}, inplace=True)
elif 'SMILES_c_original' in df_train_combined.columns:
    df_train_combined.rename(columns={'SMILES_c_original': 'SMILES'}, inplace=True)

if 'SMILES' not in df_train_combined.columns:
    print("警告: 訓練データにSMILESカラムが見つかりません。これはAD判定のSMILES表示に影響する可能性があります。", file=sys.stderr)

if len(df_train_combined) == 0:
    print("エラー: 訓練結合後のデータフレームが空です。処理を中止します。", file=sys.stderr)
    sys.exit(1)
print(f"訓練結合後のデータフレーム形状: {df_train_combined.shape}")

# ==============================================================================
# 2. 予測対象SMILESデータの読み込みとカノニカル化
# ==============================================================================
print(f"--- 予測対象SMILESファイル '{input_smiles_csv_path}' を読み込み中 ---")
try:
    df_smiles_input = pd.read_csv(input_smiles_csv_path)
    if 'SMILES' not in df_smiles_input.columns:
        raise ValueError("入力CSVファイルには 'SMILES' カラムが必要です。")
    
    if predict_identifier_columns[0] not in df_smiles_input.columns:
        print(f"警告: 入力CSVに '{predict_identifier_columns[0]}' カラムがありません。自動で連番IDを生成します。", file=sys.stderr)
        df_smiles_input[predict_identifier_columns[0]] = range(1, len(df_smiles_input) + 1)
        
    df_predict_base = df_smiles_input.copy()

    df_predict_base['SMILES_original'] = df_predict_base['SMILES']

    print("SMILESのカノニカル化を実行中...")
    df_predict_base['SMILES_canonical'] = [process_smiles(s) 
                                           for s in tqdm(df_predict_base['SMILES'], desc="Canonicalizing SMILES")]

    failed_canonicalization = df_predict_base[df_predict_base['SMILES_canonical'] == ""]
    if not failed_canonicalization.empty:
        print(f"警告: {len(failed_canonicalization)} 個のSMILESのカノニカル化に失敗しました。これらのSMILESは予測から除外されます。", file=sys.stderr)
        # 失敗したSMILESの最初の5件を表示
        print(failed_canonicalization[[predict_identifier_columns[0], 'SMILES_original']].head().to_string(), file=sys.stderr)  
    
    # カノニカル化に成功したSMILESのみを抽出
    df_predict_base = df_predict_base[df_predict_base['SMILES_canonical'] != ""].copy()
    if df_predict_base.empty:
        print("エラー: 有効なSMILESがありません。処理を中止します。", file=sys.stderr)
        sys.exit(1)

    print(f"処理対象のSMILES数: {len(df_predict_base)}")

except FileNotFoundError:
    print(f"エラー: 入力SMILESファイル '{input_smiles_csv_path}' が見つかりません。パスを確認してください。", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"エラー: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"SMILESファイルの読み込みまたはカノニカル化中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# 3. Mordred記述子の計算 (カノニカル化前と後)
# ==============================================================================
print("--- Mordred記述子の計算を開始 ---")

all_mordred_descriptor_names = get_all_mordred_descriptor_names()
if not all_mordred_descriptor_names:
    print("エラー: Mordred記述子名を初期化できませんでした。処理を中止します。", file=sys.stderr)
    sys.exit(1)
print(f"Mordredから取得した全記述子数 (ユニーク): {len(all_mordred_descriptor_names)}")

print("カノニカル化前SMILESの記述子を計算中...")
df_mordred_original = calculate_mordred_user_style(df_predict_base.copy(), 'SMILES_original', prefix='nc_', identifier_column_for_return=predict_identifier_columns)
print(f"カノニカル化前記述子の形状: {df_mordred_original.shape}")

print("カノニカル化後SMILESの記述子を計算中...")
df_mordred_canonical = calculate_mordred_user_style(df_predict_base.copy(), 'SMILES_canonical', prefix='c_', identifier_column_for_return=predict_identifier_columns)
print(f"カノニカル化後記述子の形状: {df_mordred_canonical.shape}")

# 予測データとMordred記述子を結合
df_predict_combined = pd.merge(df_predict_base, df_mordred_original, on=predict_identifier_columns[0], how='left')
df_predict_combined = pd.merge(df_predict_combined, df_mordred_canonical, on=predict_identifier_columns[0], how='left')

print(f"Mordred記述子マージ後のデータフレーム形状: {df_predict_combined.shape}")

print(f"\n--- Mordred記述子計算結果を '{output_mordred_descriptors_name}' に保存中 ---")

# 出力用DFのSMILESカラムが、元のSMILES_original/canonicalカラムを指すように調整
if 'SMILES_original' not in df_mordred_original.columns and 'SMILES_original' in df_predict_base.columns:
    df_mordred_original = pd.merge(df_mordred_original, df_predict_base[[predict_identifier_columns[0], 'SMILES_original']], 
                                    on=predict_identifier_columns[0], how='left')

if 'SMILES_canonical' not in df_mordred_canonical.columns and 'SMILES_canonical' in df_predict_base.columns:
    df_mordred_canonical = pd.merge(df_mordred_canonical, df_predict_base[[predict_identifier_columns[0], 'SMILES_canonical']],
                                     on=predict_identifier_columns[0], how='left')

# 出力用のMordred記述子DFを結合
df_mordred_combined_output = pd.merge(df_mordred_original, df_mordred_canonical, 
                                      on=predict_identifier_columns[0], 
                                      how='outer',
                                      suffixes=('_original_mord', '_canonical_mord'))

# 出力カラムの順序を整理
output_cols_order = [predict_identifier_columns[0]]
if 'SMILES_original' in df_mordred_combined_output.columns:
    output_cols_order.append('SMILES_original')
if 'SMILES_canonical' in df_mordred_combined_output.columns:
    output_cols_order.append('SMILES_canonical')

for col in df_mordred_combined_output.columns:
    if col not in output_cols_order:
        output_cols_order.append(col)

df_mordred_combined_output = df_mordred_combined_output[output_cols_order]

mordred_output_path = os.path.join(output_full_dir, output_mordred_descriptors_name)
try:
    df_mordred_combined_output.to_csv(mordred_output_path, index=False)
    print(f"Mordred記述子計算結果を '{mordred_output_path}' に保存しました。")
except Exception as e:
    print(f"エラー: Mordred記述子計算結果のCSV保存中にエラーが発生しました: {e}", file=sys.stderr)

# ==============================================================================
# 4. selected_descriptors.pkl から特徴量リストをロード
# ==============================================================================
print(f"--- '{selected_descriptors_pkl_path}' から選択記述子をロード中 ---")
SELECTED_DESCRIPTORS = []
try:
    with open(selected_descriptors_pkl_path, 'rb') as f:
        SELECTED_DESCRIPTORS = pickle.load(f)
    print(f"ロードされた記述子数: {len(SELECTED_DESCRIPTORS)}")
except FileNotFoundError:
    print(f"エラー: '{selected_descriptors_pkl_path}' が見つかりません。ファイルパスを確認してください。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"エラー: '{selected_descriptors_pkl_path}' の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)

# 予測データにおける欠落記述子のチェックと除外
missing_descriptors_predict = [col for col in SELECTED_DESCRIPTORS if col not in df_predict_combined.columns]
if missing_descriptors_predict:
    print(f"警告: 選択記述子リストに以下のカラムが含まれていますが、結合後の予測データフレームに存在しません: {missing_descriptors_predict[:10]} ... (最初の10個を表示)", file=sys.stderr)
    print(f"欠落している記述子の総数: {len(missing_descriptors_predict)}", file=sys.stderr)
    
    SELECTED_DESCRIPTORS = [col for col in SELECTED_DESCRIPTORS if col in df_predict_combined.columns]
    print(f"存在する記述子のみを使用します。現在の記述子数: {len(SELECTED_DESCRIPTORS)}")

if not SELECTED_DESCRIPTORS:
    print("エラー: 有効な選択記述子が見つかりませんでした。モデル構築とAD判定を中止します。", file=sys.stderr)
    sys.exit(1)

# 訓練データにおける欠落記述子のチェックと除外
missing_descriptors_train = [col for col in SELECTED_DESCRIPTORS if col not in df_train_combined.columns]
if missing_descriptors_train:
    print(f"警告: 選択記述子リストに以下のカラムが含まれていますが、訓練結合データフレームに存在しません: {missing_descriptors_train[:10]} ...", file=sys.stderr)
    print(f"欠落している訓練記述子の総数: {len(missing_descriptors_train)}", file=sys.stderr)
    df_train_combined = df_train_combined.drop(columns=missing_descriptors_train, errors='ignore')
    SELECTED_DESCRIPTORS = [col for col in SELECTED_DESCRIPTORS if col in df_train_combined.columns]
    print(f"訓練データに存在する記述子のみを使用します。現在の記述子数: {len(SELECTED_DESCRIPTORS)}")

# NaN値の除去（予測とAD計算に必要な行のみ）
initial_rows_train = len(df_train_combined)
df_train_combined.dropna(subset=SELECTED_DESCRIPTORS, inplace=True)
if len(df_train_combined) < initial_rows_train:
    print(f"訓練データから{initial_rows_train - len(df_train_combined)}行のNaNを含むサンプルを除去しました。")

initial_rows_predict = len(df_predict_combined)
df_predict_combined.dropna(subset=SELECTED_DESCRIPTORS, inplace=True)
if len(df_predict_combined) < initial_rows_predict:
    print(f"予測データから{initial_rows_predict - len(df_predict_combined)}行のNaNを含むサンプルを除去しました。")

if len(df_train_combined) == 0:
    print("エラー: NaN除去後、訓練データが空です。AD判定を行えません。", file=sys.stderr)
    sys.exit(1)
if len(df_predict_combined) == 0:
    print("エラー: NaN除去後、予測データが空です。予測とAD判定を行えません。", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# 5. LightGBMモデルのロードと予測
# ==============================================================================
print(f"--- LightGBMモデル '{lgbm_model_path}' をロード中 ---")
try:
    model = joblib.load(lgbm_model_path)
    X_predict = df_predict_combined[SELECTED_DESCRIPTORS].copy()
    predictions = model.predict(X_predict)
    df_predict_combined['predicted_y'] = predictions
    print("予測が完了しました。")
except FileNotFoundError:
    print(f"エラー: LightGBMモデルファイルが見つかりません: '{lgbm_model_path}'", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"エラー: LightGBMモデルのロードまたは予測中にエラーが発生しました: {e}", file=sys.stderr)
    df_predict_combined['predicted_y'] = np.nan # エラー時はNaNで埋める

# ==============================================================================
# 6. AD (Applicability Domain) 判定の実行
#    Nearest Neighbors - 相乗平均の対数
#    訓練データの95%範囲を使ってApplicabilityを計算する版
# ==============================================================================
print("\n--- AD (Applicability Domain) 判定中 (Nearest Neighbors - 95% interval) ---")

def calculate_log_gmean_distances(data_points, neighbors_model, k, skip_self=False):
    """
    データポイントからk個の最近傍点までの距離の対数の算術平均を計算する。

    skip_self=True:
        訓練データ自身のAD score計算用。
        0番目の距離は自分自身なので除外する。

    skip_self=False:
        予測データのAD score計算用。
        予測データは訓練データ自身ではないので、
        最も近い訓練データからk個をそのまま使う。
    """
    distances, _ = neighbors_model.kneighbors(data_points)

    if skip_self:
        used_distances = distances[:, 1:(k + 1)]
    else:
        used_distances = distances[:, 0:k]

    log_distances = np.log(used_distances + 1e-10)
    ad_scores = np.mean(log_distances, axis=1)

    return ad_scores


# ------------------------------------------------------------------------------
# 6-1. スケーリング
# ------------------------------------------------------------------------------

scaler_ad = StandardScaler()

X_train_scaled_ad = scaler_ad.fit_transform(
    df_train_combined[SELECTED_DESCRIPTORS]
)

X_predict_scaled_ad = scaler_ad.transform(
    df_predict_combined[SELECTED_DESCRIPTORS]
)


# ------------------------------------------------------------------------------
# 6-2. NearestNeighborsモデルの構築
# ------------------------------------------------------------------------------

# 訓練データ自身のAD scoreでは、自分自身を除くため k+1
nn_ad_model_train = NearestNeighbors(
    n_neighbors=N_NEIGHBORS_AD + 1,
    metric="euclidean"
)

nn_ad_model_train.fit(X_train_scaled_ad)

# 予測データでは、訓練データ中の近傍をそのままk個使うため k
nn_ad_model_predict = NearestNeighbors(
    n_neighbors=N_NEIGHBORS_AD,
    metric="euclidean"
)

nn_ad_model_predict.fit(X_train_scaled_ad)


# ------------------------------------------------------------------------------
# 6-3. 訓練データのAD scoreを計算
# ------------------------------------------------------------------------------

train_ad_scores = calculate_log_gmean_distances(
    data_points=X_train_scaled_ad,
    neighbors_model=nn_ad_model_train,
    k=N_NEIGHBORS_AD,
    skip_self=True
)

train_ad_scores_valid = train_ad_scores[
    np.isfinite(train_ad_scores)
]

if len(train_ad_scores_valid) == 0:
    print("エラー: 有効な訓練AD scoreがありません。", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------------------
# 6-4. 訓練データの95%範囲を計算
# ------------------------------------------------------------------------------

train_lower_95 = np.percentile(
    train_ad_scores_valid,
    2.5
)

train_upper_95 = np.percentile(
    train_ad_scores_valid,
    97.5
)

if train_upper_95 == train_lower_95:
    print("エラー: train_lower_95 と train_upper_95 が同じ値です。Applicabilityを計算できません。", file=sys.stderr)
    sys.exit(1)

print(f"train_lower_95: {train_lower_95:.6f}")
print(f"train_upper_95: {train_upper_95:.6f}")


# ------------------------------------------------------------------------------
# 6-5. 予測データの s value を計算
# ------------------------------------------------------------------------------

predict_ad_scores = calculate_log_gmean_distances(
    data_points=X_predict_scaled_ad,
    neighbors_model=nn_ad_model_predict,
    k=N_NEIGHBORS_AD,
    skip_self=False
)

df_predict_combined["s value"] = predict_ad_scores


# ------------------------------------------------------------------------------
# 6-6. Applicabilityを計算
#
# s value が小さい:
#   訓練データに近い
#   Applicabilityが高い
#
# s value が大きい:
#   訓練データから遠い
#   Applicabilityが低い
# ------------------------------------------------------------------------------

df_predict_combined["Applicability_raw"] = (
    (train_upper_95 - df_predict_combined["s value"]) /
    (train_upper_95 - train_lower_95)
)

df_predict_combined["Applicability"] = (
    df_predict_combined["Applicability_raw"]
    .clip(0, 1)
)

df_predict_combined["train_lower_95"] = train_lower_95
df_predict_combined["train_upper_95"] = train_upper_95


# ------------------------------------------------------------------------------
# 6-7. AD判定
#
# s value <= train_upper_95 : AD In
# s value >  train_upper_95 : AD Out
# ------------------------------------------------------------------------------

df_predict_combined["ad_category"] = "AD In"

df_predict_combined.loc[
    df_predict_combined["s value"] > train_upper_95,
    "ad_category"
] = "AD Out"

num_ad_in = (
    df_predict_combined["ad_category"] == "AD In"
).sum()

num_ad_out = (
    df_predict_combined["ad_category"] == "AD Out"
).sum()

print(f"AD In と判定されたデータ点数: {num_ad_in}")
print(f"AD Out と判定されたデータ点数: {num_ad_out}")

if num_ad_out > 0:
    print("\nAD外と判定されたデータの詳細（最初の5件）:")

    ad_out_df = df_predict_combined[
        df_predict_combined["ad_category"] == "AD Out"
    ]

    print(
        ad_out_df[
            [
                predict_identifier_columns[0],
                "SMILES_original",
                "predicted_y",
                "s value",
                "Applicability_raw",
                "Applicability"
            ]
        ]
        .head()
        .to_string()
    )


# ==============================================================================
# 7. 予測値のヒストグラム作成
# ==============================================================================
print("\n--- 予測値のヒストグラムを作成中 ---")

plt.figure(figsize=(10, 6))

sns.histplot(
    df_predict_combined["predicted_y"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution of Predicted Y Values")
plt.xlabel("Predicted Y")
plt.ylabel("Frequency")

histogram_path = os.path.join(
    output_full_dir,
    output_histogram_name
)

plt.savefig(histogram_path)
plt.close()

print(f"予測値のヒストグラムを '{histogram_path}' に保存しました。")


# ==============================================================================
# 8. 結果の保存
# ==============================================================================
output_csv_path = os.path.join(
    output_full_dir,
    f"{output_csv_prefix}.csv"
)


# ------------------------------------------------------------------------------
# 出力CSV
# ------------------------------------------------------------------------------

final_output_df = df_predict_combined[
    predict_identifier_columns +
    [
        "SMILES_original",
        "SMILES_canonical",
        "predicted_y",
        "s value",
        "Applicability_raw",
        "train_lower_95",
        "train_upper_95",
        "Applicability",
        "ad_category"
    ]
].copy()


# ------------------------------------------------------------------------------
# 列名を変更
# ------------------------------------------------------------------------------

final_output_df = final_output_df.rename(
    columns={
        "predicted_y": "Predicted value"
    }
)


# ------------------------------------------------------------------------------
# 予測結果CSV保存
# ------------------------------------------------------------------------------

try:
    final_output_df.to_csv(
        output_csv_path,
        index=False
    )

    print(f"\n予測結果とAD判定を '{output_csv_path}' に保存しました。")
    print("\n処理が完了しました。")

except Exception as e:
    print(
        f"エラー: 結果のCSV保存中にエラーが発生しました: {e}",
        file=sys.stderr
    )
    print(f"エラー: 結果のCSV保存中にエラーが発生しました: {e}", file=sys.stderr)