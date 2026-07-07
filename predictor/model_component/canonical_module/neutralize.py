import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# === neutralize_charges 用 ===
def _InitialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

_reactions = None

def neutralize_charges(smiles, reactions=None):
    try:
        global _reactions
        if reactions is None:
            if _reactions is None:
                _reactions = _InitialiseNeutralisationReactions()
            reactions = _reactions
        mol = Chem.MolFromSmiles(smiles)
        replaced = False
        for i, (reactant, product) in enumerate(reactions):
            while mol.HasSubstructMatch(reactant):
                replaced = True
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        if replaced:
            try:
                mol.UpdatePropertyCache(strict=False)
            except:
                pass
            smiles = Chem.MolToSmiles(mol, canonical=True)
            print("neutralized_smiles: {0}".format(smiles), file=sys.stderr)
        return smiles
    except:
        return smiles

# === ionize_charges 用 ===
def _InitialiseIonisationReactions():
    patts = (
        # 金属イオン → 陽イオン化
        ('[#11&+0]', '[Na+]'),   # Na
        ('[#19&+0]', '[K+]'),    # K
        ('[#3&+0]', '[Li+]'),    # Li
        ('[#20&+0]', '[Ca+2]'),  # Ca
        ('[#12&+0]', '[Mg+2]'),  # Mg
        ('[#30&+0]', '[Zn+2]'),  # Zn
        ('[#26&+0]', '[Fe+3]'),  # Fe
        ('[#25&+0]', '[Mn+2]'),  # Mn
        ('[#27&+0]', '[Co+2]'),  # Co
        ('[#28&+0]', '[Ni+2]'),  # Ni
        ('[#29&+0]', '[Cu+2]'),  # Cu
        ('[#13&+0]', '[Al+3]'),  # Al
        ('[#24&+0]', '[Cr+3]'),  # Cr
        ('[#66&+0]', '[Dy+3]'),  # Dy
        ('[#80&+0]', '[Hg+2]'),  # Hg

        # ハロゲン → 陰イオン化
        ('[#17&+0]', '[Cl-]'),   # Cl
        ('[#35&+0]', '[Br-]'),   # Br
        ('[#53&+0]', '[I-]'),    # I
        ('[#9&+0]', '[F-]'),     # F

        # 非金属原子 → イオン化（O, N含めた最新版）
        ('[#8&+0]', '[O-]'),    # O → [O-]
        ('[#7&+0]', '[N+]'),    # N → [N+]
        ('[#16&+0]', '[S-]'),   # S → [S-]
        ('[#15&+0]', '[P+]'),   # P → [P+]
        ('[#5&+0]', '[B+]'),    # B → [B+]
        ('[#14&+0]', '[Si+]'),  # Si → [Si+]
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


_ion_reactions = None

def ionize_charges(smiles, reactions=None):
    try:
        global _ion_reactions
        if reactions is None:
            if _ion_reactions is None:
                _ion_reactions = _InitialiseIonisationReactions()
            reactions = _ion_reactions
        mol = Chem.MolFromSmiles(smiles)
        replaced = False
        for i, (reactant, product) in enumerate(reactions):
            if mol.HasSubstructMatch(reactant):
                replaced = True
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        if replaced:
            try:
                mol.UpdatePropertyCache(strict=False)
            except:
                pass
            smiles = Chem.MolToSmiles(mol, canonical=True)
            print("ionized_smiles: {0}".format(smiles), file=sys.stderr)
        return smiles
    except:
        return smiles

# === 簡易テスト用 ===
if __name__ == '__main__':
    smis = (
        "Clc1c(C2([NH2+]C)C(=O)CCCC2)cccc1",
        "c1cccc[nH+]1",
        "C[N+](C)(C)C", "c1ccccc1[NH3+]",
        "CC(=O)[O-]", "c1ccccc1[O-]",
        "CCS",
        "CO[O-]",
        "[OH-]",
        "[NH4+]",
        "O=S(=O)([O-])[O-]",
        "[N+](=O)([O-])[O-]",
        "C[N-]S(=O)(=O)C",
        "C[N-]C=C", "C[N-]N=C",
        "[O-]Cl(=O)(=O)=O",
        "c1ccc[n-]1",
        "[Li+]",
        "CC[N-]C(=O)CC",
        "O",  # O (should ionize to [O-] if run through ionize_charges)
        "N",  # N (should ionize to [N+])
    )
    for smi in smis:
        molSmiles = neutralize_charges(smi)
        print("neutralize:", smi + " -> " + molSmiles)

    print("\n--- Testing ionize_charges ---\n")
    for smi in smis:
        molSmiles = ionize_charges(smi)
        print("ionize:", smi + " -> " + molSmiles)
