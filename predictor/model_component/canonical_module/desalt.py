from rdkit import Chem
import numpy as np

CATIONIC_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Po",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
}

def is_cationic(mol):
    if mol is None:
        return False
    return any(atom.GetSymbol() in CATIONIC_ELEMENTS for atom in mol.GetAtoms())

def get_cation_mass(mol):
    if mol is None:
        return 0
    return sum(atom.GetMass() for atom in mol.GetAtoms() if atom.GetSymbol() in CATIONIC_ELEMENTS)

def desalt(smiles):
    fragments = smiles.split('.')
    if len(fragments) == 1:
        return smiles

    mols = [Chem.MolFromSmiles(f) for f in fragments]
    atom_counts = np.array([mol.GetNumAtoms() if mol else 0 for mol in mols])
    max_atoms = atom_counts.max()
    max_atom_indices = np.where(atom_counts == max_atoms)[0]
    candidate_idxs = max_atom_indices

    # 陽イオン優先
    cation_flags = [is_cationic(mols[i]) for i in candidate_idxs]
    cation_idxs = [i for i, flag in zip(candidate_idxs, cation_flags) if flag]

    if cation_idxs:
        # 陽イオンが複数ある場合、原子量最大のものを選択
        if len(cation_idxs) == 1:
            return fragments[cation_idxs[0]]
        else:
            masses = [get_cation_mass(mols[i]) for i in cation_idxs]
            return fragments[cation_idxs[np.argmax(masses)]]
    else:
        # 陽イオンがなければ、候補のうち最初のもの（原子数最大）を返す
        return fragments[candidate_idxs[0]]
