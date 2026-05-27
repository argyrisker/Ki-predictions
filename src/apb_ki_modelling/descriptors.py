"""Descriptor generation utilities for Ki modelling."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.feature_selection import VarianceThreshold

ECFP_RADIUS = 2
ECFP_N_BITS = 1024


def ecfp4_bits(mol, radius: int = ECFP_RADIUS, n_bits: int = ECFP_N_BITS) -> np.ndarray:
    """Return ECFP4/Morgan fingerprint bits as a NumPy array."""
    arr = np.zeros((n_bits,), dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    ConvertToNumpyArray(fp, arr)
    return arr


def maccs_keys(mol) -> np.ndarray:
    arr = np.zeros((167,), dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    ConvertToNumpyArray(fp, arr)
    return arr


core_rdkit_descriptor_fns = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "BertzCT": Descriptors.BertzCT,
}

custom_smarts = {
    "Custom_HalogenCount": "[F,Cl,Br,I]",
    "Custom_BasicAmineCount": "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N=O)]",
    "Custom_AmideCount": "[NX3][CX3](=O)[#6]",
    "Custom_SulfonamideCount": "[NX3][SX4](=O)(=O)",
    "Custom_CarboxylicAcidCount": "[CX3](=O)[OX2H1]",
    "Custom_AromaticHeteroAtomCount": "[a;!c]",
    "Custom_PhenylLikeRingCount": "c1ccccc1",
}
custom_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in custom_smarts.items()}

custom_ratio_names = [
    "Custom_HeteroatomFraction",
    "Custom_AromaticAtomFraction",
    "Custom_RingAtomFraction",
    "Custom_HalogenFraction",
    "Custom_RingDensity",
    "Custom_AromaticRingRatio",
    "Custom_FlexibilityIndex",
    "Custom_TPSA_per_HeavyAtom",
    "Custom_LogP_per_HeavyAtom",
    "Custom_DonorAcceptorBalance",
    "Custom_DonorAcceptorTotal",
    "Custom_ChargedAtomCount",
    "Custom_NetFormalCharge",
    "Custom_HeteroToCarbonRatio",
]
custom_descriptor_names = list(custom_patterns.keys()) + custom_ratio_names


def safe_divide(num: float, den: float) -> float:
    return float(num) / float(den) if den not in (0, None) else 0.0


def descriptor_values(mol, descriptor_fns: dict) -> list[float]:
    vals = []
    for fn in descriptor_fns.values():
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(np.nan)
    return vals


def custom_descriptor_values(mol) -> list[float]:
    heavy = mol.GetNumHeavyAtoms()
    atoms = list(mol.GetAtoms())
    ring_info = mol.GetRingInfo()
    ring_atoms = set(i for ring in ring_info.AtomRings() for i in ring)

    halogen_count = sum(1 for atom in atoms if atom.GetAtomicNum() in [9, 17, 35, 53])
    hetero_count = sum(1 for atom in atoms if atom.GetAtomicNum() not in [1, 6])
    carbon_count = sum(1 for atom in atoms if atom.GetAtomicNum() == 6)
    aromatic_atom_count = sum(1 for atom in atoms if atom.GetIsAromatic())
    charged_atom_count = sum(1 for atom in atoms if atom.GetFormalCharge() != 0)
    net_charge = Chem.GetFormalCharge(mol)

    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Descriptors.MolLogP(mol)
    donors = rdMolDescriptors.CalcNumHBD(mol)
    acceptors = rdMolDescriptors.CalcNumHBA(mol)

    values = []
    for patt in custom_patterns.values():
        values.append(float(len(mol.GetSubstructMatches(patt))) if patt is not None else np.nan)

    values.extend([
        safe_divide(hetero_count, heavy),
        safe_divide(aromatic_atom_count, heavy),
        safe_divide(len(ring_atoms), heavy),
        safe_divide(halogen_count, heavy),
        safe_divide(ring_count, heavy),
        safe_divide(aromatic_rings, ring_count),
        safe_divide(rot_bonds, heavy),
        safe_divide(tpsa, heavy),
        safe_divide(logp, heavy),
        float(donors - acceptors),
        float(donors + acceptors),
        float(charged_atom_count),
        float(net_charge),
        safe_divide(hetero_count, carbon_count),
    ])
    return values


def build_descriptor_matrices(mols: list, use_extended: bool = True) -> dict:
    """Build ECFP, MACCS, RDKit and custom descriptor matrices."""
    x_ecfp = np.vstack([ecfp4_bits(mol) for mol in mols]).astype(np.float32)
    x_maccs = np.vstack([maccs_keys(mol) for mol in mols]).astype(np.float32)
    x_rdkit_core = np.array([descriptor_values(mol, core_rdkit_descriptor_fns) for mol in mols], dtype=np.float32)
    x_custom = np.array([custom_descriptor_values(mol) for mol in mols], dtype=np.float32)

    for arr in [x_ecfp, x_maccs, x_rdkit_core, x_custom]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ecfp_feature_names = [f"ECFP4_{i}" for i in range(ECFP_N_BITS)]
    maccs_feature_names = [f"MACCS_{i}" for i in range(167)]
    rdkit_core_feature_names = list(core_rdkit_descriptor_fns.keys())

    x_base = np.hstack([x_ecfp, x_rdkit_core]).astype(np.float32)
    base_names = ecfp_feature_names + rdkit_core_feature_names

    x_extended_raw = np.hstack([x_ecfp, x_maccs, x_rdkit_core, x_custom]).astype(np.float32)
    extended_names_raw = ecfp_feature_names + maccs_feature_names + rdkit_core_feature_names + custom_descriptor_names

    selector = VarianceThreshold(threshold=0.0)
    x_extended = selector.fit_transform(x_extended_raw).astype(np.float32)
    extended_names = [name for name, keep in zip(extended_names_raw, selector.get_support()) if keep]

    return {
        "X": x_extended if use_extended else x_base,
        "feature_names": extended_names if use_extended else base_names,
        "X_ecfp": x_ecfp,
        "X_maccs": x_maccs,
        "X_rdkit_core": x_rdkit_core,
        "X_custom": x_custom,
        "X_base": x_base,
        "X_extended": x_extended,
    }
