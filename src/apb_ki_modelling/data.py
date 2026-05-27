"""Data cleaning and molecule standardisation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass(frozen=True)
class CleaningSummary:
    raw_rows: int
    cleaned_rows: int
    valid_smiles: int
    invalid_smiles: int
    multi_fragment_smiles: int
    unique_canonical_molecules: int


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the Ki dataset and validate the required columns."""
    path = Path(path)
    df = pd.read_csv(path)
    required = {"SMILES", "Ki (nM)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    return df


def mol_from_smiles(smiles: str):
    """Parse a SMILES string into an RDKit molecule, returning None on failure."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def keep_largest_fragment(mol):
    """Return the largest covalent fragment from a molecule."""
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def canonical_smiles_from_mol(mol) -> str | None:
    """Return canonical isomeric SMILES for an RDKit molecule."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def clean_ki_values(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without positive numeric Ki values and add log10(Ki)."""
    df = df_raw.rename(columns={"Ki (nM)": "Ki_nM"}).copy()
    df = df.dropna(subset=["SMILES", "Ki_nM"])
    df["Ki_nM"] = pd.to_numeric(df["Ki_nM"], errors="coerce")
    df = df.dropna(subset=["Ki_nM"])
    df = df[df["Ki_nM"] > 0].copy()
    df["log10_Ki"] = np.log10(df["Ki_nM"])
    return df


def standardise_and_aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    """Canonicalise molecules and aggregate duplicate canonical SMILES using median Ki."""
    df = df.copy()
    df["mol_raw"] = df["SMILES"].apply(mol_from_smiles)
    df["valid_smiles"] = df["mol_raw"].notna()
    df["multi_fragment"] = df["SMILES"].astype(str).str.contains(".", regex=False)
    df["mol_largest_fragment"] = df["mol_raw"].apply(keep_largest_fragment)
    df["canonical_smiles"] = df["mol_largest_fragment"].apply(canonical_smiles_from_mol)

    df_valid = df.dropna(subset=["canonical_smiles"]).copy()
    agg = (
        df_valid.groupby("canonical_smiles", as_index=False)
        .agg(
            Ki_nM=("Ki_nM", "median"),
            n_records=("Ki_nM", "size"),
            original_smiles_example=("SMILES", "first"),
            had_multifragment=("multi_fragment", "max"),
        )
    )
    agg["log10_Ki"] = np.log10(agg["Ki_nM"])
    agg["mol"] = agg["canonical_smiles"].apply(Chem.MolFromSmiles)

    summary = CleaningSummary(
        raw_rows=len(df),
        cleaned_rows=len(df_valid),
        valid_smiles=int(df["valid_smiles"].sum()),
        invalid_smiles=int((~df["valid_smiles"]).sum()),
        multi_fragment_smiles=int(df["multi_fragment"].sum()),
        unique_canonical_molecules=len(agg),
    )
    return agg, summary
