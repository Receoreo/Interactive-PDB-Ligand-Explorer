import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib

# --- KRİTİK DÜZELTME: SUNUCU MODU ---
# Matplotlib'in sunucuda ekran aramamasını sağlar.
# Bu satır 'import matplotlib.pyplot'tan ÖNCE gelmelidir.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Sayfa ayarı her şeyden önce gelmeli
st.set_page_config(page_title="BioVis Pro V3.3 (Stable)", layout="wide", page_icon="🧬")

# Kütüphane yükleme kontrolü
try:
    from Bio.PDB import PDBList, PDBParser, NeighborSearch, Polypeptide
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from stmol import showmol
    import py3Dmol
except ImportError as e:
    st.error(f"Kritik kütüphane eksik: {e}. Lütfen requirements.txt dosyasını kontrol edin.")
    st.stop()

# --- FONKSİYONLAR ---

@st.cache_data
def get_data(pdb_id):
    """PDB dosyasını indirir."""
    pdbl = PDBList()
    try:
        file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='data', file_format='pdb', obsolete=False)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, file_path)
        return structure, file_path, structure.header
    except Exception as e:
        return None, None, None

@st.cache_data
def analyze_sequence(_structure):
    """Zincir analizi yapar."""
    chain_data = []
    for model in _structure:
        for chain in model:
            ppb = Polypeptide.PPBuilder()
            pp_list = ppb.build_peptides(chain)
            
            if len(pp_list) > 0:
                sequence = "".join([str(pp.get_sequence()) for pp in pp_list])
                try:
                    analyzed_seq = ProteinAnalysis(sequence)
                    mw = analyzed_seq.molecular_weight()
                    isoelectric = analyzed_seq.isoelectric_point()
                    aa_count = analyzed_seq.count_amino_acids()
                    instability = analyzed_seq.instability_index()
                except:
                    mw, isoelectric, instability = 0, 0, 0
                    aa_count = {}

                chain_data.append({
                    "Zincir": chain.id,
                    "Tip": "Protein",
                    "Uzunluk": len(sequence),
                    "Mol. Ağırlık": round(mw, 2),
                    "pI": round(isoelectric, 2),
                    "Kararsızlık": round(instability, 2),
                    "Dizi": sequence,
                    "AA_Count": aa_count
                })
            else:
                residues = list(chain.get_residues())
                chain_data.append({
                    "Zincir": chain.id,
                    "Tip": "Ligand/DNA/RNA",
                    "Uzunluk": len(residues),
                    "Mol. Ağırlık": 0,
                    "pI": 0,
                    "Kararsızlık": 0,
                    "Dizi": "",
                    "AA_Count": {}
                })
    return pd.DataFrame(chain_data)

@st.cache_data
def find_interactions(_structure, distance_cutoff=5.0):
    """Etkileşimleri hesaplar."""
    atoms = list(_structure.get_atoms())
    ns = NeighborSearch(atoms)
    interactions = []
    
    for model in _structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith("H_") and residue.resname != "HOH":
                    try:
                        ligand_center = residue.center_of_mass()
                        neighbors = ns.search(ligand_center, distance_cutoff, level='R')
                        for n in neighbors:
                            if n != residue:
                                dist = 0
                                if 'CA' in n:
                                    diff = n['CA'].coord - residue.center_of_mass()
                                    dist = np.linalg.norm(diff)
                                interactions.append({
                                    "Ligand": residue.resname,
                                    "Zincir": chain.id,
                                    "Etkileşen": n.resname,
                                    "Res ID": n.id[1],
                                    "Mesafe (Å)": round(dist, 2)
                                })
                    except:
                        continue
    return pd.
