import streamlit as st
from Bio.PDB import PDBList, PDBParser, NeighborSearch
import pandas as pd
import os
from stmol import showmol
import py3Dmol
import numpy as np 

st.set_page_config(page_title="BioVis Pro", layout="wide", page_icon="🧬")

@st.cache_data
def get_structure(pdb_id):
    pdbl = PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='data', file_format='pdb')
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, file_path)
    return structure, file_path

def find_interactions(structure, distance_cutoff=5.0):
    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)
    interactions = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith("H_") and residue.resname != "HOH":
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
                                "Chain": chain.id,
                                "Residue": n.resname,
                                "Res ID": n.id[1],
                                "Distance (Å)": round(dist, 2)
                            })
                            
    return pd.DataFrame(interactions)

def render_3d_view(pdb_file_path, ligand_resname, show_surface):
    with open(pdb_file_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'white', 'opacity': 0.8}})
    
    if show_surface:
        view.addSurface(py3Dmol.VDW, {'opacity':0.4, 'color':'#f0f2f6'})

    view.addStyle({'resn': ligand_resname}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})
    view.addStyle({'within': {'distance': 5, 'sel': {'resn': ligand_resname}}}, 
                  {'stick': {'colorscheme': 'grayCarbon', 'radius': 0.15}})
    
    view.zoomTo({'resn': ligand_resname})
    return view

def main():
    st.sidebar.title("⚙️ Kontrol Paneli")
    pdb_id = st.sidebar.text_input("PDB ID:", value="1CBS").upper()
    show_surf = st.sidebar.checkbox("Yüzeyi Göster (Surface)")
    
    if st.sidebar.button("Analiz Et"):
        with st.spinner('RCSB Veritabanına Bağlanılıyor...'):
            try:
                if not os.path.exists('data'): os.makedirs('data')
                structure, file_path = get_structure(pdb_id)
                df = find_interactions(structure)
                
                if not df.empty:
                    unique_ligands = df['Ligand'].unique()
                    selected_ligand = st.selectbox("İncelenecek Ligand'ı Seçin:", unique_ligands)
                    subset_df = df[df['Ligand'] == selected_ligand]
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("📊 Etkileşim Listesi")
                        st.dataframe(subset_df, height=500)
                    with col2:
                        st.subheader("🧬 3D Bağlanma Cebi")
                        view = render_3d_view(file_path, selected_ligand, show_surf)
                        showmol(view, height=600, width=800)
                else:
                    st.warning("Bu yapıda uygun bir ligand bulunamadı.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
