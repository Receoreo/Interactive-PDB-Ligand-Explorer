import streamlit as st
from Bio.PDB import PDBList, PDBParser, NeighborSearch
import pandas as pd
import numpy as np
import os
from stmol import showmol
import py3Dmol

st.set_page_config(page_title="BioVis Pro V2", layout="wide", page_icon="🧬")

@st.cache_data
def get_data(pdb_id):
    pdbl = PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='data', file_format='pdb')
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, file_path)
    return structure, file_path, structure.header

@st.cache_data
def find_interactions(_structure, distance_cutoff=5.0):
    atoms = list(_structure.get_atoms())
    ns = NeighborSearch(atoms)
    interactions = []
    
    for model in _structure:
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

def get_chain_info(structure):
    chain_data = []
    for model in structure:
        for chain in model:
            residue_count = len(list(chain.get_residues()))
            chain_data.append({
                "Chain ID": chain.id,
                "Residue Count": residue_count,
            })
    return pd.DataFrame(chain_data)

def render_3d_view(pdb_file_path, ligand_resname, show_surface, style_type):
    with open(pdb_file_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    
    if style_type == "Cartoon":
        view.setStyle({'cartoon': {'color': 'white', 'opacity': 0.8}})
    elif style_type == "Stick":
        view.setStyle({'stick': {'colorscheme': 'grayCarbon', 'opacity': 0.8}})
    
    if show_surface:
        view.addSurface(py3Dmol.VDW, {'opacity':0.5, 'color':'#f0f2f6'})

    if ligand_resname:
        view.addStyle({'resn': ligand_resname}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})
        view.addStyle({'within': {'distance': 5, 'sel': {'resn': ligand_resname}}}, 
                      {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.15}})
        view.zoomTo({'resn': ligand_resname})
    else:
        view.zoomTo()
        
    return view

def main():
    st.title("🧬 BioVis Pro: Advanced Structure Explorer")
    
    with st.sidebar.form(key='control_panel'):
        st.header("⚙️ Kontrol Paneli")
        pdb_input = st.text_input("PDB ID:", value="3HTB").upper()
        
        st.markdown("---")
        st.subheader("Görsel Ayarlar")
        show_surf = st.checkbox("Yüzeyi Göster (Surface)")
        style_type = st.selectbox("Protein Stili", ["Cartoon", "Stick"])
        
        submit_btn = st.form_submit_button("Analizi Güncelle")

    if submit_btn or pdb_input:
        try:
            if not os.path.exists('data'): os.makedirs('data')
            
            with st.spinner('Veriler çekiliyor ve analiz ediliyor...'):
                structure, file_path, header = get_data(pdb_input)
                
                tab1, tab2, tab3 = st.tabs(["📋 Genel Bilgiler", "🧬 3D & Ligand", "🔗 Zincir Detayları"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Protein Adı:** {header.get('name', 'Bilinmiyor')}")
                        st.write(f"**Yayınlanma Tarihi:** {header.get('deposition_date', 'Yok')}")
                        st.write(f"**Sınıf:** {header.get('head', 'Yok')}")
                    with col2:
                        st.metric("Çözünürlük (Resolution)", f"{header.get('resolution', 'N/A')} Å")
                        st.metric("Yapı Yöntemi", header.get('structure_method', 'Bilinmiyor'))
                        
                    st.markdown("### 📜 Referans / Yazar Bilgisi")
                    st.write(header.get('author', 'Yazar bilgisi bulunamadı.'))

                with tab2:
                    df_interactions = find_interactions(structure)
                    
                    if not df_interactions.empty:
                        ligand_list = df_interactions['Ligand'].unique()
                        selected_ligand = st.selectbox("İncelenecek Ligand:", ligand_list)
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.write("### 📏 Etkileşimler")
                            subset_df = df_interactions[df_interactions['Ligand'] == selected_ligand]
                            st.dataframe(subset_df, height=400)
                        with c2:
                            st.write("### 🧪 3D Yapı")
                            view = render_3d_view(file_path, selected_ligand, show_surf, style_type)
                            showmol(view, height=500, width=700)
                    else:
                        st.warning("Bu yapıda uygun bir ligand bulunamadı. Sadece proteini görüntülüyorsunuz.")
                        view = render_3d_view(file_path, None, show_surf, style_type)
                        showmol(view, height=500, width=700)

                with tab3:
                    st.subheader("Protein Zincir İstatistikleri")
                    chain_df = get_chain_info(structure)
                    st.table(chain_df)
                    
                    st.bar_chart(chain_df.set_index("Chain ID"))

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
