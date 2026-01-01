import streamlit as st
from Bio.PDB import PDBList, PDBParser, NeighborSearch, Polypeptide
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import os
from stmol import showmol
import py3Dmol
import altair as alt

st.set_page_config(page_title="BioVis Pro V3", layout="wide", page_icon="🧬")

# --- CSS İle Biraz Makyaj ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data(pdb_id):
    """PDB dosyasını indirir ve parse eder."""
    pdbl = PDBList()
    try:
        file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='data', file_format='pdb')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, file_path)
        return structure, file_path, structure.header
    except Exception as e:
        st.error(f"PDB ID hatalı veya indirilemedi: {e}")
        return None, None, None

@st.cache_data
def analyze_sequence(structure):
    """Zincirlerin biyokimyasal özelliklerini analiz eder (RCSB benzeri)."""
    chain_data = []
    
    for model in structure:
        for chain in model:
            # Polypeptide.PPBuilder ile sadece protein kısımlarını al
            ppb = Polypeptide.PPBuilder()
            pp_list = ppb.build_peptides(chain)
            
            # Eğer protein dizisi varsa (DNA/RNA değilse)
            if len(pp_list) > 0:
                sequence = str(pp_list[0].get_sequence())
                analyzed_seq = ProteinAnalysis(sequence)
                
                mw = analyzed_seq.molecular_weight()
                aromaticity = analyzed_seq.aromaticity()
                instability = analyzed_seq.instability_index()
                isoelectric = analyzed_seq.isoelectric_point()
                aa_count = analyzed_seq.count_amino_acids()
                
                chain_data.append({
                    "Zincir": chain.id,
                    "Tip": "Protein",
                    "Uzunluk": len(sequence),
                    "Mol. Ağırlık (Da)": round(mw, 2),
                    "İzoelektrik (pI)": round(isoelectric, 2),
                    "Aromatiklik": round(aromaticity, 3),
                    "Kararsızlık İndeksi": round(instability, 2),
                    "Dizi": sequence,
                    "AA_Count": aa_count
                })
            else:
                # Protein değilse (Örn: DNA, RNA veya sadece Ligand zinciri)
                residues = list(chain.get_residues())
                chain_data.append({
                    "Zincir": chain.id,
                    "Tip": "Non-Protein/Ligand",
                    "Uzunluk": len(residues),
                    "Mol. Ağırlık (Da)": "N/A",
                    "İzoelektrik (pI)": "N/A",
                    "Aromatiklik": "N/A",
                    "Kararsızlık İndeksi": "N/A",
                    "Dizi": "N/A",
                    "AA_Count": {}
                })
                
    return pd.DataFrame(chain_data)

@st.cache_data
def find_interactions(_structure, distance_cutoff=5.0):
    atoms = list(_structure.get_atoms())
    ns = NeighborSearch(atoms)
    interactions = []
    
    for model in _structure:
        for chain in model:
            for residue in chain:
                # Sadece HETATM (Ligandlar) ve Su olmayanlar
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
                                "Zincir": chain.id,
                                "Etkileşen": n.resname,
                                "Res ID": n.id[1],
                                "Mesafe (Å)": round(dist, 2)
                            })
                            
    return pd.DataFrame(interactions)

def render_3d_view(pdb_file_path, ligand_resname, show_surface, style_type, color_scheme):
    with open(pdb_file_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    
    # Renk Şeması Mantığı
    color_prop = {}
    if color_scheme == "Gökkuşağı (Rainbow)":
        color_prop = {'colorscheme': 'spectrum'}
    elif color_scheme == "Zincire Göre (Chain)":
        color_prop = {'colorscheme': 'chain'}
    elif color_scheme == "Atom Tipi (Element)":
        color_prop = {'colorscheme': 'default'}
    elif color_scheme == "B-Faktörü (Sıcaklık)":
        color_prop = {'colorscheme': 'b'}

    # Stil Mantığı
    style_prop = {}
    if style_type == "Cartoon":
        style_prop = {'cartoon': {**color_prop, 'opacity': 0.9}}
    elif style_type == "Stick":
        style_prop = {'stick': {**color_prop, 'radius': 0.2}}
    elif style_type == "Sphere (VDW)":
        style_prop = {'sphere': {**color_prop, 'scale': 0.3}}
    elif style_type == "Line":
        style_prop = {'line': {**color_prop}}

    view.setStyle(style_prop)
    
    if show_surface:
        view.addSurface(py3Dmol.VDW, {'opacity':0.4, 'color':'#f0f2f6'})

    if ligand_resname:
        # Seçili ligandı vurgula
        view.addStyle({'resn': ligand_resname}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.4}})
        view.addStyle({'resn': ligand_resname}, {'sphere': {'scale': 0.3, 'opacity': 0.6}})
        view.zoomTo({'resn': ligand_resname})
    else:
        view.zoomTo()
        
    return view

def main():
    st.title("🧬 BioVis Pro: RCSB Style Explorer")
    
    # --- Sidebar ---
    with st.sidebar.form(key='control_panel'):
        st.header("⚙️ Konfigürasyon")
        pdb_input = st.text_input("PDB ID:", value="9NXY").upper()
        
        st.markdown("### 🎨 Görselleştirme")
        style_type = st.selectbox("Görünüm Stili", ["Cartoon", "Stick", "Sphere (VDW)", "Line"])
        color_scheme = st.selectbox("Renklendirme", ["Gökkuşağı (Rainbow)", "Zincire Göre (Chain)", "Atom Tipi (Element)", "B-Faktörü (Sıcaklık)"])
        show_surf = st.checkbox("Yüzey (Surface)", value=False)
        
        submit_btn = st.form_submit_button("Analizi Başlat 🚀")

    if submit_btn or pdb_input:
        if not os.path.exists('data'): os.makedirs('data')
        
        with st.spinner('PDB verisi indiriliyor ve RCSB metrikleri hesaplanıyor...'):
            structure, file_path, header = get_data(pdb_input)
            
            if structure:
                tab1, tab2, tab3 = st.tabs(["📋 Özet Bilgiler", "🧪 3D Yapı & Ligand", "🧬 Dizi & Biyo-Analiz"])
                
                # --- TAB 1: ÖZET ---
                with tab1:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sınıf", header.get('head', 'N/A'))
                    col2.metric("Çözünürlük", f"{header.get('resolution', 'N/A')} Å")
                    col3.metric("Yöntem", header.get('structure_method', 'N/A'))
                    
                    st.info(f"**Protein Adı:** {header.get('name', 'Bilinmiyor')}")
                    
                    with st.expander("📚 Referans ve Yazarlar"):
                        st.write(header.get('author', 'Veri yok'))
                        st.write
