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
    return pd.DataFrame(interactions)

def render_3d_view(pdb_file_path, ligand_resname, show_surface, style_type, color_scheme):
    if not pdb_file_path: return None
    with open(pdb_file_path, 'r') as f: pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    
    # Renk Ayarı
    color_prop = {}
    if color_scheme == "Gökkuşağı": color_prop = {'colorscheme': 'spectrum'}
    elif color_scheme == "Zincir": color_prop = {'colorscheme': 'chain'}
    elif color_scheme == "Element": color_prop = {'colorscheme': 'default'}
    elif color_scheme == "B-Faktörü": color_prop = {'colorscheme': 'b'}

    # Stil Ayarı
    if style_type == "Cartoon": view.setStyle({'cartoon': {**color_prop, 'opacity': 0.9}})
    elif style_type == "Stick": view.setStyle({'stick': {**color_prop, 'radius': 0.2}})
    elif style_type == "Sphere": view.setStyle({'sphere': {**color_prop, 'scale': 0.3}})
    
    if show_surface: view.addSurface(py3Dmol.VDW, {'opacity':0.4, 'color':'#f0f2f6'})

    if ligand_resname:
        view.addStyle({'resn': ligand_resname}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.4}})
        view.zoomTo({'resn': ligand_resname})
    else:
        view.zoomTo()
        
    return view

# --- ANA UYGULAMA ---
def main():
    st.title("🧬 BioVis Pro: Stable Mode")
    
    with st.sidebar.form(key='control_panel'):
        st.header("⚙️ Ayarlar")
        pdb_input = st.text_input("PDB ID:", value="9NXY").upper()
        style_type = st.selectbox("Stil", ["Cartoon", "Stick", "Sphere"])
        color_scheme = st.selectbox("Renk", ["Gökkuşağı", "Zincir", "Element", "B-Faktörü"])
        show_surf = st.checkbox("Yüzey Göster", value=False)
        submit_btn = st.form_submit_button("Analiz Et")

    if submit_btn or pdb_input:
        if not os.path.exists('data'): os.makedirs('data')
        
        with st.spinner('Veriler İşleniyor...'):
            structure, file_path, header = get_data(pdb_input)
            
            if structure:
                tab1, tab2, tab3 = st.tabs(["Genel", "3D Yapı", "Analiz"])
                
                with tab1:
                    c1, c2 = st.columns(2)
                    c1.metric("Çözünürlük", f"{header.get('resolution', 'N/A')} Å")
                    c2.metric("Metot", header.get('structure_method', 'N/A'))
                    st.info(header.get('name', 'İsimsiz'))

                with tab2:
                    df_int = find_interactions(structure)
                    ligand = None
                    if not df_int.empty:
                        ligand = st.selectbox("Ligand Seç:", df_int['Ligand'].unique())
                    
                    view = render_3d_view(file_path, ligand, show_surf, style_type, color_scheme)
                    showmol(view, height=500, width=700)

                with tab3:
                    df_chains = analyze_sequence(structure)
                    df_prot = df_chains[df_chains['Tip'] == 'Protein']
                    
                    if not df_prot.empty:
                        st.dataframe(df_prot[["Zincir", "Uzunluk", "Mol. Ağırlık", "pI"]])
                        
                        chain_sel = st.selectbox("Zincir Analizi:", df_prot['Zincir'].unique())
                        row = df_prot[df_prot['Zincir'] == chain_sel].iloc[0]
                        
                        # --- MATPLOTLIB GRAFİĞİ (AGG BACKEND İLE) ---
                        st.write("Amino Asit Dağılımı:")
                        aa_counts = row['AA_Count']
                        if aa_counts:
                            # Figure oluştururken explicit boyut veriyoruz
                            fig = plt.figure(figsize=(10, 4))
                            plt.bar(aa_counts.keys(), aa_counts.values(), color='#4e79a7')
                            plt.xlabel("Amino Asit")
                            plt.ylabel("Sayı")
                            st.pyplot(fig) # Streamlit'e figürü gönder
                        
                        with st.expander("FASTA Dizisini Göster"):
                            st.code(row['Dizi'], language='text')
                    else:
                        st.warning("Protein zinciri bulunamadı.")
            else:
                st.error("PDB yüklenemedi. ID'yi kontrol edin.")

if __name__ == "__main__":
    main()
