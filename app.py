import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib

# --- KRİTİK: SUNUCU MODU (Donmayı Önler) ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa Ayarları
st.set_page_config(page_title="PDB Explorer by GeneticsBubble", layout="wide", page_icon="🧬")

# Hata Yakalama ve Import
try:
    from Bio.PDB import PDBList, PDBParser, NeighborSearch, Polypeptide
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from stmol import showmol
    import py3Dmol
except ImportError as e:
    st.error(f"Kritik kütüphane eksik: {e}. requirements.txt dosyasını kontrol et.")
    st.stop()

# --- SABİTLER ---
# Kyte-Doolittle Hidrofobiklik Skalası
KD_SCALE = {
    'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
    'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
    'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
    'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2
}

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
def get_detailed_chain_metrics(_structure):
    """Zincir bazlı detaylı sayısal veriler çıkarır (Düzeltilmiş Versiyon)."""
    chain_metrics = {}
    
    for model in _structure:
        for chain in model:
            residues = []
            for res in chain:
                # Sadece standart atom kayıtlarını işle (HETATM olmayanlar)
                if res.id[0] == ' ':
                    try:
                        # 3 harfli kodu 1 harfli koda çevirmeyi dene
                        # Standart dışı amino asitleri (MSE vb.) X olarak işaretle veya atla
                        try:
                            one_letter = Polypeptide.three_to_one(res.resname)
                        except KeyError:
                            continue # Tanınmayan amino asidi atla
                        
                        # B-Factor (Sıcaklık Faktörü) ortalaması
                        b_factors = [atom.bfactor for atom in res]
                        avg_bfactor = sum(b_factors) / len(b_factors) if b_factors else 0
                        
                        residues.append({
                            'Residue Index': res.id[1], # Rezidü numarası
                            'AA': one_letter,
                            'Hydrophobicity': KD_SCALE.get(one_letter, 0),
                            'B-Factor': avg_bfactor,
                            # Basit bir moleküler ağırlık (ProteinAnalysis tek harf ile çalışır)
                            'Mol Weight': ProteinAnalysis(one_letter).molecular_weight()
                        })
                    except Exception:
                        continue
            
            # Eğer zincirde veri toplandıysa kaydet
            if residues:
                chain_metrics[chain.id] = pd.DataFrame(residues)
                
    return chain_metrics

@st.cache_data
def find_interactions(_structure, distance_cutoff=5.0):
    """Ligand-Protein etkileşimleri."""
    atoms = list(_structure.get_atoms())
    ns = NeighborSearch(atoms)
    interactions = []
    
    for model in _structure:
        for chain in model:
            for residue in chain:
                # Ligandları bul (H_ ile başlayanlar veya HETATM)
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
    if style_type == "Cartoon": 
        view.setStyle({'cartoon': {**color_prop, 'opacity': 0.9}})
    elif style_type == "Stick": 
        view.setStyle({'stick': {**color_prop, 'radius': 0.2}})
    elif style_type == "Sphere": 
        view.setStyle({'sphere': {**color_prop, 'scale': 0.3}})
    
    # Yüzey Ayarı (Daha şeffaf yapıldı ki ligand görünsün)
    if show_surface: 
        view.addSurface(py3Dmol.VDW, {'opacity':0.3, 'color':'#f0f2f6'})

    if ligand_resname:
        # Ligand Görünümü: Ball and Stick (Daha belirgin)
        # Önce çubuklar
        view.addStyle({'resn': ligand_resname}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}})
        # Sonra atom topları (böylece garip görünmez, şık durur)
        view.addStyle({'resn': ligand_resname}, {'sphere': {'colorscheme': 'greenCarbon', 'scale': 0.3}})
        
        # Liganda zoom yap
        view.zoomTo({'resn': ligand_resname})
    else:
        view.zoomTo()
        
    return view

# --- ANA UYGULAMA ---
def main():
    st.title("🧬 Interactive PDB Ligand Explorer by GeneticsBubble")
    
    with st.sidebar.form(key='control_panel'):
        st.header("⚙️ Ayarlar")
        pdb_input = st.text_input("PDB ID:", value="3HTB").upper()
        
        style_type = st.selectbox("Stil", ["Cartoon", "Stick", "Sphere"])
        color_scheme = st.selectbox("Renk", ["Gökkuşağı", "Zincir", "Element", "B-Faktörü"])
        show_surf = st.checkbox("Yüzey Göster (Surface)", value=False)
        submit_btn = st.form_submit_button("Analiz Et 🚀")

    if submit_btn or pdb_input:
        if not os.path.exists('data'): os.makedirs('data')
        
        with st.spinner('GeneticsBubble motoru çalışıyor... Veriler işleniyor...'):
            structure, file_path, header = get_data(pdb_input)
            
            if structure:
                # --- VERİ HAZIRLIĞI ---
                chain_dfs = get_detailed_chain_metrics(structure)
                
                # --- TABLAR ---
                tab1, tab2, tab3 = st.tabs(["📋 Genel Bakış", "🧪 3D Yapı & Etkileşim", "📈 İleri Düzey Grafik Analizi"])
                
                # --- TAB 1: GENEL ---
                with tab1:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Çözünürlük", f"{header.get('resolution', 'N/A')} Å")
                    c2.metric("Yöntem", header.get('structure_method', 'N/A'))
                    c3.metric("Yayın Tarihi", header.get('deposition_date', 'N/A'))
                    
                    st.info(f"**Makromolekül Adı:** {header.get('name', 'Bilinmiyor')}")
                    st.write(f"**Kaynak:** {header.get('source', 'Bilinmiyor')}")
                    st.caption(f"Yazarlar: {header.get('author', '-')}")

                # --- TAB 2: 3D YAPI ---
                with tab2:
                    df_int = find_interactions(structure)
                    ligand = None
                    
                    col_3d, col_table = st.columns([2, 1])
                    
                    with col_table:
                        if not df_int.empty:
                            st.subheader("Ligand Listesi")
                            ligand = st.selectbox("İncelenecek Ligand:", df_int['Ligand'].unique())
                            
                            st.write("Etkileşimler:")
                            subset = df_int[df_int['Ligand'] == ligand]
                            st.dataframe(subset[['Etkileşen', 'Res ID', 'Mesafe (Å)']], height=400)
                        else:
                            st.warning("Bu yapıda belirgin bir ligand etkileşimi bulunamadı.")
                            
                    with col_3d:
                        view = render_3d_view(file_path, ligand, show_surf, style_type, color_scheme)
                        showmol(view, height=500, width=700)

                # --- TAB 3: GRAFİKLER (SEABORN POWER) ---
                with tab3:
                    if chain_dfs and len(chain_dfs) > 0:
                        selected_chain = st.selectbox("Analiz Edilecek Zincir:", list(chain_dfs.keys()))
                        df_chain = chain_dfs[selected_chain]
                        
                        if not df_chain.empty:
                            st.markdown(f"### 🧬 Zincir {selected_chain} - Biyoistatistiksel Analiz")
                            
                            # GRAFİK 1: HİDROPATİ & B-FACTOR
                            st.write("#### 🌊 Hidrofobiklik ve Stabilite Analizi")
                            fig1, ax1 = plt.subplots(figsize=(10, 4))
                            sns.lineplot(data=df_chain, x='Residue Index', y='Hydrophobicity', label='Hidrofobiklik', color='blue', alpha=0.6, ax=ax1)
                            ax2 = ax1.twinx()
                            sns.lineplot(data=df_chain, x='Residue Index', y='B-Factor', label='B-Factor (Esneklik)', color='red', alpha=0.4, ax=ax2)
                            ax1.set_ylabel("Hidrofobiklik")
                            ax2.set_ylabel("B-Factor")
                            st.pyplot(fig1)

                            col_heat, col_dist = st.columns(2)
                            
                            # GRAFİK 2: ISI HARİTASI
                            with col_heat:
                                st.write("#### 🔥 Özellik İlişkileri")
                                corr_data = df_chain[['Hydrophobicity', 'B-Factor', 'Mol Weight']].corr()
                                fig2, ax2 = plt.subplots(figsize=(6, 5))
                                sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
                                st.pyplot(fig2)

                            # GRAFİK 3: AA DAĞILIMI
                            with col_dist:
                                st.write("#### 📊 Amino Asit Sayıları")
                                fig3, ax3 = plt.subplots(figsize=(6, 5))
                                top_aa = df_chain['AA'].value_counts().head(10)
                                sns.barplot(x=top_aa.values, y=top_aa.index, palette='viridis', ax=ax3)
                                ax3.set_xlabel("Adet")
                                st.pyplot(fig3)
                        else:
                            st.warning("Seçilen zincir için yeterli veri toplanamadı.")
                        
                    else:
                        st.warning("Analiz edilecek uygun protein zinciri bulunamadı (Sadece DNA/RNA veya Ligand olabilir).")
            else:
                st.error("PDB verisi yüklenemedi. ID'yi kontrol edin.")

if __name__ == "__main__":
    main()
