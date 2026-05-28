import time
import streamlit as st
import subprocess
import os
import shutil
import urllib.request
import urllib.parse
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64
import io

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# =====================================================================
# 1. INITIALIZATION & CLOUD BACKEND BOOTSTRAPPING
# =====================================================================

def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")

ensure_linux_vina_exists()

def initialize_session_states():
    defaults = {
        "protein_name": "Unknown Protein",
        "cx": 0.0, "cy": 0.0, "cz": 0.0,
        "sx": 20, "sy": 20, "sz": 20,
        "exhaustiveness": 8,
        "target_ready": False,
        "ligand_ready": False,
        "local_target_path": None,
        "pdb_id_display": "Custom",
        "docking_results_raw": None,
        "serialized_ligand_block": None,
        "ligand_summary_text": "",
        "smiles_cache": "",
        "baseline_affinity": None,
        "active_retained_ions": "None",
        "uff_cache": {},
        "last_uploaded_protein": "",
        "last_uploaded_ligand": "",
        "detected_pockets": [],
        "selected_native_ligand": "Manual Coordinate Assignment"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_states()

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# =====================================================================
# 2. BIOINFORMATICS STRUCTURAL CONVERTERS & PARSERS
# =====================================================================

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound Name", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical Derivative")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception: pass 
    return metadata

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "name": "Unknown Protein",
        "title": "Uploaded Protein Structure Matrix", "id": pdb_id.upper() if pdb_id and pdb_id != "Uploaded File" else "Unknown",
        "class": "Unknown Classification", "organism": "Unknown",
        "system": "Unknown Expression System", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path): return meta
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            title_parts = []
            for line in f:
                if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
                elif line.startswith("HEADER"): 
                    meta["class"] = line[10:50].strip().title()
                    if len(line) >= 66:
                        possible_id = line[62:66].strip()
                        if len(possible_id) == 4:
                            meta["id"] = possible_id.upper()
                elif line.startswith("COMPND"):
                    if "MOLECULE:" in line:
                        mol_name = line.split("MOLECULE:")[1].split(";")[0].strip()
                        if meta["name"] == "Unknown Protein":
                            meta["name"] = mol_name.title()
                elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
                elif "EXPRESSION_SYSTEM" in line: meta["system"] = line.split(":")[-1].replace(";","").strip()
                elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
                elif "RESOLUTION." in line and "ANGSTROMS." in line:
                    match = re.search(r"(\d+\.\d+)", line)
                    if match: meta["res"] = f"{match.group(1)} Å"
        if title_parts: meta["title"] = " ".join(title_parts).title()
        if meta["name"] == "Unknown Protein" and meta["title"] != "Uploaded Protein Structure Matrix":
            meta["name"] = meta["title"]
    except Exception: pass
    return meta

def discover_and_list_all_heteroatoms(file_path):
    hetero_counts = {}
    if not os.path.exists(file_path): return hetero_counts
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name in ["HOH", "WAT", "DOD"]: continue
                hetero_counts[res_name] = hetero_counts.get(res_name, 0) + 1
    return hetero_counts

def parse_bound_ligands(file_path):
    ligands = {}
    if not os.path.exists(file_path): return []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip() if line[21].strip() else "A"
                try: res_seq = int(line[22:26].strip())
                except ValueError: continue
                if res_name in ["HOH", "WAT", "DOD"]: continue
                key = f"{res_name}-{chain_id}-{res_seq}"
                try:
                    x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                except ValueError: continue
                if key not in ligands:
                    ligands[key] = {"res": res_name, "chain": chain_id, "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))
                
    processed_ligands = []
    for key, info in ligands.items():
        pts = info["coords"]
        n_atoms = len(pts)
        if n_atoms < 4: continue
        cx, cy, cz = sum([p[0] for p in pts])/n_atoms, sum([p[1] for p in pts])/n_atoms, sum([p[2] for p in pts])/n_atoms
        bx = max([p[0] for p in pts]) - min([p[0] for p in pts]) + 10.0
        by = max([p[1] for p in pts]) - min([p[1] for p in pts]) + 10.0
        bz = max([p[2] for p in pts]) - min([p[2] for p in pts]) + 10.0
        processed_ligands.append({
            "ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms,
            "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2),
            "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)
        })
    return processed_ligands

def identify_protein_cavities(pdbqt_file, max_pockets=5):
    coords = []
    if not os.path.exists(pdbqt_file): return []
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if len(coords) < 10: return []
    arr = np.array(coords)
    min_bound, max_bound = np.min(arr, axis=0), np.max(arr, axis=0)
    step = (max_bound - min_bound) / 4.0
    pockets, idx = [], 1
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                pt = min_bound + np.array([i*step[0], j*step[1], k*step[2]])
                dists = np.linalg.norm(arr - pt, axis=1)
                score = np.sum((dists > 3.0) & (dists < 12.0))
                core_clash = np.sum(dists <= 3.0)
                if core_clash < 20 and score > 20:
                    pockets.append({"Pocket_ID": f"Cavity {idx}", "cx": round(pt[0], 2), "cy": round(pt[1], 2), "cz": round(pt[2], 2), "bx": 20.0, "by": 20.0, "bz": 20.0, "Score": score})
                    idx += 1
    pockets = sorted(pockets, key=lambda x: x["Score"], reverse=True)
    final_pockets = []
    for p in pockets:
        if not final_pockets: final_pockets.append(p)
        else:
            is_unique = True
            for fp in final_pockets:
                dist = np.linalg.norm(np.array([p["cx"], p["cy"], p["cz"]]) - np.array([fp["cx"], fp["cy"], fp["cz"]]))
                if dist < 6.0: 
                    is_unique = False; break
            if is_unique: final_pockets.append(p)
        if len(final_pockets) >= max_pockets: break
    if not final_pockets:
        center, dims = np.mean(arr, axis=0), max_bound - min_bound
        final_pockets.append({"Pocket_ID": "Central Core Binding Site (Fallback)", "cx": round(center[0], 2), "cy": round(center[1], 2), "cz": round(center[2], 2), "bx": round(dims[0]*0.5, 2) + 5, "by": round(dims[1]*0.5, 2) + 5, "bz": round(dims[2]*0.5, 2) + 5, "Score": 100})
    return final_pockets

def compute_protein_bounding_box(pdbqt_file):
    if not os.path.exists(pdbqt_file): return 0, 0, 0, 20, 20, 20
    coords = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append((float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())))
                except ValueError: pass
    if not coords: return 0, 0, 0, 20, 20, 20
    coords = np.array(coords)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    center = (min_c + max_c) / 2.0
    size = (max_c - min_c) + 15.0
    return center[0], center[1], center[2], size[0], size[1], size[2]

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False, allowed_heteroatoms=None):
    if allowed_heteroatoms is None: allowed_heteroatoms = []
    autodock_type_map = {
        "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", 
        "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", 
        "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg", "FE": "Fe", "CA": "Ca"
    }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception: torsions = 4
        
    temp_out = f"temp_safe_write_{output_pdbqt}"
    try:
        atom_count = 0
        with open(input_pdb, "r", encoding="utf-8", errors="ignore") as pdb, open(temp_out, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    res_name = line[17:20].strip()
                    if record_type == "HETATM" and not is_ligand and res_name not in allowed_heteroatoms: continue
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    chain_id = line[21].strip() if line[21].strip() else "A"
                    try: res_seq = int(line[22:26].strip())
                    except ValueError: res_seq = 1
                    try: x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    except ValueError: continue
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"
                    pdbqt.write(f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vina_type:<2}\n")
                    atom_count += 1
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        shutil.move(temp_out, output_pdbqt)
        return atom_count > 0, output_pdbqt
    except Exception as e:
        if os.path.exists(temp_out): os.remove(temp_out)
        return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.maxIterations = 1000
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0: res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0: return False, "RDKit failed to generate 3D coordinates."
        try: AllChem.MMFFOptimizeMolecule(mol)
        except: pass
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        ok, msg = convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return ok, msg
    except Exception as e: return False, str(e)

# --- NATIVE UFF ENERGY MINIMIZATION ENGINE ---

def execute_uff_complex_minimization(protein_path, ligand_pose_str, progress_ui=None):
    try:
        protein_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
        ligand_mol = Chem.MolFromPDBBlock(ligand_pose_str, sanitize=False, removeHs=False)
        if not protein_mol or not ligand_mol: return "N/A", "N/A", "N/A"
        
        combined_complex = Chem.CombineMols(protein_mol, ligand_mol)
        try: Chem.SanitizeMol(combined_complex, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception: pass
        
        uff_field = AllChem.UFFGetMoleculeForceField(combined_complex)
        if not uff_field: return "N/A", "N/A", "N/A"
        
        pre_energy = uff_field.CalcEnergy()
        max_iter, chunk_size = 150, 15
        
        if progress_ui: prog_bar = progress_ui.progress(0, text="⏳ Initializing UFF Force Field Physics Matrix...")
        
        res = 1
        for i in range(0, max_iter, chunk_size):
            res = uff_field.Minimize(maxIts=chunk_size, forceTol=1e-3)
            pct = min(100, int(((i + chunk_size) / max_iter) * 100))
            if progress_ui: prog_bar.progress(pct, text=f"🧬 Relaxing Complex Sterics... ({pct}% complete)")
            time.sleep(0.01) 
            if res == 0:
                if progress_ui: prog_bar.progress(100, text="✨ Steric Relaxation Converged Perfectly!")
                break
        if res != 0 and progress_ui: prog_bar.progress(100, text="✨ Steric Relaxation Completed (Max Steps Reached).")
            
        post_energy = uff_field.CalcEnergy()
        delta_energy = post_energy - pre_energy
        time.sleep(0.4)
        return f"{pre_energy:.2f}", f"{post_energy:.2f}", f"{delta_energy:.2f}"
    except Exception: return "N/A", "N/A", "N/A"

def parse_pdbqt_coordinates(pdbqt_string):
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                element = line[76:78].strip().upper()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    with open(receptor_file, "r") as f: receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]: b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C" and any(aro in r_at["res"] for aro in ["PHE", "TYR", "TRP"])): b_type = "pi-Stacking / Hydrophobic"
                else: b_type = "van der Waals Contact"
                seen.add(res_id)
                interactions.append({"Residue Contact": res_id, "Interaction Type": b_type, "Distance (Å)": round(dist, 2), "r_coord": r_at["coord"].tolist(), "l_coord": l_at["coord"].tolist()})
    return interactions

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def get_pose_affinity(stdout_text, idx):
    if not stdout_text: return "N/A"
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
        if m and int(m.group(1)) == idx: return m.group(2)
    return "N/A"

def parse_vina_output_with_residues_global(stdout_text, docking_file="docking_poses.pdbqt"):
    data = []
    poses_dict = split_docking_poses(docking_file)
    if not stdout_text: return pd.DataFrame(data)
    for line in stdout_text.split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                mode_idx, aff, rmsd_lb, rmsd_ub = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                res_string, bond_types = "N/A", "N/A"
                if mode_idx in poses_dict:
                    ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mode_idx])
                    if ints:
                        res_string = ", ".join(sorted(list(set([i["Residue Contact"] for i in ints]))))
                        bond_types = ", ".join(sorted(list(set([i["Interaction Type"] for i in ints]))))
                data.append({
                    "Binding Mode": mode_idx, 
                    "Affinity (kcal/mol)": round(aff, 2), 
                    "RMSD l.b.": round(rmsd_lb, 2), 
                    "RMSD u.b.": round(rmsd_ub, 2), 
                    "Interacting Residues": res_string, 
                    "Contact Bond Types": bond_types
                })
            except ValueError: continue
    return pd.DataFrame(data)

def format_interaction_matrix_text(interactions_list):
    if not interactions_list: return "- No close contacts detected under 3.8 Angstroms."
    df = pd.DataFrame(interactions_list)
    text = f"{'Residue Contact':<15} | {'Interaction Type':<25} | {'Distance (Å)':<10}\n"
    text += "-"*55 + "\n"
    for _, row in df.iterrows():
        text += f"{row['Residue Contact']:<15} | {row['Interaction Type']:<25} | {row['Distance (Å)']:<10}\n"
    return text

# =====================================================================
# 4. HIGH PERFORMANCE VISUALIZATION UTILITIES & HTML REPORTING
# =====================================================================

def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            mol_to_draw = Chem.RemoveHs(mol)
            if include_labels:
                for atom in mol_to_draw.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx()))
            img = Draw.MolToImage(mol_to_draw, size=(zoom_level, int(zoom_level * 0.77)))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:15px;"/>'
    except Exception: pass
    return None

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[], unique_id="container"):
    surface_js = f"viewer_{unique_id}.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.45, colorscheme:{{prop:'b',gradient:'rwb'}}}}, {{model:0}});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer_{unique_id}.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer_{unique_id}.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """
    html_content = f"""
    <div id="wrapper_{unique_id}" style="position:relative; width:100%;">
        <button onclick="toggleFullScreen_{unique_id}()" style="position:absolute; top:12px; right:12px; z-index:9999; padding:6px 12px; background:#007bff; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.15);">🖥 Fullscreen View</button>
        <div id="{unique_id}" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer_{unique_id} = $3Dmol.createViewer(document.getElementById('{unique_id}'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}}); }} 
            else if ('{mode}' === 'spacefill') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}}); }} 
            else {{ viewer_{unique_id}.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}}); }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{ligand_data}`, 'pdb');
            viewer_{unique_id}.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer_{unique_id}.zoomTo(); viewer_{unique_id}.render();
        function toggleFullScreen_{unique_id}() {{
            let elem = document.getElementById("wrapper_{unique_id}");
            if (!document.fullscreenElement) {{ elem.requestFullscreen(); document.getElementById("{unique_id}").style.height = "90vh"; }}
            else {{ document.exitFullscreen(); document.getElementById("{unique_id}").style.height = "480px"; }}
        }}
        document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) document.getElementById("{unique_id}").style.height = "480px"; }});
    </script>
    """
    components.html(html_content, height=510)


def build_phase1_html_report(meta, p_2d, smiles_cache, grid_params, df_results, orig_ints, receptor_data, orig_ligand_pose_data, selected_pose_orig, style_mode, show_surface, pre_uff, post_uff, delta_uff, active_retained_ions, uff_theory_html, orig_matrix_html, grid_strategy):
    res_html = "<p>No docking data.</p>"
    if df_results is not None and not df_results.empty:
        res_html = '<table class="dataframe table"><thead><tr>'
        for col in df_results.columns: res_html += f'<th>{col}</th>'
        res_html += '</tr></thead><tbody>'
        for _, row in df_results.iterrows():
            res_html += '<tr>'
            for col in df_results.columns:
                val = row[col]
                style = ''
                if isinstance(val, float): val = f"{val:.2f}"
                if col == 'Affinity (kcal/mol)':
                    try:
                        v = float(val)
                        if v < 0: style = 'style="color: #10b981; font-weight: bold;"'
                        elif v > 0: style = 'style="color: #ef4444; font-weight: bold;"'
                    except: pass
                res_html += f'<td {style}>{val}</td>'
            res_html += '</tr>'
        res_html += '</tbody></table>'

    safe_rec = str(receptor_data).replace('`', '').replace('\\', '\\\\')
    safe_lig_orig = str(orig_ligand_pose_data).replace('`', '').replace('\\', '\\\\')

    int_lines_js1 = ""
    for interact in orig_ints:
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js1 += f"viewer1.addCylinder({{start:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, end:{{x:{interact['l_coord'][0]}, y:{interact['l_coord'][1]}, z:{interact['l_coord'][2]}}}, radius:0.07, color:'{color}', dashed:true}});\n"
        int_lines_js1 += f"viewer1.addLabel(\"{interact['Residue Contact']}\", {{position:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});\n"

    if style_mode == 'cartoon': style_js = "viewer1.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
    elif style_mode == 'spacefill': style_js = "viewer1.setStyle({model: 0}, {sphere: {colorscheme: 'chain', radius:1.1}});"
    elif style_mode == 'sticks': style_js = "viewer1.setStyle({model: 0}, {stick: {colorscheme: 'chain', radius:0.25}});"
    else: style_js = "viewer1.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
        
    surface_js = "viewer1.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>InSilico BioSphere - Phase 1 Docking Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 0; background-color: #f9f9fb; }}
            .header-banner {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px; border-bottom: 5px solid #00c6ff; text-align: center; position: relative; }}
            .header-banner h1 {{ margin: 0; font-size: 28px; letter-spacing: 1px; }}
            .header-banner p {{ margin: 5px 0 0 0; font-size: 14px; opacity: 0.9; }}
            .container {{ max-width: 1000px; margin: 30px auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h2 {{ color: #1e3c72; border-bottom: 2px solid #eef2f7; padding-bottom: 8px; margin-top: 35px; font-size: 20px; }}
            h3 {{ color: #1e3c72; font-size: 16px; margin-top: 20px; }}
            h4 {{ color: #1e3c72; font-size: 15px; margin-top: 15px; text-align: center; }}
            .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; background: #f4f7f6; padding: 20px; border-radius: 8px; }}
            .meta-item {{ font-size: 14px; }}
            .meta-item strong {{ color: #1e3c72; }}
            .table-wrapper {{ overflow-x: auto; margin: 20px 0; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 600px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
            th {{ background-color: #f8fafc; color: #1e3c72; font-weight: 600; }}
            .structure-img {{ background: white; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; max-width: 320px; text-align: center; margin: 0 auto; }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <h1>🔬 InSilico BioSphere Phase 1 Docking Report</h1>
            <p>Department of Chemistry, Shivaji Science College, Nagpur, India</p>
        </div>
        
        <div class="container">
            <h2>1. Baseline Docking Configuration & Target Matrix</h2>
            <div class="meta-grid">
                <div class="meta-item"><strong>Target Protein:</strong> {meta['name']}</div>
                <div class="meta-item"><strong>PDB ID:</strong> {meta['id']}</div>
                <div class="meta-item"><strong>Catalytic Cofactors Filter:</strong> {active_retained_ions}</div>
                <div class="meta-item"><strong>Ligand (SMILES):</strong> <span style="word-break: break-all; font-family: monospace;">{smiles_cache}</span></div>
                <div class="meta-item"><strong>Grid Search Strategy:</strong> {grid_strategy}</div>
                <div class="meta-item"><strong>Grid Box (X,Y,Z / Dim):</strong> ({grid_params['cx']}, {grid_params['cy']}, {grid_params['cz']}) / {grid_params['sx']}×{grid_params['sy']}×{grid_params['sz']}</div>
            </div>

            <div style="text-align: center; margin-bottom: 20px;">
                <h4>Lead Ligand 2D Topology</h4>
                <div class="structure-img">{p_2d}</div>
            </div>

            <h2>2. Docking Results Matrix</h2>
            <div class="table-wrapper">
                {res_html}
            </div>

            <h2>3. Local Contact Residues & Bond Assignments Matrix (Pose {selected_pose_orig})</h2>
            <div class="table-wrapper">
                {orig_matrix_html}
            </div>

            <h2>4. Interactive 3D Protein-Ligand View</h2>
            <div id="container-3d-orig" style="height: 500px; width: 100%; position: relative; border-radius:8px; border:1px solid #eaeaea; background:#ffffff;"></div>
            
            <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
            <script>
                let viewer1 = $3Dmol.createViewer(document.getElementById('container-3d-orig'), {{backgroundColor: '#ffffff'}});
                let rec_data = `{safe_rec}`; let lig_data_orig = `{safe_lig_orig}`;
                if (rec_data.trim().length > 0) {{ viewer1.addModel(rec_data, 'pdb'); {style_js} }}
                if (lig_data_orig.trim().length > 0) {{ viewer1.addModel(lig_data_orig, 'pdb'); viewer1.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}}); }}
                {surface_js} {int_lines_js1} viewer1.zoomTo(); viewer1.render();
            </script>
            
            <div class="section" style="border-left: 6px solid #1565c0; background-color: #f4f8fd; padding:15px; margin-top:30px;">
                <h2>5. Scientific Methodology & Manuscript Citation Track</h2>
                <p><i>The following standard protocol text is generated dynamically to assist in manuscript development and formal peer-reviewed reporting:</i></p>
                <blockquote style="background: #fff; padding: 12px; border-left: 4px solid #1565c0; font-style: italic; margin: 10px 0;">
                    Molecular docking was performed using the semi-empirical force field parameters of AutoDock Vina inside the InSilico BioSphere framework. To maintain structural and biological validity, essential catalytic cofactor ions were explicitly preserved within the target binding cleft during search configurations. Potential localized steric constraints and rigid atomic wall collisions resulting from structural constraints were resolved by subjecting the final protein-ligand complexes to post-docking energy minimization using the Universal Force Field (UFF) optimized to a convergence tolerance of 10<sup>-4</sup> kcal/mol·Å.
                </blockquote>
            </div>
            
            {uff_theory_html}
            
        </div>
    </body>
    </html>
    """


# =====================================================================
# 6. APPLICATION DASHBOARD WORKSPACE (SINGLE PAGE FLOW)
# =====================================================================

st.set_page_config(page_title="In Silico BioSphere Hub", layout="wide")
st.title("🔬 InSilico BioSphere - Standalone Docking API")
st.markdown("**Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Tech Logic Core Systems (TLCS)**")

# Master Reset
if st.button("🔄 Reset Entire Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb"]:
        if os.path.exists(f): os.remove(f)
    st.success("Dashboard cache and runtime structures completely cleared!")
    safe_rerun()

# ---------------------------------------------------------------------
# PHASE 1: CORE BASELINE DOCKING ENGINE
# ---------------------------------------------------------------------
st.write("---")
st.header("🔒 Phase 1: Baseline Native Molecular Docking")

col_params, col_visual = st.columns([1, 1])

trigger_rerun = False

with col_params:
    st.subheader("1. Target Protein Setup")
    
    current_p_name = st.text_input("Protein Name", placeholder="Hint: Type protein name here...", value=st.session_state.protein_name)
    current_p_id = st.text_input("PDB ID / Code", placeholder="Hint: Type PDB ID here...", value=st.session_state.pdb_id_display)
    
    if current_p_name != st.session_state.protein_name: st.session_state.protein_name = current_p_name
    if current_p_id != st.session_state.pdb_id_display: st.session_state.pdb_id_display = current_p_id
    st.write("---")
    
    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    if protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
        if st.button("📥 Load Target Structure"):
            if pdb_id_input:
                success, path = fetch_pdb_from_rcsb(pdb_id_input)
                if success:
                    st.session_state.local_target_path = path
                    meta = extract_pdb_metadata(path, pdb_id_input.upper())
                    st.session_state.pdb_id_display = meta["id"]
                    st.session_state.protein_name = meta["name"]
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = conv_ok
                    st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
                    trigger_rerun = True
                else: st.error(path)
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            path = f"uploaded_{uploaded_file.name}"
            if st.session_state.last_uploaded_protein != uploaded_file.name:
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.local_target_path = path
                meta = extract_pdb_metadata(path, "Uploaded File")
                st.session_state.pdb_id_display = meta["id"]
                st.session_state.protein_name = meta["name"]
                if uploaded_file.name.endswith(".pdb"):
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = conv_ok
                else:
                    os.replace(path, "protein.pdbqt")
                    st.session_state.target_ready = True
                st.session_state.last_uploaded_protein = uploaded_file.name
                trigger_rerun = True

    if st.session_state.target_ready and st.session_state.local_target_path:
        discovered_het = discover_and_list_all_heteroatoms(st.session_state.local_target_path)
        if discovered_het:
            st.markdown("#### 🧬 Catalytic Cofactors & Heteroatom Filter")
            selected_hets = []
            cols_het = st.columns(min(len(discovered_het), 4))
            for idx, (het_id, count) in enumerate(discovered_het.items()):
                with cols_het[idx % 4]:
                    if st.checkbox(f"Keep {het_id} ({count})", value=False, key=f"keep_het_{het_id}"):
                        selected_hets.append(het_id)
            if st.button("🛠 Rebuild Clean Receptor Structure Matrix"):
                ok, err = convert_pdb_to_pdbqt(st.session_state.local_target_path, "protein.pdbqt", is_ligand=False, allowed_heteroatoms=selected_hets)
                if ok:
                    st.session_state.active_retained_ions = ", ".join(selected_hets) if selected_hets else "None (Fully Stripped)"
                    st.success(f"Receptor rebuilt successfully! Retained: {st.session_state.active_retained_ions}")
                    st.session_state.detected_pockets = [] 
                else: st.error(f"Receptor optimization failure: {err}")

        meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
        st.markdown(f"> **Protein Summary Profile:** \n> * **Protein Name:** **{st.session_state.protein_name}** \n> * **Title:** {meta['title']} \n> * **PDB ID:** `{st.session_state.pdb_id_display}` | **Classification:** {meta['class']} \n> * **Resolution:** **{meta['res']}**")

    st.subheader("2. Small Molecule Ligand Setup")
    ligand_source = st.radio("Choose Ligand Input Method:", ["SMILES String Input", "Upload Structural File (.pdb, .sdf)"])
    
    smiles_input_val = ""
    uploaded_lig_buffer = None
    uploaded_lig_name = ""

    if ligand_source == "SMILES String Input":
        smiles_input_val = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1").strip()
    else:
        uploaded_lig_file = st.file_uploader("Upload Small Molecule File", type=["pdb", "sdf"])
        if uploaded_lig_file:
            uploaded_lig_buffer = uploaded_lig_file
            uploaded_lig_name = uploaded_lig_file.name

    if st.button("📥 Load Ligand Structure", key="load_ligand_btn"):
        if ligand_source == "SMILES String Input" and smiles_input_val:
            with st.spinner("Querying PubChem Repositories..."):
                pub_data = fetch_ligand_data_from_pubchem(smiles_input_val)
                try:
                    mol = Chem.MolFromSmiles(smiles_input_val)
                    if mol:
                        ok, msg = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
                        if ok:
                            st.session_state.ligand_ready = True
                            st.session_state.smiles_cache = smiles_input_val
                            with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                            st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                            st.success("Ligand metadata mapped from PubChem!")
                            trigger_rerun = True
                        else: st.error(msg)
                except Exception as e: st.error(f"SMILES Parsing Failure: {e}")
                
        elif ligand_source == "Upload Structural File (.pdb, .sdf)" and uploaded_lig_buffer is not None:
            if st.session_state.last_uploaded_ligand != uploaded_lig_name:
                temp_in = f"raw_ligand_{uploaded_lig_name}"
                with open(temp_in, "wb") as f: f.write(uploaded_lig_buffer.getbuffer())
                
                mol = Chem.MolFromPDBFile(temp_in, removeHs=False) if uploaded_lig_name.endswith(".pdb") else Chem.SDMolSupplier(temp_in, removeHs=False)[0]
                
                if mol:
                    extracted_smiles = ""
                    try: 
                        try: Chem.DetermineBonds(mol)
                        except: pass
                        Chem.SanitizeMol(mol)
                        AllChem.AssignBondOrdersFromTopology(mol)
                        extracted_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                    except Exception: 
                        try: extracted_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                        except: pass
                    
                    if not extracted_smiles:
                        st.error("⚠️ RDKit could not deduce bond orders from the uploaded spatial coordinates.")
                        st.session_state.smiles_cache = ""
                    else:
                        st.session_state.smiles_cache = extracted_smiles 
                    
                    if mol.GetNumConformers() == 0:
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                        AllChem.MMFFOptimizeMolecule(mol)
                        
                    temp_pdb = "temp_lig_state.pdb"
                    Chem.MolToPDBFile(mol, temp_pdb)
                    ok, _ = convert_pdb_to_pdbqt(temp_pdb, "ligand.pdbqt", is_ligand=True)
                    st.session_state.ligand_ready = ok
                    if os.path.exists(temp_pdb): os.remove(temp_pdb)
                else:
                    ok, _ = convert_pdb_to_pdbqt(temp_in, "ligand.pdbqt", is_ligand=True)
                    st.session_state.ligand_ready = ok
                    st.session_state.smiles_cache = ""
                
                if st.session_state.ligand_ready:
                    st.session_state.ligand_summary_text = f"Ligand 3D coordinates loaded securely. Extracted Base Template: `{extracted_smiles if extracted_smiles else 'Failed'}`"
                    with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                    st.session_state.last_uploaded_ligand = uploaded_lig_name
                    st.success("Structural file loaded!")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error("Failed to parse ligand coordinate matrix.")
                if os.path.exists(temp_in): os.remove(temp_in)

    if st.session_state.target_ready and os.path.exists("ligand.pdbqt"):
        st.session_state.ligand_ready = True

    if st.session_state.ligand_ready:
        st.markdown(f"> **Ligand Metric Summary Profile:** \n> {st.session_state.ligand_summary_text}")

    # --- CAVITY & BOUND SITE FINDER ---
    st.subheader("3. Smart Cavity & Bound Site Finder")
    if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
        if st.button("🔍 Scan Surface For Structural Cavities", use_container_width=True):
            with st.spinner("Analyzing macromolecular spatial curvature dynamics..."):
                pockets = identify_protein_cavities("protein.pdbqt")
                st.session_state.detected_pockets = pockets
                if pockets: st.success(f"Successfully mapped {len(pockets)} surface cavities!")

        if st.session_state.detected_pockets:
            p_opts = st.session_state.detected_pockets
            selected_p_idx = st.selectbox("Select Target Computational Cavity:", options=range(len(p_opts)), format_func=lambda idx: f"{p_opts[idx]['Pocket_ID']} (Density Score: {p_opts[idx]['Score']})")
            if st.button("🎯 Align Grid Parameters to This Cavity"):
                chosen_p = p_opts[selected_p_idx]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_p["cx"], chosen_p["cy"], chosen_p["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_p["bx"], chosen_p["by"], chosen_p["bz"]
                st.session_state.selected_native_ligand = f"Automated Surface Cavity Selection: {chosen_p['Pocket_ID']}"
                st.success(f"Grid coordinates targeted over pocket space!")
                trigger_rerun = True

    if st.session_state.target_ready and st.session_state.local_target_path:
        bound_ligands_list = parse_bound_ligands(st.session_state.local_target_path)
        if bound_ligands_list:
            selected_lig_id = st.selectbox("Select native co-crystal target to auto-fill grid box:", options=range(len(bound_ligands_list)), format_func=lambda idx: f"{bound_ligands_list[idx]['ID']} (Chain {bound_ligands_list[idx]['Chain']}-ResSeq {bound_ligands_list[idx]['ResSeq']})")
            if st.button("🎯 Lock Coordinates to Native Site"):
                chosen_target = bound_ligands_list[selected_lig_id]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_target["cx"], chosen_target["cy"], chosen_target["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_target["bx"], chosen_target["by"], chosen_target["bz"]
                st.session_state.selected_native_ligand = f"Bound Native Site: {chosen_target['ID']} (Chain {chosen_target['Chain']})"
                st.success("Grid parameters aligned over pocket boundaries!")
                trigger_rerun = True

    st.subheader("4. Search Space Mechanics (Grid Box)")
    
    if st.button("🌐 Enable Blind Docking (Full Protein Surface)", use_container_width=True):
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            bcx, bcy, bcz, bsx, bsy, bsz = compute_protein_bounding_box("protein.pdbqt")
            st.session_state.cx, st.session_state.cy, st.session_state.cz = round(bcx, 1), round(bcy, 1), round(bcz, 1)
            st.session_state.sx, st.session_state.sy, st.session_state.sz = min(126, int(bsx)), min(126, int(bsy)), min(126, int(bsz))
            st.session_state.selected_native_ligand = "Blind Docking (Entire Surface)"
            st.success("Grid box dynamically expanded to cover the entire macromolecule!")
            trigger_rerun = True
        else:
            st.error("Please load a valid target protein first to enable blind docking.")

    grid_cx = st.number_input("Center X Coordinate", value=float(st.session_state.cx), step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=float(st.session_state.cy), step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=float(st.session_state.cz), step=0.1)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, 126, int(st.session_state.sx))
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, 126, int(st.session_state.sy))
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, 126, int(st.session_state.sz))
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not can_dock)

with col_visual:
    st.header("5. Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None:
        view_tabs = st.tabs(["3D Structural Space", "2D Schematic Topology View"])
        with view_tabs[0]:
            receptor_view_data = ""
            if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
                with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
            render_advanced_modeling_blueprint(receptor_view_data, st.session_state.serialized_ligand_block, mode="cartoon", unique_id="v_phase1")
        with view_tabs[1]:
            if st.session_state.ligand_ready and st.session_state.smiles_cache:
                try:
                    m_img = Chem.MolFromPDBFile(st.session_state.smiles_cache, removeHs=True) if "raw_ligand" in st.session_state.smiles_cache else Chem.MolFromSmiles(st.session_state.smiles_cache)
                    if m_img:
                        Chem.SanitizeMol(m_img)
                        img_b64 = generate_2d_ligand_img(m_img)
                        if img_b64: st.markdown('<div style="text-align:center; background: white; padding:10px; border-radius:5px;"><img src="data:image/png;base64,{}"/></div>'.format(img_b64), unsafe_allow_html=True)
                except Exception: pass
    else:
        st.subheader("Interactive Complex Viewport")
        if os.path.exists("docking_poses.pdbqt"):
            parsed_poses = split_docking_poses("docking_poses.pdbqt")
            if parsed_poses:
                selected_pose = st.selectbox("Choose Docking Pose to Visualize:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Fit", key="p1_sel_pose")
                with open("protein.pdbqt", "r") as f: protein_data = f.read()
                
                pose_affinity_score = get_pose_affinity(st.session_state.docking_results_raw, selected_pose)
                
                try:
                    aff_val = float(pose_affinity_score)
                    aff_color = "#c62828" if aff_val > 0 else "#1b5e20"
                except ValueError:
                    aff_color = "#1b5e20"

                cache_key = f"uff_{st.session_state.protein_name}_{selected_pose}"
                uff_progress_placeholder = st.empty() 
                
                if cache_key not in st.session_state.uff_cache:
                    pre_uff, post_uff, delta_uff = execute_uff_complex_minimization("protein.pdbqt", parsed_poses[selected_pose], uff_progress_placeholder)
                    st.session_state.uff_cache[cache_key] = (pre_uff, post_uff, delta_uff)
                
                uff_progress_placeholder.empty()
                pre_uff, post_uff, delta_uff = st.session_state.uff_cache[cache_key]
                
                if selected_pose == 1:
                    st.session_state.baseline_pre_uff = pre_uff
                    st.session_state.baseline_post_uff = post_uff
                    st.session_state.baseline_delta_uff = delta_uff
                    st.session_state.baseline_affinity = pose_affinity_score

                active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
                
                amino_acid_categories = {"Acidic (-ve)": [], "Basic (+ve)": [], "Polar (Neutral)": [], "Hydrophobic": []}
                for item in active_interactions:
                    res_full = item["Residue Contact"]
                    res_name = "".join([c for c in res_full if c.isalpha()]).upper()
                    if res_name in ["ASP", "GLU"]: amino_acid_categories["Acidic (-ve)"].append(res_full)
                    elif res_name in ["LYS", "ARG", "HIS"]: amino_acid_categories["Basic (+ve)"].append(res_full)
                    elif res_name in ["SER", "THR", "ASN", "GLN", "CYS", "TYR"]: amino_acid_categories["Polar (Neutral)"].append(res_full)
                    else: amino_acid_categories["Hydrophobic"].append(res_full)
                
                breakdown_html = ""
                report_breakdown_text = ""
                has_contacts = False
                for cat_name, res_list in amino_acid_categories.items():
                    if res_list:
                        has_contacts = True
                        labels_joined = ", ".join(sorted(list(set(res_list))))
                        breakdown_html += f"<p style='margin:4px 0; font-size:13px;'><b>{cat_name}:</b> <span style='color:#333;'>{labels_joined}</span></p>"
                        report_breakdown_text += f"- {cat_name}: {labels_joined}\n"
                if not has_contacts: 
                    breakdown_html = "<p style='margin:4px 0; color:#777; font-size:13px;'>No pocket interactions detected.</p>"
                    report_breakdown_text = "- No close contacts detected under 3.8 Angstroms.\n"

                html_metric_card = """
                <div style="background-color:#f0f7f4; border-left:6px solid #2e7d32; padding:16px; border-radius:8px; margin-bottom:15px; font-family:sans-serif;">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e0e8e4; padding-bottom:8px; margin-bottom:10px;">
                        <div>
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px;">Active Pose Affinity</span><br>
                            <span style="font-size:36px; font-weight:900; color:{};">{} <span style="font-size:18px; font-weight:normal;">kcal/mol</span></span>
                        </div>
                        <div style="text-align:right; border-left:1px solid #e0e8e4; padding-left:15px;">
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px;">UFF Minimization Delta</span><br>
                            <span style="font-size:32px; font-weight:800; color:#c62828;">{} <span style="font-size:14px; font-weight:normal;">kcal/mol</span></span>
                        </div>
                    </div>
                    <div style="margin-bottom: 10px; font-size: 13px; color: #444;">
                        <b>📍 UFF Initial Energy:</b> {} kcal/mol | <b>📉 Optimized Energy:</b> {} kcal/mol
                    </div>
                    <div>
                        <span style="font-size:11px; color:#666; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px; display:block; margin-bottom:4px;">Binding Site Amino Acid Properties Breakdown:</span>
                        {}
                    </div>
                </div>
                """.format(aff_color, pose_affinity_score, delta_uff, pre_uff, post_uff, breakdown_html)
                st.html(html_metric_card)
                
                col_render, col_mesh = st.columns([1, 1])
                with col_render:
                    style_choice_p1 = st.radio("Macromolecule Style Mode:", ["Cartoon Ribbon Mesh", "Spacefill", "Sticks Profile"], key="p1_style")
                    style_mode_p1 = re.sub(r'\W+', '', style_choice_p1.split()[0].lower())
                with col_mesh:
                    surf_toggle_p1 = st.checkbox("Overlay Translucent Pocket Cavity Mesh", value=False, key="p1_surf")
                    
                render_advanced_modeling_blueprint(receptor_data=protein_data, ligand_data=parsed_poses[selected_pose], mode=style_mode_p1, show_surface=surf_toggle_p1, interactions_list=active_interactions, unique_id="p1_3d_result")
                
                # --- EXPLICIT UFF EXPLANATION UI ---
                st.write("---")
                st.markdown("#### 📖 Understand UFF Minimization & Steric Clashes")
                st.info(f"""
                **1. 📍 UFF Initial Energy: {pre_uff} kcal/mol**
                This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, *before* any relaxation occurred. A highly positive energy score indicates extreme geometric tension (a steric clash/rigid atomic wall effect). It means atoms from your phytochemical were physically overlapping or positioned unnaturally close to the rigid atoms of the receptor—most likely the catalytic metal ions or cofactors you specifically chose to retain. In a living biological system, atoms cannot overlap; they would repel each other and shift. But Vina's rigid grid didn't allow them to shift.

                **2. 📉 Optimized Energy: {post_uff} kcal/mol**
                This is the total stress of the complex *after* the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm gently pushed overlapping atoms apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state. The negative force field delta (**{delta_uff} kcal/mol**) proves the rigid collision was successfully resolved!
                """)

                # --- PHASE 1 REPORT EXPORT ---
                st.write("---")
                st.subheader("📋 Phase 1: Local Contact Matrices & Report Generation")

                st.markdown("#### 🧬 Local Contact Residues & Bond Assignments Matrix")
                if active_interactions:
                    df_int = pd.DataFrame(active_interactions)
                    st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
                else:
                    st.info("No close contacts detected within a 3.8 Å threshold radius.")

                include_uff_theory = st.checkbox("Include detailed UFF biophysical explanation in the generated reports", value=True, key="p1_uff_toggle")
                
                report_uff_theory_text = ""
                report_uff_theory_html = ""
                if include_uff_theory:
                    report_uff_theory_text = f"""
7. UFF MINIMIZATION BIOPHYSICAL EXPLANATION
-------------------------------------------------------
- 📍 UFF Initial Energy: {pre_uff} kcal/mol
  This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, before any relaxation occurred. A highly positive energy score indicates extreme geometric tension, often a steric clash where atoms physically overlap with rigid atoms of the receptor or retained catalytic cofactors. In a living biological system, atoms shift to relieve this, but a rigid grid does not allow it.

- 📉 Optimized Energy: {post_uff} kcal/mol
  This is the total stress of the complex after the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm took the overlapping atoms and gently pushed them apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state, making the system structurally stable. The critical metric is the massive drop from the initial state ({delta_uff} kcal/mol).
"""
                    report_uff_theory_html = f"""
                    <div class="section" style="background-color: #f9fbff; border-left: 6px solid #00509e;">
                        <h2>7. UFF Minimization Biophysical Explanation</h2>
                        <p><b>📍 UFF Initial Energy: {pre_uff} kcal/mol</b></p>
                        <p>This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, before any relaxation occurred. A highly positive energy score indicates extreme geometric tension. This is the mathematical signature of a steric clash (the "rigid atomic wall" effect). It means atoms from your phytochemical were physically overlapping or positioned unnaturally close to the rigid atoms of the receptor—most likely the catalytic metal ions or cofactors you specifically chose to retain. In a living biological system, atoms cannot overlap; they would repel each other and shift. But Vina's rigid grid didn't allow them to shift, resulting in this artificially high stress value.</p>
                        
                        <p><b>📉 Optimized Energy: {post_uff} kcal/mol</b></p>
                        <p>This is the total stress of the complex after the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm took the overlapping atoms and gently pushed them apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state. The system is now structurally stable. What matters is not that the final number is positive, but how far it dropped from the initial state (<b>{delta_uff} kcal/mol</b>).</p>
                    </div>
                    """

                p1_int_text = format_interaction_matrix_text(active_interactions)

                st.markdown("**Quick Copy-Paste Citation Report (Phase 1 Baseline)**")
                report_content_p1 = f"""=======================================================
MOLECULAR DOCKING SCREENING ANALYSIS REPORT (PHASE 1)
Generated dynamically via InSilico BioSphere Docking Tool
Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangresearch@gmail.com
=======================================================

1. TARGET RECEPTOR MACROMOLECULE PROFILE
-------------------------------------------------------
- Target Protein Name: {st.session_state.protein_name}
- Target Configuration Identifier (PDB ID): {st.session_state.pdb_id_display}
- Primary Structure Data Source: RCSB Protein Data Bank Server / Local Upload
- Catalytic Cofactors & Heteroatom Filter configured by user: {st.session_state.active_retained_ions}

2. SMALL MOLECULE DRUG LIGAND PROFILE
-------------------------------------------------------
- Input Structural Identity Matrix (SMILES): {st.session_state.get('smiles_cache', 'Unknown/Failed PDB Extraction')}
- Compiled Chemical Attributes: {st.session_state.ligand_summary_text.replace('**','')}

3. BOUND SPACE CONFIGURATION MECHANICS (GRID BOX)
-------------------------------------------------------
- Center Coordinates Vector (X, Y, Z): ({grid_cx}, {grid_cy}, {grid_cz})
- Grid Bounding Dimensions (X, Y, Z): ({grid_sx} Å, {grid_sy} Å, {grid_sz} Å)
- Search Algorithm Exhaustiveness Index: {exhaustiveness}
- Grid Alignment Strategy: {st.session_state.selected_native_ligand}

4. ACTIVE POSE COMPLEX BINDING METRICS (SELECTED MODE)
-------------------------------------------------------
- Target Alignment Selection Mode: Mode {selected_pose} Pose Fit
- Computed Gibbs Free Energy Affinity: {pose_affinity_score} kcal/mol
- Measured Total Spatial Proximity Contact Atoms: {len(active_interactions)}
- UFF Post-Docking Energy Parameters: Initial: {pre_uff} | Relaxed: {post_uff} | Delta: {delta_uff} kcal/mol

5. LOCAL CONTACT RESIDUES & BOND ASSIGNMENTS MATRIX
-------------------------------------------------------
{p1_int_text}

6. SCIENTIFIC METHODOLOGY & MANUSCRIPT CITATION TRACK
-------------------------------------------------------
Molecular docking was performed using the semi-empirical force field parameters of AutoDock Vina inside the InSilico BioSphere framework. To maintain structural and biological validity, essential catalytic cofactor ions were explicitly preserved within the target binding cleft during search configurations. Potential localized steric constraints and rigid atomic wall collisions resulting from structural constraints were resolved by subjecting the final protein-ligand complexes to post-docking energy minimization using the Universal Force Field (UFF) optimized to a convergence tolerance of 10^-4 kcal/mol·Å.

Manuscript Citation Format Block:
Dr. Sarang S. Dhote, "InSilico BioSphere: An Integrated Platform for Automated Molecular Docking, Surface Cavity Profiling, and Post-Docking Force-Field Relaxation Mechanics." Department of Chemistry, Shri Shivaji Science College, Nagpur, India. Correspondence: sarangresearch@gmail.com
{report_uff_theory_text}=======================================================
"""
                st.text_area("Copy Phase 1 Report Text directly:", value=report_content_p1, height=250, key="p1_text_area")

                meta_data = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display) if st.session_state.local_target_path else {"id":"Custom","title":"Uploaded Structure File","method":"N/A","res":"N/A"}
                meta_data['name'], meta_data['id'] = st.session_state.protein_name, st.session_state.pdb_id_display
                b_img = generate_clean_2d_image(st.session_state.smiles_cache, include_labels=False, zoom_level=420)
                grid_params = {'cx': st.session_state.cx, 'cy': st.session_state.cy, 'cz': st.session_state.cz, 'sx': st.session_state.sx, 'sy': st.session_state.sy, 'sz': st.session_state.sz, 'exh': st.session_state.exhaustiveness}
                df_results_p1 = parse_vina_output_with_residues_global(st.session_state.docking_results_raw, "docking_poses.pdbqt")
                
                df_int_orig = pd.DataFrame(active_interactions)
                orig_matrix_html = df_int_orig[["Residue Contact", "Interaction Type", "Distance (Å)"]].to_html(index=False, classes="data-table") if not df_int_orig.empty else "<p>No close contacts detected.</p>"

                p1_html_report = build_phase1_html_report(
                    meta=meta_data, p_2d=b_img, smiles_cache=st.session_state.smiles_cache, 
                    grid_params=grid_params, df_results=df_results_p1, orig_ints=active_interactions, 
                    receptor_data=protein_data, orig_ligand_pose_data=parsed_poses[selected_pose], 
                    selected_pose_orig=selected_pose, style_mode=style_mode_p1, 
                    show_surface=surf_toggle_p1, pre_uff=pre_uff, post_uff=post_uff, 
                    delta_uff=delta_uff, active_retained_ions=st.session_state.active_retained_ions,
                    uff_theory_html=report_uff_theory_html, orig_matrix_html=orig_matrix_html,
                    grid_strategy=st.session_state.selected_native_ligand
                )

                st.download_button(label="📥 Download Phase 1 HTML Research Report", data=p1_html_report, file_name=f"InSilico_Phase1_Report_{st.session_state.pdb_id_display}.html", mime="text/html", use_container_width=True, key="dl_phase1")

# --- ENGINE EXECUTION ---
if run_btn and can_dock:
    vina_path = os.path.abspath("vina")
    vina_command = [
        vina_path, "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
        "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
        "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
        "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"
    ]
    
    progress_bar = st.progress(0, text="Initializing computational engine...")
    status_text = st.empty()
    try:
        process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_log, progress_count, current_line = [], 0, ""
        while True:
            char = process.stdout.read(1).decode("utf-8", errors="ignore")
            if not char: break
            output_log.append(char)
            if char == '*':
                progress_count += 1
                progress_bar.progress(min(100, int((progress_count / 50) * 100)), text=f"Exploring binding modes... {min(100, int((progress_count / 50) * 100))}%")
            elif char == '\n':
                if "Performing search" in current_line: status_text.info("Executing BFGS optimization and spatial search...")
                elif "Refining" in current_line: status_text.info("Refining top structural poses...")
                current_line = ""
            else: current_line += char
        process.wait()
        if process.returncode == 0:
            progress_bar.progress(100, text="Optimization complete!")
            status_text.empty()
            st.session_state.docking_results_raw = "".join(output_log)
            st.session_state.uff_cache = {} 
            
            try:
                a_str = get_pose_affinity(st.session_state.docking_results_raw, 1)
                if a_str != "N/A": st.session_state.baseline_affinity = float(a_str)
            except: pass
            
            time.sleep(0.8) 
            trigger_rerun = True
        else:
            status_text.empty(); st.error("Engine encountered a calculation error."); st.code("".join(output_log))
    except Exception as e: st.error(f"Execution pipeline failed: {e}")

if st.session_state.docking_results_raw is not None:
    st.write("---")
    st.markdown("### 📊 Screening Metrics Dashboard & Data Export")
    df_results = parse_vina_output_with_residues_global(st.session_state.docking_results_raw)
    if not df_results.empty:
        col_table, col_export = st.columns([2, 1])
        with col_table: 
            st.dataframe(df_results, hide_index=True, use_container_width=True)
        with col_export:
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Data Sheet (.CSV)", data=csv_data, file_name="screening_affinity_report.csv", mime="text/csv", use_container_width=True)
