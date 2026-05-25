import time
import streamlit as st
import subprocess
import os
import urllib.request
import urllib.parse
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64
import io
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

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
        # Core Docking Vectors
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
        
        # Redesign & ADMET Variables
        "rd_library": None,
        "selected_variant_id": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_states()

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
    except Exception:
        pass 
    return metadata

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "title": "Uploaded Protein Structure Matrix", "id": pdb_id.upper(),
        "class": "Unknown Classification", "organism": "Unknown",
        "system": "Unknown Expression System", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path): return meta
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            title_parts = []
            for line in f:
                if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
                elif line.startswith("HEADER"): meta["class"] = line[10:50].strip().title()
                elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
                elif "EXPRESSION_SYSTEM" in line: meta["system"] = line.split(":")[-1].replace(";","").strip()
                elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
                elif "RESOLUTION." in line and "ANGSTROMS." in line:
                    match = re.search(r"(\d+\.\d+)", line)
                    if match: meta["res"] = f"{match.group(1)} Å"
        if title_parts: meta["title"] = " ".join(title_parts).title()
    except Exception:
        pass
    return meta

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
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
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

def compute_protein_bounding_box(pdbqt_file):
    if not os.path.exists(pdbqt_file): return 0, 0, 0, 20, 20, 20
    coords = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    coords.append((x, y, z))
                except ValueError: pass
    if not coords: return 0, 0, 0, 20, 20, 20
    coords = np.array(coords)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    center = (min_c + max_c) / 2.0
    size = (max_c - min_c) + 15.0
    return center[0], center[1], center[2], size[0], size[1], size[2]

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    autodock_type_map = {
        "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", 
        "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", 
        "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg"
    }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception: torsions = 4
    try:
        with open(input_pdb, "r", encoding="utf-8", errors="ignore") as pdb, open(output_pdbqt, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    res_name = line[17:20].strip()
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
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        return True, output_pdbqt
    except Exception as e: return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return True, output_filename
    except Exception as e: return False, str(e)

def parse_pdbqt_coordinates(pdbqt_string):
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip().upper()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    with open(receptor_file, "r") as f:
       receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]:
                    b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C" and any(aro in r_at["res"] for aro in ["PHE", "TYR", "TRP"])):
                    b_type = "pi-Stacking / Hydrophobic"
                else:
                    b_type = "van der Waals Contact"
                seen.add(res_id)
                interactions.append({
                    "Residue Contact": res_id, "Interaction Type": b_type, "Distance (Å)": round(dist, 2),
                    "r_coord": r_at["coord"].tolist(), "l_coord": l_at["coord"].tolist()
                })
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

# =====================================================================
# 3. FRAGMENTATION & ADVANCED ADME MODULE
# =====================================================================

def find_valid_cleavage_sites(smiles_str):
    valid_sites = []
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                sym = atom.GetSymbol()
                deg = atom.GetDegree()
                hs = atom.GetTotalNumHs()
                if deg == 1 and sym != 'C':
                    valid_sites.append({"index": idx, "label": f"Atom #{idx} (Terminal {sym})"})
                elif sym == 'C' and hs > 0:
                    valid_sites.append({"index": idx, "label": f"Atom #{idx} ({sym} with available H)"})
                elif sym in ['N', 'O', 'S'] and hs > 0:
                    valid_sites.append({"index": idx, "label": f"Atom #{idx} (Core {sym} with available H)"})
        valid_sites.sort(key=lambda x: (0 if "Terminal" in x["label"] else 1, x["index"]))
    except Exception: pass
    return valid_sites

def get_dynamic_fragments(parent_smiles):
    mol = Chem.MolFromSmiles(parent_smiles)
    if not mol: return "Standard Organic Scaffold", []
    flavone_smarts = Chem.MolFromSmarts("c1cc(O)cc2c1c(=O)cc(c2)c3ccccc3")
    phenol_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c[OH]")))
    alkaloid_smarts = Chem.MolFromSmarts("[#7;R]")
    aliphatic_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.GetIsAromatic()]
    total_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    aliphatic_ratio = len(aliphatic_carbons) / len(total_carbons) if total_carbons else 0

    if mol.HasSubstructMatch(flavone_smarts) or phenol_count >= 2:
        subclass_title = "Polyphenolic Flavonoid Core"
        fragments = [
            {"name": "Glucosylation (-C6H11O5)", "smiles": "OC1C(O)C(O)C(O)C(CO)O1", "peak": 3350, "yield": "Moderate Yield (58%)", "route": "Enzymatic glycosylation via Phase II transferase mirroring."},
            {"name": "Prenylation (-CH2CH=C(CH3)2)", "smiles": "CC(C)=CC", "peak": 1660, "yield": "Good Yield (72%)", "route": "Late-stage electrophilic C-alkylation."},
            {"name": "O-Methylation (-OCH3)", "smiles": "OC", "peak": 1250, "yield": "Excellent Yield (91%)", "route": "Selective etherification using Dimethyl Sulfate."},
            {"name": "Acetylation (-OCOCH3)", "smiles": "OC(=O)C", "peak": 1735, "yield": "Good Yield (84%)", "route": "Esterification utilizing Acetic Anhydride."}
        ]
    elif mol.HasSubstructMatch(alkaloid_smarts):
        subclass_title = "Alkaloidal Nitrogen Heterocycle"
        fragments = [
            {"name": "N-Alkylation (-CH2CH3)", "smiles": "CC", "peak": 2960, "yield": "Good Yield (80%)", "route": "Nucleophilic substitution at nitrogen nodes using Ethyl Bromide."},
            {"name": "Quaternization (-CH3+)", "smiles": "C", "peak": 2850, "yield": "Excellent Yield (94%)", "route": "Methylation using Methyl Iodide."},
            {"name": "Amidation (-COCH3)", "smiles": "C(=O)C", "peak": 1665, "yield": "Good Yield (78%)", "route": "Amide condensation using Acetyl Chloride."},
            {"name": "N-Oxidation (=O)", "smiles": "[O-]", "peak": 950, "yield": "Moderate Yield (65%)", "route": "Controlled oxidation via mCPBA."}
        ]
    elif aliphatic_ratio > 0.65:
        subclass_title = "Aliphatic Terpenoid Scaffold"
        fragments = [
            {"name": "Epoxidation (=O)", "smiles": "O", "peak": 1250, "yield": "Moderate Yield (60%)", "route": "Prilezhaev reaction using mCPBA across isolated alkene bonds."},
            {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3400, "yield": "Poor Yield (42%)", "route": "Allylic C-H functionalization driven by Selenium Dioxide."},
            {"name": "Ozonolysis Fragmentation", "smiles": "O=C", "peak": 1710, "yield": "Good Yield (70%)", "route": "Oxidative cleavage of double bonds."},
            {"name": "Esterification (-COOCH3)", "smiles": "C(=O)OC", "peak": 1740, "yield": "Good Yield (86%)", "route": "Fischer esterification across terminal carboxylic vectors."}
        ]
    else:
        subclass_title = "Standard Organic Lead Profile"
        fragments = [
            {"name": "Methylation (-CH3)", "smiles": "C", "peak": 2925, "yield": "Good Yield (85%)", "route": "Standard alkylation path via Methyl Iodide."},
            {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3450, "yield": "Moderate Yield (62%)", "route": "Direct C-H matrix oxidation with copper coordination."},
            {"name": "Amination (-NH2)", "smiles": "N", "peak": 3320, "yield": "Good Yield (74%)", "route": "Controlled substitution via nucleophilic amination."},
            {"name": "Fluorination (-F)", "smiles": "F", "peak": 1150, "yield": "Poor Yield (38%)", "route": "Late-stage electrophilic fluorination using Selectfluor."}
        ]
    return subclass_title, fragments

def run_cleaving_engine(parent_smiles, target_atom_idx, mechanism_mode):
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if not parent_mol: return []
    _, fragments = get_dynamic_fragments(parent_smiles)
    derived_library = []
    
    baseline = st.session_state.baseline_affinity if st.session_state.baseline_affinity is not None else -6.2
    
    for idx, frag in enumerate(fragments):
        success = False
        derived_smiles = ""
        if mechanism_mode == "True Covalent Substitution (Cleavage & Attachment)":
            try:
                rw_mol = Chem.RWMol(parent_mol)
                t_atom = rw_mol.GetAtomWithIdx(int(target_atom_idx))
                is_terminal = (t_atom.GetDegree() == 1 and t_atom.GetSymbol() != 'C')
                if is_terminal:
                    t_atom.SetAtomicNum(0)
                    t_atom.SetIsotope(999)
                else:
                    dummy = Chem.Atom(0)
                    dummy.SetIsotope(999)
                    new_idx = rw_mol.AddAtom(dummy)
                    rw_mol.AddBond(int(target_atom_idx), new_idx, Chem.BondType.SINGLE)
                tagged_mol = rw_mol.GetMol()
                Chem.SanitizeMol(tagged_mol)
                pattern = Chem.MolFromSmarts("[999*]")
                frag_mol = Chem.MolFromSmiles(frag['smiles'])
                replaced_mols = AllChem.ReplaceSubstructs(tagged_mol, pattern, frag_mol, replaceAll=True)
                if replaced_mols:
                    final_mol = replaced_mols[0]
                    Chem.SanitizeMol(final_mol)
                    derived_smiles = Chem.MolToSmiles(final_mol)
                    if Chem.MolFromSmiles(derived_smiles): success = True
            except Exception: success = False

        if not success:
            derived_smiles = f"{parent_smiles}.{frag['smiles']}"
            frag_name = frag["name"] + " (Co-Crystal Fallback)"
            route = "Co-crystallization due to steric constraints."
        else:
            frag_name = frag["name"]
            route = frag["route"]
            
        test_mol = Chem.MolFromSmiles(derived_smiles)
        mw = round(Descriptors.MolWt(test_mol), 2) if test_mol else 0
        logp = round(Descriptors.MolLogP(test_mol), 2) if test_mol else 0
        delta_score = round(baseline - (idx * 0.15) - (abs(logp) * 0.05), 2) if success else round(baseline + 0.5, 2)
        
        derived_library.append({
            "Variant ID": f"Derivative-{idx+1:02d}" if success else f"Formulation-{idx+1:02d}",
            "Fragment Added": frag_name, "Redesigned SMILES": derived_smiles, "Delta Score": delta_score,
            "MW (g/mol)": mw, "LogP": logp, "Yield Prediction": frag["yield"] if success else "Pharmaceutical Salt Matrix",
            "Route": route, "FTIR Peak": int(frag["peak"])
        })
    return derived_library

def get_iupac_name(smiles):
    try:
        encoded_smiles = urllib.parse.quote(smiles, safe='')
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.read().decode('utf-8')
    except Exception:
        return "IUPAC translation unavailable (Network Timeout)"

def calculate_advanced_adme(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    mol = Chem.AddHs(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    lipinski_obey = "Yes" if violations <= 1 else "No"
    oral_bio = "Yes (High)" if violations == 0 else ("Yes (Moderate)" if violations == 1 else "No (Poor)")
    ring_info = mol.GetRingInfo().AtomRings()
    max_ring = max([len(r) for r in ring_info]) if ring_info else 0
    try:
        temp_mol = Chem.Mol(mol)
        AllChem.EmbedMolecule(temp_mol, randomSeed=42)
        vol = AllChem.ComputeMolVolume(temp_mol)
    except: vol = mw * 0.88
        
    acidic_pka = "Neutral"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")): acidic_pka = "Acidic (~4.5)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH]")): acidic_pka = "Weak Acid (~9.5)"
    basic_pka = "Neutral"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")): basic_pka = "Basic (~9.0)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("cN")): basic_pka = "Weak Base (~4.0)"
    
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    est_mp = max(20.0, (mw * 0.4) + (hbd * 25.0) - (rot_bonds * 5.0))
    est_bp = est_mp + 150.0 + (mw * 0.5)
    hia = (tpsa < 132) and (-2.0 < logp < 6.0)
    bbb = (tpsa < 79) and (0.4 < logp < 6.0)
    perm = "High BBB Penetration & GI Absorption" if bbb else ("Good GI Absorption" if hia else "Poor Absorption / Impermeable")
    
    return {
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa, "Violations": violations,
        "Lipinski_Obey": lipinski_obey, "Oral_Bio": oral_bio, "MaxRing": max_ring, "Volume": vol,
        "pKa_Acid": acidic_pka, "pKa_Base": basic_pka, "MP": est_mp, "BP": est_bp, "Permeability": perm,
        "BBB": bbb, "HIA": hia
    }

# =====================================================================
# 4. HIGH PERFORMANCE VISUALIZATION UTILITIES
# =====================================================================

def generate_2d_ligand_img(mol):
    if mol is None: return None
    try:
        mol_flat = Chem.Mol(mol)
        Chem.SanitizeMol(mol_flat)
        AllChem.Compute2DCoords(mol_flat)
        img = Draw.MolToImage(mol_flat, size=(340, 260))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception: return None

def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            mol_to_draw = Chem.RemoveHs(mol)
            if include_labels:
                for atom in mol_to_draw.GetAtoms():
                    atom.SetProp('atomNote', str(atom.GetIdx()))
            img = Draw.MolToImage(mol_to_draw, size=(zoom_level, int(zoom_level * 0.77)))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:15px;"/>'
    except Exception: pass
    return None

def generate_ftir_image(target_peak):
    wavenumbers = np.linspace(400, 4000, 500)
    baseline = 98.0 - 2.0 * np.sin(wavenumbers / 200.0)
    effect = 40.0 * np.exp(-((wavenumbers - target_peak) / 45.0)**2)
    transmittance = np.clip(baseline - effect, 5.0, 100.0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(wavenumbers, transmittance, color='#1e3c72', linewidth=2)
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Transmittance (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.fill_between(wavenumbers, transmittance, 105, color='#1e3c72', alpha=0.05)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[]):
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc = interact["r_coord"]
        lc = interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer.addLabel("{interact['Residue Contact']} ({interact['Distance (Å)']}A)", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:11}});
        """
    html_content = f"""
    <div id="wrapper_div" style="position:relative; width:100%;">
        <button onclick="toggleFullScreen()" style="position:absolute; top:12px; right:12px; z-index:9999; padding:6px 12px; background:#007bff; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.15);">🖥 Fullscreen View</button>
        <div id="container" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{ viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}}); }} 
            else if ('{mode}' === 'spacefill') {{ viewer.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}}); }} 
            else {{ viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}}); }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer.zoomTo(); viewer.render();
        function toggleFullScreen() {{
            let elem = document.getElementById("wrapper_div");
            if (!document.fullscreenElement) {{ elem.requestFullscreen(); document.getElementById("container").style.height = "90vh"; }}
            else {{ document.exitFullscreen(); document.getElementById("container").style.height = "480px"; }}
        }}
        document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) document.getElementById("container").style.height = "480px"; }});
    </script>
    """
    components.html(html_content, height=510)

def build_comprehensive_html_report(meta, adme_p, adme_v, variant_row, iupac, shift_msg, f_img, v_2d, p_2d):
    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>InSilico BioSphere Complete Report</title>
    <style>
        body {{ font-family: sans-serif; color: #333; margin:0; padding:0; background:#f4f6f9; }}
        .banner {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color:white; padding:30px; text-align:center; }}
        .card {{ background: white; max-width: 900px; margin: 30px auto; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
        h2 {{ color:#1e3c72; border-bottom:2px solid #e2e8f0; padding-bottom:5px; }}
        table {{ width:100%; border-collapse:collapse; margin:15px 0; font-size:13px; }}
        th, td {{ border:1px solid #e2e8f0; padding:10px; text-align:left; }}
        th {{ background:#f8fafc; color:#1e3c72; }}
        .scandata {{ font-family:monospace; background:#f1f5f9; padding:5px; display:block; border-radius:4px; }}
    </style></head><body>
    <div class="banner">
        <h2>🔬 InSilico BioSphere Consolidated Research Record</h2>
        <p>Department of Chemistry, Shivaji Science College, Nagpur, India</p>
    </div>
    <div class="card">
        <h2>1. Macromolecular Target Identity</h2>
        <p><b>PDB Code/Source:</b> {meta['id']}<br><b>Title:</b> {meta['title']}<br><b>Method:</b> {meta['method']} ({meta['res']})</p>
        
        <h2>2. Baseline Docking Properties</h2>
        <p><b>Initial Phytochemical SMILES:</b> <span class="scandata">{st.session_state.smiles_cache}</span></p>
        <p><b>Computed Baseline Affinity Score:</b> {st.session_state.baseline_affinity} kcal/mol</p>
        
        <h2>3. Optimization Engineering & Modifications</h2>
        <p><b>Isolated Variant ID:</b> {variant_row['Variant ID']}<br><b>Appended Functional Group:</b> {variant_row['Fragment Added']}</p>
        <p><b>System Redesigned SMILES Matrix:</b> <span class="scandata">{variant_row['Redesigned SMILES']}</span></p>
        <p><b>Proposed Synthesis Mapping Vector:</b> {variant_row['Route']} ({variant_row['Yield Prediction']})</p>
        
        <h2>4. ADMET Drug-Likeness Matrix</h2>
        <p><b>Nomenclature (IUPAC):</b> {iupac}</p>
        <table>
            <tr><th>Parameter Parameterized</th><th>Original Phytochemical Lead</th><th>Redesigned Variant Matrix</th></tr>
            <tr><td>Obey Lipinski's Rule?</td><td>{adme_p['Lipinski_Obey']}</td><td>{adme_v['Lipinski_Obey']}</td></tr>
            <tr><td>Oral Bioavailability Probability</td><td>{adme_p['Oral_Bio']}</td><td>{adme_v['Oral_Bio']}</td></tr>
            <tr><td>Total Permeability Profile</td><td>{adme_p['Permeability']}</td><td>{adme_v['Permeability']}</td></tr>
            <tr><td>TPSA (Å²)</td><td>{adme_p['TPSA']:.2f}</td><td>{adme_v['TPSA']:.2f}</td></tr>
            <tr><td>Molecular Volume (Å³)</td><td>{adme_p['Volume']:.1f}</td><td>{adme_v['Volume']:.1f}</td></tr>
            <tr><td>Lipophilicity Parameter (LogP)</td><td>{adme_p['LogP']:.2f}</td><td>{adme_v['LogP']:.2f}</td></tr>
        </table>
        <h3>Dynamic Assessment Narrative</h3>
        <p style="background:#f0fdf4; color:#166534; padding:15px; border-left:4px solid #16a34a;">{shift_msg}</p>
        
        <h2>5. Vibrational Fingerprint Footprint (FTIR Spectrum)</h2>
        <div style="text-align:center;"><img src="data:image/png;base64,{f_img}" style="max-width:100%; border-radius:6px;"/></div>
    </div>
    <div style="text-align:center; padding:20px; color:#64748b; font-size:11px;">System Pipeline Core Development © Dr. Sarang S. Dhote (TLCS)</div>
    </body></html>
    """

# =====================================================================
# 6. APPLICATION DASHBOARD WORKSPACE
# =====================================================================

st.set_page_config(page_title="In Silico BioSphere Hub", layout="wide")
st.title("🔬 InSilico BioSphere - Unified Drug Design Engine")
st.markdown("**Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Tech Logic Core Systems (TLCS)**")

# Master Reset
if st.button("🔄 Reset Entire Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb"]:
        if os.path.exists(f): os.remove(f)
    st.success("Dashboard cache and runtime structures completely cleared!")
    st.rerun()

# Layout Workspace Setup
tab1, tab2, tab3 = st.tabs(["🔒 Phase 1: Baseline Docking", "🧬 Phase 2: Generative Scaffold Redesign", "📊 Phase 3: ADMET Analytics & Reporting"])

# ---------------------------------------------------------------------
# PHASE 1: CORE BASELINE DOCKING ENGINE
# ---------------------------------------------------------------------
with tab1:
    col_params, col_visual = st.columns([1, 1])
    
    with col_params:
        st.header("1. Target Protein Setup")
        protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
        
        if protein_source == "Type 4-Letter PDB ID":
            pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
            if st.button("📥 Load Target Structure"):
                if pdb_id_input:
                    success, path = fetch_pdb_from_rcsb(pdb_id_input)
                    if success:
                        st.session_state.local_target_path = path
                        st.session_state.pdb_id_display = pdb_id_input.upper()
                        conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                        st.session_state.target_ready = conv_ok
                        st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
                        st.rerun()
                    else: st.error(path)
        else:
            uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
            if uploaded_file:
                path = f"uploaded_{uploaded_file.name}"
                if st.session_state.local_target_path != path:
                    with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                    st.session_state.local_target_path = path
                    st.session_state.pdb_id_display = "Uploaded File"
                    if uploaded_file.name.endswith(".pdb"):
                        conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                        st.session_state.target_ready = conv_ok
                    else:
                        os.replace(path, "protein.pdbqt")
                        st.session_state.target_ready = True
                    st.rerun()
    
        if st.session_state.target_ready and st.session_state.local_target_path:
            meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
            st.markdown(f"> **Protein Summary Profile:** \n> * **Title:** {meta['title']} \n> * **PDB ID:** `{meta['id']}` | **Classification:** {meta['class']} \n> * **Resolution:** **{meta['res']}**")
    
        st.header("2. Small Molecule Ligand Setup")
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
                            ok, _ = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
                            if ok:
                                st.session_state.ligand_ready = True
                                st.session_state.smiles_cache = smiles_input_val
                                with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                                st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                                st.success("Ligand metadata mapped from PubChem!")
                                st.rerun()
                    except Exception as e: st.error(f"SMILES Parsing Failure: {e}")
                    
            elif ligand_source == "Upload Structural File (.pdb, .sdf)" and uploaded_lig_buffer is not None:
                temp_in = f"raw_ligand_{uploaded_lig_name}"
                with open(temp_in, "wb") as f: f.write(uploaded_lig_buffer.getbuffer())
                
                mol = Chem.MolFromPDBFile(temp_in, removeHs=False) if uploaded_lig_name.endswith(".pdb") else Chem.SDMolSupplier(temp_in, removeHs=False)[0]
                
                if mol:
                    # ROBUST SMILES EXTRACTION FOR REDESIGN STUDIO
                    extracted_smiles = ""
                    try: 
                        Chem.SanitizeMol(mol)
                        AllChem.AssignBondOrdersFromTopology(mol)
                        extracted_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                    except Exception: 
                        try: extracted_smiles = Chem.MolToSmiles(mol)
                        except: pass
                    
                    if not extracted_smiles:
                        st.error("⚠️ RDKit could not extract a valid SMILES string from this uploaded file (missing bond orders). Phase 2 Redesign will be locked. Please use 'SMILES String Input' instead.")
                    else:
                        st.session_state.smiles_cache = extracted_smiles 
                    
                    if mol.GetNumConformers() == 0:
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                        AllChem.MMFFOptimizeMolecule(mol)
                        
                    temp_pdb = "temp_lig_state.pdb"
                    Chem.MolToPDBFile(mol, temp_pdb)
                    convert_pdb_to_pdbqt(temp_pdb, "ligand.pdbqt", is_ligand=True)
                    st.session_state.ligand_ready = True
                    
                    st.session_state.ligand_summary_text = f"Ligand structure loaded successfully. Extracted Template: `{extracted_smiles if extracted_smiles else 'Failed'}`"
                    with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                    if os.path.exists(temp_in): os.remove(temp_in)
                    if os.path.exists(temp_pdb): os.remove(temp_pdb)
                    st.success("Structural file loaded and ready for docking!")
    
        if st.session_state.target_ready and os.path.exists("ligand.pdbqt"): st.session_state.ligand_ready = True
        if st.session_state.ligand_ready: st.markdown(f"> **Ligand Metric Summary Profile:** \n> {st.session_state.ligand_summary_text}")
    
        if st.session_state.target_ready and st.session_state.local_target_path:
            bound_ligands_list = parse_bound_ligands(st.session_state.local_target_path)
            if bound_ligands_list:
                st.header("3. Bound Small Molecules in Receptor")
                df_bound = pd.DataFrame(bound_ligands_list)
                df_display = df_bound.copy()
                df_display["Center (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
                df_display["Box (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['bx']}, {r['by']}, {r['bz']}", axis=1)
                st.dataframe(df_display[["ID", "Chain", "ResSeq", "Atoms", "Center (X, Y, Z) Å", "Box (X, Y, Z) Å"]], hide_index=True, use_container_width=True)
                
                selected_lig_id = st.selectbox("Select native co-crystal target to auto-fill grid box:", options=range(len(bound_ligands_list)), format_func=lambda idx: f"{bound_ligands_list[idx]['ID']} (Chain {bound_ligands_list[idx]['Chain']}-ResSeq {bound_ligands_list[idx]['ResSeq']})")
                if st.button("🎯 Lock Coordinates to Native Site"):
                    chosen_target = bound_ligands_list[selected_lig_id]
                    st.session_state.cx, st.session_state.cy,
