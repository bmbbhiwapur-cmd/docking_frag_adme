"""
==========================================================================
InSilico BioSphere - Unified Docking + Redesign + Comparative Docking + ADME
==========================================================================
A single Streamlit application that walks the user through:
    1. Loading a protein (PDB ID or upload) and an original ligand (SMILES / file).
    2. Running molecular docking of the original ligand vs the receptor.
    3. Generating a redesigned ligand library (fragment substitution).
    4. Letting the user PICK a redesigned variant and re-dock it against the
       same protein with the same grid box.
    5. Showing a side-by-side comparative report of binding affinities
       (Original vs. Redesigned) AND a full ADME comparison.
    6. Exporting a comprehensive HTML report.

Developed by: Mr. Sarang S. Dhote, Assistant Professor,
Department of Chemistry, Shivaji Science College, Nagpur, India.
Contact: sarangresearch@gmail.com
==========================================================================
"""

import time
import os
import io
import re
import json
import base64
import urllib.request
import urllib.parse
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors


# =====================================================================
# 0. PAGE CONFIG + GLOBAL STATE
# =====================================================================
st.set_page_config(page_title="InSilico BioSphere - Unified Studio", layout="wide")

st.title("🧬 InSilico BioSphere - Unified Docking + Redesign Studio")
st.markdown(
    "**Workflow:** Original Docking → Ligand Redesign → Pick a Variant → Comparative Docking → ADME Analysis  \n"
    "Developed by **Mr. Sarang S. Dhote**, Assistant Professor, Department of Chemistry, "
    "Shivaji Science College, Nagpur, India | sarangresearch@gmail.com"
)

DEFAULT_STATE = {
    # Protein
    "target_ready": False,
    "local_target_path": None,
    "pdb_id_display": "Custom",
    "protein_metadata": {},
    # Original ligand
    "ligand_ready": False,
    "original_smiles": "",
    "original_ligand_summary": "",
    "original_ligand_block": None,
    # Grid box
    "cx": 0.0, "cy": 0.0, "cz": 0.0,
    "sx": 20,  "sy": 20,  "sz": 20,
    "exhaustiveness": 8,
    # Original docking results
    "original_docking_raw": None,
    "original_best_affinity": None,
    "original_df": None,
    "original_poses_file": None,
    # Redesign
    "redesign_library": None,
    "selected_variant_id": None,
    # Redesigned docking results
    "redesign_docking_raw": None,
    "redesign_best_affinity": None,
    "redesign_df": None,
    "redesign_poses_file": None,
    "redesign_smiles_used": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =====================================================================
# 1. VINA BINARY BOOTSTRAP
# =====================================================================
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing AutoDock Vina binary..."):
            try:
                url = ("https://github.com/ccsb-scripps/AutoDock-Vina/releases/"
                       "download/v1.2.5/vina_1.2.5_linux_x86_64")
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Vina binary mounted successfully!")
            except Exception as e:
                st.error(f"Failed to bootstrap Vina binary: {e}")

ensure_linux_vina_exists()


# =====================================================================
# 2. EXTERNAL DATA UTILITIES (PubChem, RCSB, NCI Cactus)
# =====================================================================
def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound", "mw": "N/A", "formula": "N/A"}
    try:
        escaped = urllib.parse.quote(smiles_string)
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
               f"{escaped}/property/Title,MolecularWeight,MolecularFormula/JSON")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
            metadata["name"] = props.get("Title", "Target Chemical Derivative")
            metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
            metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception:
        pass
    return metadata


def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."


def get_iupac_name(smiles):
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/iupac_name"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return "IUPAC translation unavailable (network/timeout)"


# =====================================================================
# 3. PDB PARSING (metadata + bound co-crystal ligand finder)
# =====================================================================
def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "title": "Uploaded Protein Structure", "id": pdb_id.upper(),
        "class": "Unknown", "organism": "Unknown",
        "system": "Unknown", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path):
        return meta
    with open(file_path, "r") as f:
        title_parts = []
        for line in f:
            if line.startswith("TITLE"):
                title_parts.append(line[10:80].strip())
            elif line.startswith("HEADER"):
                meta["class"] = line[10:50].strip().title()
            elif "ORGANISM_SCIENTIFIC" in line:
                meta["organism"] = line.split(":")[-1].replace(";", "").strip()
            elif "EXPRESSION_SYSTEM" in line:
                meta["system"] = line.split(":")[-1].replace(";", "").strip()
            elif line.startswith("EXPDTA"):
                meta["method"] = line[10:80].strip()
            elif "RESOLUTION." in line and "ANGSTROMS." in line:
                m = re.search(r"(\d+\.\d+)", line)
                if m:
                    meta["res"] = f"{m.group(1)} Å"
    if title_parts:
        meta["title"] = " ".join(title_parts).title()
    return meta


def parse_bound_ligands(file_path):
    ligands = {}
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip() or "A"
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    continue
                if res_name in ["HOH", "WAT", "DOD"]:
                    continue
                key = f"{res_name}-{chain_id}-{res_seq}"
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue
                if key not in ligands:
                    ligands[key] = {"res": res_name, "chain": chain_id,
                                    "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))

    out = []
    for _, info in ligands.items():
        pts = info["coords"]
        n = len(pts)
        if n < 4:
            continue
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        cz = sum(p[2] for p in pts) / n
        bx = max(p[0] for p in pts) - min(p[0] for p in pts) + 10
        by = max(p[1] for p in pts) - min(p[1] for p in pts) + 10
        bz = max(p[2] for p in pts) - min(p[2] for p in pts) + 10
        out.append({
            "ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"],
            "Atoms": n,
            "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2),
            "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)
        })
    return out


# =====================================================================
# 4. PDBQT CONVERSION (protein + ligand)
# =====================================================================
AUTODOCK_TYPE_MAP = {
    "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N",
    "NA": "NA", "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA",
    "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
    "ZN": "Zn", "MG": "Mg"
}


def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    torsions = 0
    if is_ligand:
        try:
            m = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if m:
                torsions = AllChem.CalcNumRotatableBonds(m)
        except Exception:
            torsions = 4
    try:
        with open(input_pdb, "r") as pdb, open(output_pdbqt, "w") as pq:
            if is_ligand:
                pq.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    rec = line[:6].strip()
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip() or "A"
                    try: res_seq = int(line[22:26].strip())
                    except ValueError: res_seq = 1
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except ValueError:
                        continue
                    element = line[76:78].strip()
                    if not element:
                        element = "".join(c for c in atom_name if c.isalpha())[0]
                    element = "".join(c for c in element if c.isalpha()).upper()
                    vt = AUTODOCK_TYPE_MAP.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper():
                        vt = "A"
                    pq.write(f"{rec:<6}{atom_id:>5} {atom_name:<4} "
                             f"{res_name:>3} {chain_id}{res_seq:>4}    "
                             f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}"
                             f"{0.00:>6.2f}    +0.000 {vt:<2}\n")
            if is_ligand:
                pq.write("ENDROOT\n")
                pq.write(f"TORSDOF {torsions}\n")
            else:
                pq.write("ENDMDL\n")
        return True, output_pdbqt
    except Exception as e:
        return False, str(e)


def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        tmp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, tmp_pdb)
        convert_pdb_to_pdbqt(tmp_pdb, output_filename, is_ligand=True)
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)
        return True, output_filename
    except Exception as e:
        return False, str(e)


# =====================================================================
# 5. DOCKING OUTPUT PARSERS + INTERACTION ENGINE
# =====================================================================
def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path):
        return poses
    current_mode, lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None:
                    poses[current_mode] = "".join(lines)
                current_mode = None
            else:
                lines.append(line)
    return poses


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
                atoms.append({
                    "coord": np.array([x, y, z]),
                    "element": element,
                    "res": f"{res_name}{res_seq}"
                })
            except ValueError:
                continue
    return atoms


def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    out = []
    if not os.path.exists(receptor_file):
        return out
    with open(receptor_file, "r") as f:
        receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    seen = set()
    for la in ligand_atoms:
        for ra in receptor_atoms:
            d = np.linalg.norm(la["coord"] - ra["coord"])
            if d < 3.8:
                res_id = ra["res"]
                if res_id in seen:
                    continue
                if la["element"] in ["N","O","F","S"] and ra["element"] in ["N","O","F","S"]:
                    bond = "Hydrogen Bond"
                elif ("A" in ra["element"]) or (la["element"]=="C" and ra["element"]=="C"
                       and any(a in ra["res"] for a in ["PHE","TYR","TRP"])):
                    bond = "pi-Stacking / Hydrophobic"
                else:
                    bond = "van der Waals Contact"
                seen.add(res_id)
                out.append({
                    "Residue Contact": res_id,
                    "Interaction Type": bond,
                    "Distance (Å)": round(d, 2),
                    "r_coord": ra["coord"].tolist(),
                    "l_coord": la["coord"].tolist()
                })
    return out


def parse_vina_output(stdout_text, poses_file):
    rows = []
    pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    poses_dict = split_docking_poses(poses_file)
    for line in stdout_text.split("\n"):
        m = pattern.match(line)
        if m:
            mi = int(m.group(1))
            res_string, bond_types = "N/A", "N/A"
            if mi in poses_dict:
                ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mi])
                if ints:
                    res_string = ", ".join(sorted({i["Residue Contact"] for i in ints}))
                    bond_types = ", ".join(sorted({i["Interaction Type"] for i in ints}))
            rows.append({
                "Binding Mode": mi,
                "Affinity (kcal/mol)": float(m.group(2)),
                "RMSD l.b.": float(m.group(3)),
                "RMSD u.b.": float(m.group(4)),
                "Interacting Residues": res_string,
                "Contact Bond Types": bond_types
            })
    return pd.DataFrame(rows)


# =====================================================================
# 6. DOCKING EXECUTION ENGINE (shared by original + redesigned ligands)
# =====================================================================
def run_vina_docking(ligand_pdbqt, out_poses_file, grid, exhaustiveness=8):
    """Runs AutoDock Vina with a streaming progress bar. Returns (stdout, return_code)."""
    cmd = [
        "./vina",
        "--receptor", "protein.pdbqt",
        "--ligand", ligand_pdbqt,
        "--center_x", str(grid["cx"]),
        "--center_y", str(grid["cy"]),
        "--center_z", str(grid["cz"]),
        "--size_x", str(grid["sx"]),
        "--size_y", str(grid["sy"]),
        "--size_z", str(grid["sz"]),
        "--exhaustiveness", str(exhaustiveness),
        "--out", out_poses_file
    ]
    progress_bar = st.progress(0, text="Initializing docking engine...")
    status = st.empty()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out_log, cnt, line = [], 0, ""
        while True:
            ch = proc.stdout.read(1).decode("utf-8", errors="ignore")
            if not ch:
                break
            out_log.append(ch)
            if ch == "*":
                cnt += 1
                pct = min(100, int((cnt/50)*100))
                progress_bar.progress(pct, text=f"Exploring binding modes... {pct}%")
            elif ch == "\n":
                if "Performing search" in line:
                    status.info("Running BFGS optimization & spatial search...")
                elif "Refining" in line:
                    status.info("Refining top poses...")
                line = ""
            else:
                line += ch
        proc.wait()
        progress_bar.progress(100, text="Docking complete!")
        time.sleep(0.4)
        progress_bar.empty()
        status.empty()
        return "".join(out_log), proc.returncode
    except Exception as e:
        progress_bar.empty()
        status.error(f"Engine failed: {e}")
        return "", -1


def best_affinity_from_df(df):
    if df is None or df.empty:
        return None
    return float(df["Affinity (kcal/mol)"].min())


# =====================================================================
# 7. REDESIGN ENGINE (fragment substitution)
# =====================================================================
def find_valid_cleavage_sites(smiles_str):
    sites = []
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            for atom in mol.GetAtoms():
                idx, sym, deg = atom.GetIdx(), atom.GetSymbol(), atom.GetDegree()
                hs = atom.GetTotalNumHs()
                if deg == 1 and sym != "C":
                    sites.append({"index": idx, "label": f"Atom #{idx} (Terminal {sym})"})
                elif sym == "C" and hs > 0:
                    sites.append({"index": idx, "label": f"Atom #{idx} ({sym} with available H)"})
                elif sym in ["N","O","S"] and hs > 0:
                    sites.append({"index": idx, "label": f"Atom #{idx} (Core {sym} with available H)"})
        sites.sort(key=lambda x: (0 if "Terminal" in x["label"] else 1, x["index"]))
    except Exception:
        pass
    return sites


def get_dynamic_fragments(parent_smiles):
    mol = Chem.MolFromSmiles(parent_smiles)
    if not mol:
        return "Standard Organic Scaffold", []
    flavone_smarts = Chem.MolFromSmarts("c1cc(O)cc2c1c(=O)cc(c2)c3ccccc3")
    phenol_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c[OH]")))
    alkaloid_smarts = Chem.MolFromSmarts("[#7;R]")
    aliphatic_c = [a for a in mol.GetAtoms() if a.GetSymbol()=="C" and not a.GetIsAromatic()]
    all_c = [a for a in mol.GetAtoms() if a.GetSymbol()=="C"]
    ratio = len(aliphatic_c)/len(all_c) if all_c else 0

    if mol.HasSubstructMatch(flavone_smarts) or phenol_count >= 2:
        cls = "Polyphenolic Flavonoid Core"
        frags = [
            {"name":"Glucosylation (-C6H11O5)","smiles":"OC1C(O)C(O)C(O)C(CO)O1","peak":3350,
             "yield":"Moderate Yield (58%)","route":"Enzymatic glycosylation via Phase II transferase."},
            {"name":"Prenylation (-CH2CH=C(CH3)2)","smiles":"CC(C)=CC","peak":1660,
             "yield":"Good Yield (72%)","route":"Late-stage electrophilic C-alkylation."},
            {"name":"O-Methylation (-OCH3)","smiles":"OC","peak":1250,
             "yield":"Excellent Yield (91%)","route":"Selective etherification with Dimethyl Sulfate."},
            {"name":"Acetylation (-OCOCH3)","smiles":"OC(=O)C","peak":1735,
             "yield":"Good Yield (84%)","route":"Esterification using Acetic Anhydride."}
        ]
    elif mol.HasSubstructMatch(alkaloid_smarts):
        cls = "Alkaloidal Nitrogen Heterocycle"
        frags = [
            {"name":"N-Alkylation (-CH2CH3)","smiles":"CC","peak":2960,
             "yield":"Good Yield (80%)","route":"Nucleophilic substitution with Ethyl Bromide."},
            {"name":"Quaternization (-CH3+)","smiles":"C","peak":2850,
             "yield":"Excellent Yield (94%)","route":"Methylation using Methyl Iodide."},
            {"name":"Amidation (-COCH3)","smiles":"C(=O)C","peak":1665,
             "yield":"Good Yield (78%)","route":"Amide condensation with Acetyl Chloride."},
            {"name":"N-Oxidation (=O)","smiles":"[O-]","peak":950,
             "yield":"Moderate Yield (65%)","route":"Controlled oxidation via mCPBA."}
        ]
    elif ratio > 0.65:
        cls = "Aliphatic Terpenoid Scaffold"
        frags = [
            {"name":"Epoxidation (=O)","smiles":"O","peak":1250,
             "yield":"Moderate Yield (60%)","route":"Prilezhaev reaction with mCPBA."},
            {"name":"Hydroxylation (-OH)","smiles":"O","peak":3400,
             "yield":"Poor Yield (42%)","route":"Allylic C-H functionalization via SeO2."},
            {"name":"Ozonolysis Fragmentation","smiles":"O=C","peak":1710,
             "yield":"Good Yield (70%)","route":"Oxidative cleavage of double bonds."},
            {"name":"Esterification (-COOCH3)","smiles":"C(=O)OC","peak":1740,
             "yield":"Good Yield (86%)","route":"Fischer esterification."}
        ]
    else:
        cls = "Standard Organic Lead Profile"
        frags = [
            {"name":"Methylation (-CH3)","smiles":"C","peak":2925,
             "yield":"Good Yield (85%)","route":"Standard alkylation with Methyl Iodide."},
            {"name":"Hydroxylation (-OH)","smiles":"O","peak":3450,
             "yield":"Moderate Yield (62%)","route":"Direct C-H oxidation with copper coordination."},
            {"name":"Amination (-NH2)","smiles":"N","peak":3320,
             "yield":"Good Yield (74%)","route":"Nucleophilic amination."},
            {"name":"Fluorination (-F)","smiles":"F","peak":1150,
             "yield":"Poor Yield (38%)","route":"Late-stage electrophilic fluorination with Selectfluor."}
        ]
    return cls, frags


def run_cleaving_engine(parent_smiles, target_atom_idx, mechanism_mode,
                        original_best_affinity=None):
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if not parent_mol:
        return []
    _, fragments = get_dynamic_fragments(parent_smiles)
    library = []
    for idx, frag in enumerate(fragments):
        success = False
        derived_smiles = ""

        if mechanism_mode == "True Covalent Substitution (Cleavage & Attachment)":
            try:
                rw = Chem.RWMol(parent_mol)
                t = rw.GetAtomWithIdx(int(target_atom_idx))
                is_terminal = (t.GetDegree() == 1 and t.GetSymbol() != "C")
                if is_terminal:
                    t.SetAtomicNum(0)
                    t.SetIsotope(999)
                else:
                    dummy = Chem.Atom(0)
                    dummy.SetIsotope(999)
                    nd = rw.AddAtom(dummy)
                    rw.AddBond(int(target_atom_idx), nd, Chem.BondType.SINGLE)
                tagged = rw.GetMol()
                Chem.SanitizeMol(tagged)
                pat = Chem.MolFromSmarts("[999*]")
                frag_mol = Chem.MolFromSmiles(frag["smiles"])
                replaced = AllChem.ReplaceSubstructs(tagged, pat, frag_mol, replaceAll=True)
                if replaced:
                    final = replaced[0]
                    Chem.SanitizeMol(final)
                    derived_smiles = Chem.MolToSmiles(final)
                    if Chem.MolFromSmiles(derived_smiles):
                        success = True
            except Exception:
                success = False

        if not success:
            derived_smiles = f"{parent_smiles}.{frag['smiles']}"
            frag_name = frag["name"] + " (Co-Crystal Fallback)" if "Co-Crystal" not in mechanism_mode else frag["name"] + " (Co-Crystal)"
            route = "Co-crystallization (steric block on covalent bond)."
        else:
            frag_name = frag["name"]
            route = frag["route"]

        test_mol = Chem.MolFromSmiles(derived_smiles)
        mw   = round(Descriptors.MolWt(test_mol), 2)    if test_mol else 0
        logp = round(Descriptors.MolLogP(test_mol), 2)  if test_mol else 0

        # Predicted delta score - if we know original docking affinity, predict ~ improvement
        if original_best_affinity is not None:
            base = original_best_affinity - 0.4 - (idx * 0.15) - (abs(logp)*0.04)
        else:
            base = -6.2 - (idx * 0.15) - (abs(logp)*0.05)
        predicted_score = round(base if success else base + 0.7, 2)

        library.append({
            "Variant ID": f"Derivative-{idx+1:02d}" if success else f"Formulation-{idx+1:02d}",
            "Fragment Added": frag_name,
            "Redesigned SMILES": derived_smiles,
            "Predicted Affinity (kcal/mol)": predicted_score,
            "MW (g/mol)": mw, "LogP": logp,
            "Yield Prediction": frag["yield"] if success else "Pharmaceutical Salt Matrix",
            "Route": route,
            "FTIR Peak": int(frag["peak"])
        })
    return library


# =====================================================================
# 8. ADME ENGINE
# =====================================================================
def calculate_advanced_adme(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mol = Chem.AddHs(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    violations = sum([mw>500, logp>5, hbd>5, hba>10])
    lipinski_obey = "Yes" if violations <= 1 else "No"
    oral_bio = ("Yes (High Probability)" if violations==0
                else "Yes (Moderate Probability)" if violations==1
                else "No (Poor Bioavailability)")
    rings = mol.GetRingInfo().AtomRings()
    max_ring = max([len(r) for r in rings]) if rings else 0
    try:
        tm = Chem.Mol(mol)
        AllChem.EmbedMolecule(tm, randomSeed=42)
        vol = AllChem.ComputeMolVolume(tm)
    except Exception:
        vol = mw * 0.88
    acid = "Neutral (None)"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")): acid = "Acidic (~4.5)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH]")):   acid = "Weak Acid (~9.5)"
    base = "Neutral (None)"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")): base = "Basic (~9.0)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("cN")):                base = "Weak Base (~4.0)"
    rot = Descriptors.NumRotatableBonds(mol)
    mp = max(20.0, (mw*0.4) + (hbd*25.0) - (rot*5.0))
    bp = mp + 150.0 + (mw*0.5)
    hia = (tpsa < 132) and (-2.0 < logp < 6.0)
    bbb = (tpsa < 79)  and ( 0.4 < logp < 6.0)
    if bbb:    perm = "High BBB Penetration & GI Absorption"
    elif hia:  perm = "Good GI Absorption (No BBB Penetration)"
    else:      perm = "Poor Absorption / Impermeable"
    return {"MW":mw, "LogP":logp, "HBD":hbd, "HBA":hba, "TPSA":tpsa,
            "Violations":violations, "Lipinski_Obey":lipinski_obey, "Oral_Bio":oral_bio,
            "MaxRing":max_ring, "Volume":vol, "pKa_Acid":acid, "pKa_Base":base,
            "MP":mp, "BP":bp, "Permeability":perm, "BBB":bbb, "HIA":hia}


# =====================================================================
# 9. VISUALIZATION (2D + 3D)
# =====================================================================
def smiles_to_2d_img_b64(smiles, size=340, include_labels=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        m2 = Chem.RemoveHs(mol)
        if include_labels:
            for a in m2.GetAtoms():
                a.SetProp("atomNote", str(a.GetIdx()))
        img = Draw.MolToImage(m2, size=(size, int(size*0.77)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def render_3d_complex(receptor_data, ligand_data, mode="cartoon",
                      show_surface=False, interactions_list=None, height=480):
    interactions_list = interactions_list or []
    surface_js = ("viewer.addSurface($3Dmol.SurfaceType.VDW, "
                  "{opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});"
                  if show_surface else "")
    int_js = ""
    for it in interactions_list:
        rc, lc = it["r_coord"], it["l_coord"]
        color = "yellow" if "Hydrogen" in it["Interaction Type"] else "cyan"
        int_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}},
                            end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}},
                            radius:0.07, color:'{color}', dashed:true}});
        viewer.addLabel("{it['Residue Contact']} ({it['Distance (Å)']}A)",
            {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}},
              backgroundColor:'white', fontColor:'black',
              backgroundOpacity:0.8, fontSize:11}});
        """
    html = f"""
    <div style="position:relative; width:100%;">
      <div id="container" style="height:{height}px; width:100%; position:relative;
           border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
      let viewer = $3Dmol.createViewer(document.getElementById('container'),
          {{backgroundColor:'#ffffff'}});
      if (`{receptor_data}`.trim().length > 0) {{
        viewer.addModel(`{receptor_data}`, 'pdb');
        if ('{mode}' === 'cartoon') {{
          viewer.setStyle({{model:0}},
            {{cartoon:{{colorscheme:'chain', style:'oval', thickness:0.6}}}});
        }} else if ('{mode}' === 'spacefill') {{
          viewer.setStyle({{model:0}}, {{sphere:{{colorscheme:'chain', radius:1.1}}}});
        }} else {{
          viewer.setStyle({{model:0}}, {{stick:{{colorscheme:'chain', radius:0.25}}}});
        }}
      }}
      {surface_js}
      if (`{ligand_data}`.trim().length > 0) {{
        viewer.addModel(`{ligand_data}`, 'pdb');
        viewer.setStyle({{model:1}}, {{stick:{{colorscheme:'greenCarbon', radius:0.28}}}});
      }}
      {int_js}
      viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html, height=height+30)


def generate_ftir_image_html(target_peak):
    wn = np.linspace(400, 4000, 500)
    baseline = 98.0 - 2.0*np.sin(wn/200.0)
    effect = 40.0*np.exp(-((wn-target_peak)/45.0)**2)
    trans = np.clip(baseline-effect, 5.0, 100.0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(wn, trans, color="#1e3c72", linewidth=2)
    ax.set_xlim(4000, 400); ax.set_ylim(0, 105)
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Transmittance (%)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.fill_between(wn, trans, 105, color="#1e3c72", alpha=0.05)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# =====================================================================
# 10. HTML REPORT (with comparative docking section)
# =====================================================================
def generate_comparative_html_report(
    protein_id, protein_meta, original_smiles, original_best_aff, original_df,
    redesigned_variant_row, redesigned_smiles, redesigned_best_aff, redesigned_df,
    iupac_name, adme_parent, adme_variant, parent_img_b64, variant_img_b64,
    ftir_img_b64, shift_text
):
    def b64img(b64, size=320):
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:{size}px; border-radius:6px; '
                f'border:1px solid #e2e8f0;"/>' if b64 else "[image unavailable]")

    delta = redesigned_best_aff - original_best_aff if (
        original_best_aff is not None and redesigned_best_aff is not None) else None
    delta_text = f"{delta:+.2f} kcal/mol" if delta is not None else "N/A"
    verdict = ("✅ Redesigned ligand binds MORE strongly than the original."
               if delta is not None and delta < -0.1
               else "⚖️ Comparable binding affinities."
               if delta is not None and abs(delta) <= 0.1
               else "❌ Redesigned ligand binds LESS strongly than the original."
               if delta is not None else "")

    adme_rows_html = ""
    if adme_parent and adme_variant:
        for label, k, fmt in [
            ("Obey Lipinski's Rule?", "Lipinski_Obey", "{}"),
            ("Oral Bioavailability", "Oral_Bio", "{}"),
            ("Permeability Profile", "Permeability", "{}"),
            ("TPSA (Å²)", "TPSA", "{:.2f}"),
            ("Molecular Volume (Å³)", "Volume", "{:.1f}"),
            ("Max Ring Size", "MaxRing", "{}"),
            ("pKa (Acidic)", "pKa_Acid", "{}"),
            ("pKa (Basic)", "pKa_Base", "{}"),
            ("Est. Melting Point (°C)", "MP", "{:.1f}"),
            ("Est. Boiling Point (°C)", "BP", "{:.1f}"),
            ("LogP", "LogP", "{:.2f}")]:
            adme_rows_html += (
                f"<tr><td>{label}</td>"
                f"<td>{fmt.format(adme_parent[k])}</td>"
                f"<td>{fmt.format(adme_variant[k])}</td></tr>")

    original_df_html = (original_df.to_html(index=False, classes="table")
                        if original_df is not None else "<p>No data</p>")
    redesign_df_html = (redesigned_df.to_html(index=False, classes="table")
                        if redesigned_df is not None else "<p>No data</p>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>InSilico BioSphere - Unified Report</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;color:#333;line-height:1.6;
     margin:0;padding:0;background:#f9f9fb;}}
.header-banner{{background:linear-gradient(135deg,#1e3c72,#2a5298);color:#fff;
                padding:25px;border-bottom:5px solid #00c6ff;text-align:center;}}
.header-banner h1{{margin:0;font-size:26px;letter-spacing:1px;}}
.container{{max-width:1050px;margin:30px auto;background:#fff;padding:40px;
            border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.05);}}
h2{{color:#1e3c72;border-bottom:2px solid #eef2f7;padding-bottom:8px;
    margin-top:35px;font-size:20px;}}
h3{{color:#2a5298;font-size:16px;margin-top:18px;}}
.meta-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;
            margin-bottom:18px;background:#f4f7f6;padding:18px;border-radius:8px;}}
.meta-item strong{{color:#1e3c72;}}
.table-wrapper{{overflow-x:auto;margin:20px 0;border:1px solid #e2e8f0;
                border-radius:6px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th,td{{border:1px solid #e2e8f0;padding:9px;text-align:left;}}
th{{background:#f8fafc;color:#1e3c72;font-weight:600;}}
.compare-card{{display:grid;grid-template-columns:1fr 1fr;gap:25px;margin:25px 0;}}
.compare-box{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
              padding:20px;text-align:center;}}
.compare-box .score{{font-size:34px;font-weight:800;color:#1e3c72;}}
.compare-box .label{{font-size:12px;text-transform:uppercase;letter-spacing:1px;
                     color:#555;}}
.verdict{{background:#ecfdf5;border-left:5px solid #10b981;padding:20px;
          border-radius:6px;margin:18px 0;color:#065f46;font-size:14.5px;}}
.summary-card{{background:#ecfdf5;border-left:5px solid #10b981;padding:18px;
               border-radius:6px;margin:20px 0;color:#065f46;font-size:14px;}}
.structure-box{{display:flex;gap:20px;margin:18px 0;background:#fafafa;
                padding:18px;border-radius:8px;border:1px solid #eef2f7;
                align-items:center;}}
.scandata{{font-family:monospace;background:#f1f5f9;padding:3px 6px;
           border-radius:4px;font-size:13px;word-break:break-all;}}
footer{{text-align:center;padding:18px;font-size:12px;color:#64748b;
        border-top:1px solid #e2e8f0;margin-top:8px;}}
</style></head>
<body>
<div class="header-banner">
  <h1>🧬 InSilico BioSphere — Unified Docking + Redesign + ADME Report</h1>
  <p>Original vs. Redesigned Ligand Comparative Study</p>
  <p style="font-size:12px;opacity:0.85;">Developed by Mr. Sarang S. Dhote,
     Shivaji Science College, Nagpur, India</p>
</div>

<div class="container">

  <h2>1. Target Receptor Profile</h2>
  <div class="meta-grid">
    <div class="meta-item"><strong>PDB / Source:</strong> {protein_id}</div>
    <div class="meta-item"><strong>Classification:</strong> {protein_meta.get('class','N/A')}</div>
    <div class="meta-item"><strong>Organism:</strong> {protein_meta.get('organism','N/A')}</div>
    <div class="meta-item"><strong>Expression:</strong> {protein_meta.get('system','N/A')}</div>
    <div class="meta-item"><strong>Method:</strong> {protein_meta.get('method','N/A')}</div>
    <div class="meta-item"><strong>Resolution:</strong> {protein_meta.get('res','N/A')}</div>
  </div>

  <h2>2. Ligand Structures</h2>
  <div class="structure-box">
    <div>
      <h3>Original Lead</h3>
      {b64img(parent_img_b64)}
      <div class="scandata" style="margin-top:8px;">{original_smiles}</div>
    </div>
    <div>
      <h3>Redesigned Variant — {redesigned_variant_row['Variant ID']}</h3>
      {b64img(variant_img_b64)}
      <div class="scandata" style="margin-top:8px;">{redesigned_smiles}</div>
    </div>
  </div>
  <p><strong>Appended functional group:</strong> {redesigned_variant_row['Fragment Added']}<br>
  <strong>Synthetic route:</strong> {redesigned_variant_row['Route']}<br>
  <strong>Yield class:</strong> {redesigned_variant_row['Yield Prediction']}</p>

  <h2>3. Comparative Docking Results (AutoDock Vina)</h2>
  <div class="compare-card">
    <div class="compare-box">
      <div class="label">Original Ligand — Best Affinity</div>
      <div class="score">{f'{original_best_aff:.2f}' if original_best_aff is not None else 'N/A'}</div>
      <div class="label">kcal/mol</div>
    </div>
    <div class="compare-box" style="background:#f0f7f4;border-color:#2e7d32;">
      <div class="label">Redesigned Ligand — Best Affinity</div>
      <div class="score" style="color:#1b5e20;">{f'{redesigned_best_aff:.2f}' if redesigned_best_aff is not None else 'N/A'}</div>
      <div class="label">kcal/mol</div>
    </div>
  </div>
  <div class="verdict">
    <strong>Δ Binding Affinity (Redesign − Original):</strong> {delta_text}<br>
    {verdict}
    <em>(More negative = stronger predicted binding.)</em>
  </div>

  <h3>Original Ligand — All Binding Modes</h3>
  <div class="table-wrapper">{original_df_html}</div>

  <h3>Redesigned Ligand — All Binding Modes</h3>
  <div class="table-wrapper">{redesign_df_html}</div>

  <h2>4. IUPAC Name (Redesigned)</h2>
  <div class="scandata" style="background:#e0f2fe;color:#0369a1;padding:10px;
       border-left:4px solid #0284c7;">{iupac_name}</div>

  <h2>5. ADMET Comparative Matrix</h2>
  <div class="table-wrapper">
    <table>
      <tr><th>Parameter</th><th>Original Lead</th><th>Redesigned Variant</th></tr>
      {adme_rows_html}
    </table>
  </div>
  <div class="summary-card">{shift_text}</div>

  <h2>6. Predicted Vibrational Signature (FTIR) — Redesigned</h2>
  <div style="text-align:center;">{b64img(ftir_img_b64, size=600)}</div>

</div>
<footer>InSilico BioSphere Unified Report | copyright@sarang dhote</footer>
</body></html>
"""


# =====================================================================
# 11. RESET BUTTON
# =====================================================================
if st.button("🔄 Reset Entire Environment", type="secondary", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    for f in ["protein.pdbqt","ligand.pdbqt","docking_original.pdbqt",
              "docking_redesigned.pdbqt","redesigned_ligand.pdbqt",
              "temp_lig_state.pdb"]:
        if os.path.exists(f):
            os.remove(f)
    st.success("Environment cleared.")
    st.rerun()


# =====================================================================
# 12. SECTION 1 — PROTEIN + ORIGINAL LIGAND SETUP
# =====================================================================
st.write("---")
st.header("STEP 1 — Setup: Protein + Original Ligand + Grid Box")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🧬 Target Protein")
    p_mode = st.radio("Protein input:", ["PDB ID (4-letter)", "Upload file (.pdb / .pdbqt)"],
                      horizontal=True, key="prot_mode")
    if p_mode == "PDB ID (4-letter)":
        pdb_in = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
        if st.button("📥 Load Protein", key="btn_load_prot"):
            ok, path = fetch_pdb_from_rcsb(pdb_in)
            if ok:
                st.session_state.local_target_path = path
                st.session_state.pdb_id_display = pdb_in.upper()
                conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                st.session_state.target_ready = conv_ok
                st.session_state.protein_metadata = extract_pdb_metadata(
                    path, pdb_in.upper())
                st.success(f"Protein {pdb_in.upper()} loaded.")
                st.rerun()
            else:
                st.error(path)
    else:
        up_p = st.file_uploader("Upload protein", type=["pdb","pdbqt"], key="up_prot")
        if up_p:
            path = f"uploaded_{up_p.name}"
            with open(path,"wb") as f: f.write(up_p.getbuffer())
            st.session_state.local_target_path = path
            st.session_state.pdb_id_display = "Uploaded File"
            if up_p.name.endswith(".pdb"):
                ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                st.session_state.target_ready = ok
            else:
                os.replace(path, "protein.pdbqt")
                st.session_state.target_ready = True
            st.session_state.protein_metadata = extract_pdb_metadata(
                path, "Uploaded")
            st.success("Protein uploaded.")
            st.rerun()

    if st.session_state.target_ready:
        m = st.session_state.protein_metadata
        st.markdown(
            f"> **PDB ID:** `{m.get('id','?')}` | **Class:** {m.get('class','?')}\n"
            f"> **Organism:** *{m.get('organism','?')}* | **Method:** {m.get('method','?')} | "
            f"**Resolution:** **{m.get('res','?')}**")

with col_b:
    st.subheader("💊 Original Ligand")
    l_mode = st.radio("Ligand input:", ["SMILES string","Upload (.pdb / .sdf)"],
                      horizontal=True, key="lig_mode")
    smiles_in = ""
    up_lig = None
    if l_mode == "SMILES string":
        smiles_in = st.text_input("SMILES", value="CC(=O)NC1=CC=C(O)C=C1",
                                   help="default = paracetamol").strip()
    else:
        up_lig = st.file_uploader("Upload ligand", type=["pdb","sdf"], key="up_lig")

    if st.button("📥 Load Ligand", key="btn_load_lig"):
        if l_mode == "SMILES string" and smiles_in:
            with st.spinner("Querying PubChem & preparing ligand..."):
                pub = fetch_ligand_data_from_pubchem(smiles_in)
                ok, _ = convert_smiles_to_pdbqt(smiles_in, "ligand.pdbqt")
                if ok:
                    st.session_state.ligand_ready = True
                    st.session_state.original_smiles = smiles_in
                    with open("ligand.pdbqt","r") as f:
                        st.session_state.original_ligand_block = f.read()
                    st.session_state.original_ligand_summary = (
                        f"**Name:** {pub['name']} | **Formula:** {pub['formula']} | "
                        f"**MW:** {pub['mw']}")
                    st.success("Ligand prepared from SMILES.")
                    st.rerun()
        elif up_lig:
            tmp_in = f"raw_ligand_{up_lig.name}"
            with open(tmp_in,"wb") as f: f.write(up_lig.getbuffer())
            mol = (Chem.MolFromPDBFile(tmp_in, removeHs=False)
                   if up_lig.name.endswith(".pdb")
                   else Chem.SDMolSupplier(tmp_in, removeHs=False)[0])
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                if mol.GetNumConformers() == 0:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol)
                tmp_pdb = "temp_lig_state.pdb"
                Chem.MolToPDBFile(mol, tmp_pdb)
                convert_pdb_to_pdbqt(tmp_pdb, "ligand.pdbqt", is_ligand=True)
                st.session_state.ligand_ready = True
                st.session_state.original_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                with open("ligand.pdbqt","r") as f:
                    st.session_state.original_ligand_block = f.read()
                st.session_state.original_ligand_summary = (
                    "Original ligand loaded from uploaded file.")
                if os.path.exists(tmp_in): os.remove(tmp_in)
                if os.path.exists(tmp_pdb): os.remove(tmp_pdb)
                st.success("Ligand loaded.")
                st.rerun()

    if st.session_state.ligand_ready:
        st.markdown(f"> {st.session_state.original_ligand_summary}")
        st.code(st.session_state.original_smiles, language="text")
        b64 = smiles_to_2d_img_b64(st.session_state.original_smiles, size=300)
        if b64:
            st.markdown(f'<div style="text-align:center;background:white;padding:8px;'
                        f'border-radius:6px;"><img src="data:image/png;base64,{b64}"/></div>',
                        unsafe_allow_html=True)

# Co-crystal grid auto-fill
if st.session_state.target_ready and st.session_state.local_target_path:
    bound = parse_bound_ligands(st.session_state.local_target_path)
    if bound:
        with st.expander("📦 Bound co-crystal ligands in structure (auto-grid)", expanded=False):
            dfb = pd.DataFrame(bound)
            dfb["Center (X,Y,Z) Å"] = dfb.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
            dfb["Box (X,Y,Z) Å"]    = dfb.apply(lambda r: f"{r['bx']}, {r['by']}, {r['bz']}", axis=1)
            st.dataframe(dfb[["ID","Chain","ResSeq","Atoms",
                              "Center (X,Y,Z) Å","Box (X,Y,Z) Å"]],
                         hide_index=True, use_container_width=True)
            sel = st.selectbox("Select co-crystal target to auto-fill grid:",
                               options=range(len(bound)),
                               format_func=lambda i: f"{bound[i]['ID']} "
                                                     f"(Chain {bound[i]['Chain']}-{bound[i]['ResSeq']})")
            if st.button("🎯 Lock grid to this pocket"):
                c = bound[sel]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = c["cx"], c["cy"], c["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = c["bx"], c["by"], c["bz"]
                st.success("Grid box locked.")
                st.rerun()

st.subheader("📐 Grid Box & Search")
c1, c2, c3 = st.columns(3)
with c1:
    st.session_state.cx = st.number_input("Center X", value=float(st.session_state.cx), step=0.1)
    st.session_state.sx = st.slider("Size X (Å)", 10, 40, int(st.session_state.sx))
with c2:
    st.session_state.cy = st.number_input("Center Y", value=float(st.session_state.cy), step=0.1)
    st.session_state.sy = st.slider("Size Y (Å)", 10, 40, int(st.session_state.sy))
with c3:
    st.session_state.cz = st.number_input("Center Z", value=float(st.session_state.cz), step=0.1)
    st.session_state.sz = st.slider("Size Z (Å)", 10, 40, int(st.session_state.sz))
st.session_state.exhaustiveness = st.slider(
    "Exhaustiveness", 4, 32, int(st.session_state.exhaustiveness), step=4,
    help="Higher = more thorough but slower. Same value used for both runs.")

# Show current 3D
if st.session_state.target_ready or st.session_state.ligand_ready:
    with st.expander("🔭 3D viewport — receptor + original ligand", expanded=False):
        rec_data = ""
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            with open("protein.pdbqt","r") as f: rec_data = f.read()
        render_3d_complex(rec_data, st.session_state.original_ligand_block or "",
                          mode="cartoon")


# =====================================================================
# 13. SECTION 2 — RUN ORIGINAL DOCKING
# =====================================================================
st.write("---")
st.header("STEP 2 — Run Docking on the Original Ligand")

can_run_orig = bool(st.session_state.target_ready and st.session_state.ligand_ready)
if not can_run_orig:
    st.info("⏳ Load both the protein and the original ligand to enable docking.")

if st.button("🚀 Run Original Docking", type="primary", disabled=not can_run_orig):
    grid = {"cx":st.session_state.cx, "cy":st.session_state.cy, "cz":st.session_state.cz,
            "sx":st.session_state.sx, "sy":st.session_state.sy, "sz":st.session_state.sz}
    out_log, rc = run_vina_docking("ligand.pdbqt", "docking_original.pdbqt",
                                    grid, st.session_state.exhaustiveness)
    if rc == 0:
        st.session_state.original_docking_raw = out_log
        st.session_state.original_poses_file = "docking_original.pdbqt"
        df = parse_vina_output(out_log, "docking_original.pdbqt")
        st.session_state.original_df = df
        st.session_state.original_best_affinity = best_affinity_from_df(df)
        st.success(f"Original docking complete. Best affinity: "
                   f"{st.session_state.original_best_affinity:.2f} kcal/mol")
        st.rerun()
    else:
        st.error("Original docking failed.")
        st.code(out_log)

if st.session_state.original_df is not None:
    st.subheader("📊 Original Ligand — Binding Modes")
    bcol1, bcol2 = st.columns([2,1])
    with bcol1:
        st.dataframe(st.session_state.original_df, hide_index=True, use_container_width=True)
    with bcol2:
        st.metric("Best Affinity (Original)",
                  f"{st.session_state.original_best_affinity:.2f} kcal/mol")

    with st.expander("👁 View original docking poses (3D)", expanded=False):
        poses = split_docking_poses("docking_original.pdbqt")
        if poses:
            mode_sel = st.selectbox("Pose:", list(poses.keys()),
                                    format_func=lambda x: f"Mode {x}", key="orig_pose")
            with open("protein.pdbqt","r") as f: rdata = f.read()
            ints = compute_spatial_interactions("protein.pdbqt", poses[mode_sel])
            render_3d_complex(rdata, poses[mode_sel], mode="cartoon",
                              show_surface=False, interactions_list=ints)


# =====================================================================
# 14. SECTION 3 — REDESIGN LIBRARY
# =====================================================================
st.write("---")
st.header("STEP 3 — Generate Redesigned Ligand Library")

if st.session_state.original_best_affinity is None:
    st.info("⏳ Run the original docking (Step 2) first so the redesigner can compare against it.")
else:
    cls, _ = get_dynamic_fragments(st.session_state.original_smiles)
    st.write(f"🔬 **AI-classified scaffold:** `{cls}`")
    sites = find_valid_cleavage_sites(st.session_state.original_smiles)
    if not sites:
        st.warning("No valid covalent substitution sites found. Forcing co-crystal mode.")
        reaction_mode = "Co-Crystal / Salt Formulation (Non-Covalent)"
        target_idx = 0
    else:
        reaction_mode = st.radio(
            "Modification mechanism:",
            ["True Covalent Substitution (Cleavage & Attachment)",
             "Co-Crystal / Salt Formulation (Non-Covalent)"], horizontal=False)
        if reaction_mode.startswith("True"):
            show_labels = st.toggle("Show atom indices", value=True)
            b64 = smiles_to_2d_img_b64(st.session_state.original_smiles,
                                       size=520, include_labels=show_labels)
            if b64:
                st.markdown(f'<div style="text-align:center;background:white;padding:8px;'
                            f'border-radius:6px;"><img src="data:image/png;base64,{b64}"/></div>',
                            unsafe_allow_html=True)
            site_opts = {s["label"]: s["index"] for s in sites}
            sel_label = st.selectbox("Target atom:", list(site_opts.keys()))
            target_idx = site_opts[sel_label]
        else:
            target_idx = 0
            st.info("Co-crystal mode: functional group attached non-covalently.")

    if st.button("🧪 Generate Redesigned Library", type="primary"):
        with st.spinner("Building variants..."):
            lib = run_cleaving_engine(st.session_state.original_smiles, target_idx,
                                       reaction_mode,
                                       original_best_affinity=st.session_state.original_best_affinity)
            if lib:
                st.session_state.redesign_library = pd.DataFrame(lib)
                st.session_state.selected_variant_id = None
                # Clear any prior redesigned docking
                st.session_state.redesign_docking_raw = None
                st.session_state.redesign_best_affinity = None
                st.session_state.redesign_df = None
                st.success(f"Generated {len(lib)} variants.")
                st.rerun()
            else:
                st.error("Could not generate variants.")

if st.session_state.redesign_library is not None:
    st.subheader("📚 Redesigned Variant Library")
    df_lib = st.session_state.redesign_library
    st.dataframe(df_lib[["Variant ID","Fragment Added","Redesigned SMILES",
                         "Predicted Affinity (kcal/mol)","MW (g/mol)","LogP","Yield Prediction"]],
                 hide_index=True, use_container_width=True)


# =====================================================================
# 15. SECTION 4 — SELECT A VARIANT + RE-DOCK IT
# =====================================================================
st.write("---")
st.header("STEP 4 — Select a Variant & Run Comparative Docking")

if st.session_state.redesign_library is None:
    st.info("⏳ Generate the redesign library first (Step 3).")
else:
    df_lib = st.session_state.redesign_library
    sel_id = st.selectbox("Pick a redesigned variant to dock:",
                           options=df_lib["Variant ID"].tolist())
    st.session_state.selected_variant_id = sel_id
    row = df_lib[df_lib["Variant ID"] == sel_id].iloc[0]
    st.write(f"**Variant:** `{row['Variant ID']}` | **Fragment:** {row['Fragment Added']}")
    st.code(row["Redesigned SMILES"], language="text")

    b64v = smiles_to_2d_img_b64(row["Redesigned SMILES"], size=400)
    if b64v:
        st.markdown(f'<div style="text-align:center;background:white;padding:8px;'
                    f'border-radius:6px;"><img src="data:image/png;base64,{b64v}"/></div>',
                    unsafe_allow_html=True)

    st.caption("Note: Docking will use the SAME protein and the SAME grid box "
               "you configured in Step 1 — only the ligand changes.")

    if st.button("🚀 Run Comparative Docking on this Variant", type="primary"):
        sm = str(row["Redesigned SMILES"])
        ok, _ = convert_smiles_to_pdbqt(sm, "redesigned_ligand.pdbqt")
        if not ok:
            st.error("Could not prepare redesigned ligand (SMILES may be invalid for 3D embedding).")
        else:
            grid = {"cx":st.session_state.cx, "cy":st.session_state.cy, "cz":st.session_state.cz,
                    "sx":st.session_state.sx, "sy":st.session_state.sy, "sz":st.session_state.sz}
            out_log, rc = run_vina_docking("redesigned_ligand.pdbqt",
                                            "docking_redesigned.pdbqt",
                                            grid, st.session_state.exhaustiveness)
            if rc == 0:
                st.session_state.redesign_docking_raw = out_log
                st.session_state.redesign_poses_file = "docking_redesigned.pdbqt"
                df = parse_vina_output(out_log, "docking_redesigned.pdbqt")
                st.session_state.redesign_df = df
                st.session_state.redesign_best_affinity = best_affinity_from_df(df)
                st.session_state.redesign_smiles_used = sm
                st.success(f"Redesigned docking complete. Best affinity: "
                           f"{st.session_state.redesign_best_affinity:.2f} kcal/mol")
                st.rerun()
            else:
                st.error("Redesigned docking failed.")
                st.code(out_log)


# =====================================================================
# 16. SECTION 5 — COMPARATIVE RESULTS
# =====================================================================
if (st.session_state.original_best_affinity is not None and
        st.session_state.redesign_best_affinity is not None):

    st.write("---")
    st.header("STEP 5 — Comparative Docking Results")

    orig = st.session_state.original_best_affinity
    redes = st.session_state.redesign_best_affinity
    delta = redes - orig

    cA, cB, cC = st.columns(3)
    cA.metric("Original Best Affinity", f"{orig:.2f} kcal/mol")
    cB.metric("Redesigned Best Affinity", f"{redes:.2f} kcal/mol",
              delta=f"{delta:+.2f}", delta_color=("inverse" if delta < 0 else "normal"))
    if delta < -0.1:
        cC.success("✅ Redesigned ligand binds **more strongly** than the original.")
    elif abs(delta) <= 0.1:
        cC.info("⚖️ Affinities are essentially comparable.")
    else:
        cC.error("❌ Redesigned ligand binds **less strongly** than the original.")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(["Original", "Redesigned"], [orig, redes],
                  color=["#64748b", "#1e3c72"])
    ax.set_ylabel("Binding Affinity (kcal/mol)")
    ax.set_title("Best-pose Affinity Comparison (more negative = stronger)")
    for b, v in zip(bars, [orig, redes]):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)

    cols_t1, cols_t2 = st.columns(2)
    with cols_t1:
        st.subheader("Original — Binding Modes")
        st.dataframe(st.session_state.original_df, hide_index=True, use_container_width=True)
    with cols_t2:
        st.subheader("Redesigned — Binding Modes")
        st.dataframe(st.session_state.redesign_df, hide_index=True, use_container_width=True)

    with st.expander("👁 View redesigned docking poses (3D)", expanded=False):
        poses = split_docking_poses("docking_redesigned.pdbqt")
        if poses:
            mode_sel = st.selectbox("Pose:", list(poses.keys()),
                                    format_func=lambda x: f"Mode {x}", key="redes_pose")
            with open("protein.pdbqt","r") as f: rdata = f.read()
            ints = compute_spatial_interactions("protein.pdbqt", poses[mode_sel])
            render_3d_complex(rdata, poses[mode_sel], mode="cartoon",
                              show_surface=False, interactions_list=ints)


# =====================================================================
# 17. SECTION 6 — ADME + REPORT (only when both dockings exist)
# =====================================================================
if (st.session_state.original_best_affinity is not None and
        st.session_state.redesign_best_affinity is not None and
        st.session_state.selected_variant_id is not None):

    st.write("---")
    st.header("STEP 6 — ADME Comparison + Final Report")

    df_lib = st.session_state.redesign_library
    sel_row = df_lib[df_lib["Variant ID"] == st.session_state.selected_variant_id].iloc[0]

    orig_smiles = st.session_state.original_smiles
    redes_smiles = str(sel_row["Redesigned SMILES"])

    with st.spinner("Computing ADME for original + redesigned..."):
        adme_p = calculate_advanced_adme(orig_smiles)
        adme_v = calculate_advanced_adme(redes_smiles)
        iupac_name = get_iupac_name(redes_smiles)

    if adme_p and adme_v:
        st.info(f"**IUPAC name of redesigned variant:** `{iupac_name}`")

        with st.expander("📖 ADMET parameter dictionary", expanded=False):
            st.markdown(
                "- **TPSA**: polar surface area — affects permeability "
                "(≤132 Å² for intestinal, ≤79 Å² for BBB).\n"
                "- **Volume**: 3D spatial footprint.\n"
                "- **MaxRing**: largest ring atom count.\n"
                "- **pKa**: ionization at pH 7.4.\n"
                "- **MP/BP**: estimated thermodynamics.\n"
                "- **Lipinski Rule of 5**: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.")

        comp_df = pd.DataFrame({
            "Parameter": [
                "Obey Lipinski?", "Oral Bioavailability", "Permeability Profile",
                "TPSA (Å²)", "Volume (Å³)", "Max Ring Size",
                "pKa (Acidic)", "pKa (Basic)",
                "Est. MP (°C)", "Est. BP (°C)", "LogP"
            ],
            "Original Lead": [
                adme_p["Lipinski_Obey"], adme_p["Oral_Bio"], adme_p["Permeability"],
                f"{adme_p['TPSA']:.2f}", f"{adme_p['Volume']:.1f}", adme_p["MaxRing"],
                adme_p["pKa_Acid"], adme_p["pKa_Base"],
                f"{adme_p['MP']:.1f}", f"{adme_p['BP']:.1f}", f"{adme_p['LogP']:.2f}"
            ],
            "Redesigned Variant": [
                adme_v["Lipinski_Obey"], adme_v["Oral_Bio"], adme_v["Permeability"],
                f"{adme_v['TPSA']:.2f}", f"{adme_v['Volume']:.1f}", adme_v["MaxRing"],
                adme_v["pKa_Acid"], adme_v["pKa_Base"],
                f"{adme_v['MP']:.1f}", f"{adme_v['BP']:.1f}", f"{adme_v['LogP']:.2f}"
            ]
        })
        st.subheader("📊 ADMET Comparative Matrix")
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # Shift narrative
        tpsa_shift = adme_v["TPSA"] - adme_p["TPSA"]
        vol_shift  = adme_v["Volume"] - adme_p["Volume"]
        logp_shift = adme_v["LogP"] - adme_p["LogP"]
        shift = f"The redesign produced a volumetric shift of **{vol_shift:+.1f} Å³**. "
        if tpsa_shift > 0:
            shift += f"Polarity (TPSA) **increased by {tpsa_shift:.1f} Å²**. "
        elif tpsa_shift < 0:
            shift += f"Polarity (TPSA) **decreased by {abs(tpsa_shift):.1f} Å²**. "
        if adme_p["BBB"] and not adme_v["BBB"]:
            shift += "BBB penetration was **lost** — molecule restricted to GI absorption. "
        elif not adme_p["BBB"] and adme_v["BBB"]:
            shift += "BBB penetration was **unlocked** — CNS targeting now possible. "
        elif adme_v["BBB"]:
            shift += "BBB penetration was **retained**. "
        elif adme_v["HIA"]:
            shift += "Good GI absorption was retained; BBB still restricted. "
        else:
            shift += "Molecule is currently impermeable to both GI and BBB barriers. "
        if logp_shift > 0.5:
            shift += "Lipophilicity (LogP) rose notably — may require lipid-based formulation. "
        elif logp_shift < -0.5:
            shift += "Lipophilicity (LogP) dropped — likely improves aqueous solubility. "

        if adme_v["Violations"] < adme_p["Violations"]:
            verdict = "✅ **Overall: favorable.** Redesign improves Lipinski compliance."
        elif adme_v["Violations"] > adme_p["Violations"]:
            verdict = "❌ **Overall: unfavorable.** Redesign introduces new Lipinski violations."
        else:
            if adme_v["Violations"] <= 1 and adme_v["Permeability"] != "Poor Absorption / Impermeable":
                verdict = "⚖️ **Overall: comparable.** Redesign is a viable alternative."
            else:
                verdict = "⚠️ **Overall: comparable but flawed.** Both molecules share concerns."
        shift_text = shift + "\n\n" + verdict
        st.success(shift_text)

        # FTIR for redesigned
        st.subheader("📈 Predicted FTIR Footprint of Redesigned Variant")
        ftir_b64 = generate_ftir_image_html(int(sel_row["FTIR Peak"]))
        st.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{ftir_b64}" '
                    f'style="max-width:100%;border-radius:8px;border:1px solid #e2e8f0;"/></div>',
                    unsafe_allow_html=True)

        # Generate full HTML report
        st.subheader("📄 Download Comprehensive Report")
        parent_b64 = smiles_to_2d_img_b64(orig_smiles, size=300)
        variant_b64 = smiles_to_2d_img_b64(redes_smiles, size=300)

        html = generate_comparative_html_report(
            protein_id=st.session_state.pdb_id_display,
            protein_meta=st.session_state.protein_metadata,
            original_smiles=orig_smiles,
            original_best_aff=st.session_state.original_best_affinity,
            original_df=st.session_state.original_df,
            redesigned_variant_row=sel_row,
            redesigned_smiles=redes_smiles,
            redesigned_best_aff=st.session_state.redesign_best_affinity,
            redesigned_df=st.session_state.redesign_df,
            iupac_name=iupac_name,
            adme_parent=adme_p, adme_variant=adme_v,
            parent_img_b64=parent_b64, variant_img_b64=variant_b64,
            ftir_img_b64=ftir_b64, shift_text=shift_text
        )

        st.download_button(
            label="📥 Download Full HTML Report (Docking + Redesign + ADME)",
            data=html,
            file_name=f"InSilico_Unified_Report_{sel_row['Variant ID']}.html",
            mime="text/html",
            use_container_width=True
        )

        # Also offer compact CSV of both docking runs
        merged = pd.concat([
            st.session_state.original_df.assign(Source="Original"),
            st.session_state.redesign_df.assign(Source="Redesigned ("+sel_row["Variant ID"]+")")
        ])
        st.download_button(
            label="📥 Download Combined Docking CSV",
            data=merged.to_csv(index=False).encode("utf-8"),
            file_name="comparative_docking.csv",
            mime="text/csv",
            use_container_width=True
        )

st.write("---")
st.caption("InSilico BioSphere — Unified Studio • copyright © Sarang S. Dhote")
