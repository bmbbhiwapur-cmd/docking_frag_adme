# InSilico BioSphere — Unified Docking + Redesign + ADME Studio

A single Streamlit application that walks you through a complete in-silico
drug-design workflow against a single protein target:

## Workflow

| Step | What happens |
|------|--------------|
| **1** | Load protein (PDB ID or upload) + Original ligand (SMILES or .pdb/.sdf) + Grid box |
| **2** | Run AutoDock Vina docking on the **original** ligand → record best affinity |
| **3** | Generate a redesigned ligand **library** by substituting functional groups (RDKit fragment cleaving) |
| **4** | **User picks** any variant from the library → automatically re-dock it against the **same protein** with the **same grid box** |
| **5** | **Comparative report**: side-by-side Best-Affinity, full binding-mode tables, bar chart, 3D pose viewer |
| **6** | ADME comparison (Lipinski, TPSA, Volume, LogP, BBB, HIA, MP/BP, pKa) + IUPAC name + predicted FTIR + downloadable HTML report |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app auto-downloads the Linux AutoDock Vina 1.2.5 binary on first run, so
it works out of the box on Streamlit Cloud or any Linux box with network
access to GitHub.

## What changed vs. your three original scripts

- The **docking** script and the **redesign + ADME** script are now one app
  sharing session state (no copy-paste between tools).
- The redesign library now **uses the original docking affinity as a baseline**
  when predicting the Δ score of each variant.
- A new **Step 4** lets the user pick any variant and **re-dock it against the
  same receptor with the same grid box** — the missing comparative-docking step
  you asked for.
- A new **Step 5** shows side-by-side best-affinity metrics, a delta verdict,
  a bar chart, both binding-mode tables, and a 3D pose viewer for the
  redesigned ligand.
- The downloadable HTML report now contains **both** docking runs, the ADME
  comparison, IUPAC name, predicted FTIR, and the structural-shift narrative,
  in a single document.

## Notes

- More-negative kcal/mol = stronger binding.
- The same `exhaustiveness` and grid box are deliberately used for both dockings
  so the comparison is apples-to-apples.
- If the redesigned SMILES can't be embedded in 3D (rare — usually disconnected
  co-crystal SMILES with a `.`), the app will tell you and you can pick another
  variant.

---
Developed by **Mr. Sarang S. Dhote**, Assistant Professor,
Department of Chemistry, Shivaji Science College, Nagpur, India.
Contact: sarangresearch@gmail.com
