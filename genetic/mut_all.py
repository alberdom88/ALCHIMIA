import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import secrets

RNG_SEED = secrets.randbelow(10000000000000000000)
random.seed(RNG_SEED)

ALLOWED_ELEMENTS = {"C", "N", "O", "F", "S", "Cl", "Br", "I", "P", "B"}
MAX_HEAVY_ATOMS = 40

MAX_CORE_FRAG_HEAVY_ATOMS = 18
MAX_DECOR_FRAG_HEAVY_ATOMS = 6

MAX_DECOR_DELETE_SIZE = 6
MAX_GENERAL_DELETE_SIZE = 12

MAX_ACTION_RETRIES = 20

BASE_CORE_FRAGMENTS_SMILES = [
    # === AROMATIC & HETEROAROMATIC CORES (45 fragments) ===
    # Original aromatic/heteroaromatic
    "[*:1]c1ccccc1", "[*:1]c1ccncc1", "[*:1]c1ncccc1",
    "[*:1]c1nccnc1", "[*:1]c1nccs1",  "[*:1]c1ccoc1",
    "[*:1]c1ccsc1",  "[*:1]c1ccc2ccccc2c1",
    "[*:1]c1ccc(N)cc1", "[*:1]c1ccc(Cl)cc1", "[*:1]c1ccc(F)cc1",

    # Pyrazoles, imidazoles, triazoles
    "[*:1]c1ccnn1", "[*:1]c1cnnc1", "[*:1]c1nncn1",
    "[*:1]c1nncc1", "[*:1]c1ncnc1", "[*:1]c1nnnc1", "[*:1]c1nnnn1",

    # Oxazoles and thiazoles variants
    "[*:1]c1ocnc1", "[*:1]c1scnc1", "[*:1]c1ncco1", 
    "[*:1]c1nccs1", "[*:1]c1nocs1",

    # Indoles and benzofused systems
    "[*:1]c1ccc2[nH]ccc2c1", "[*:1]c1ccc2ncccc2c1",
    "[*:1]c1ccc2occc2c1", "[*:1]c1ccc2sccc2c1",

    # Pyrimidines and bicyclic aromatics
    "[*:1]c1ncccn1", "[*:1]c1nccnc1", "[*:1]c1ncncc1",
    "[*:1]c1ccnc2ccccc12", "[*:1]c1ncc2ccccc2c1",

    # Substituted benzenes
    "[*:1]c1ccc(O)cc1", "[*:1]c1ccc(S)cc1", "[*:1]c1ccc(C#N)cc1",
    "[*:1]c1ccc(CF3)cc1", "[*:1]c1ccc(NO2)cc1", "[*:1]c1ccc(C)cc1",
    "[*:1]c1ccc(OC)cc1", "[*:1]c1ccc(C(=O)O)cc1",

    # Disubstituted benzenes
    "[*:1]c1cc(F)ccc1F", "[*:1]c1cc(Cl)ccc1Cl", "[*:1]c1cc(O)ccc1O",
    "[*:1]c1cc(N)ccc1N", "[*:1]c1cc(C)ccc1C",

    # Complex aromatic systems
    "[*:1]c1ccc2nc3ccccc3nc2c1",  # Phenanthroline
    "[*:1]c1cnc2c(c1)ccc1ccccc12",  # Phenanthridine
    "[*:1]c1ccc2c(c1)ncs2",  # Benzothiazole
    "[*:1]c1ccc2c(c1)nco2",  # Benzoxazole
    "[*:1]c1ccc2c(c1)nc[nH]2",  # Benzimidazole variant

    # === SATURATED & SEMI-SATURATED HETEROCYCLES (29 fragments) ===
    # Original heterocycles
    "[*:1]C1CCCCC1", "[*:1]N1CCCCC1", "[*:1]N1CCNCC1",
    "[*:1]N1CCOCC1", "[*:1]C1COCCO1",

    # 4-membered rings
    "[*:1]C1CCC1", "[*:1]N1CCC1", "[*:1]O1CCC1",

    # 5-membered rings
    "[*:1]C1CCCC1", "[*:1]N1CCCC1", "[*:1]O1CCCC1", 
    "[*:1]S1CCCC1", "[*:1]C1CNCC1", "[*:1]C1COCC1", "[*:1]C1CSCC1",

    # 6-membered ring variants
    "[*:1]O1CCCCC1", "[*:1]S1CCCCC1", "[*:1]C1CNCCC1",
    "[*:1]C1COCCC1", "[*:1]C1CSCCC1",

    # 7-membered rings
    "[*:1]C1CCCCCC1", "[*:1]N1CCCCCC1", "[*:1]O1CCCCCC1",

    # Bicyclic and spiro systems
    "[*:1]C1CCC2CCCCC2C1", "[*:1]C1CCN2CCCCC2C1",
    "[*:1]C1CCC2(CCCC2)CC1", "[*:1]C1CCN2(CCCC2)CC1",
    "[*:1]C1CC2CCC1C2", "[*:1]C1CN2CCC1CC2",

    # Lactams and sultams
    "[*:1]C1CCNC1=O", "[*:1]C1CCCNC1=O", "[*:1]C1CCCCNC1=O",
    "[*:1]C1CCS(=O)(=O)N1", "[*:1]C1CCCS(=O)(=O)N1",

    # === PRIVILEGED DRUG STRUCTURES (23 fragments) ===
    # Piperazine variants
    "[*:1]N1CCN(C)CC1", "[*:1]N1CCN(C(=O)C)CC1",
    "[*:1]N1CCN(CC)CC1", "[*:1]N1CCN(c2ccccc2)CC1",

    # Piperidine variants  
    "[*:1]N1CCCCC1", "[*:1]N1CCC(O)CC1", "[*:1]N1CCC(C)CC1",
    "[*:1]N1CCC(N)CC1", "[*:1]N1CCC(=O)CC1",

    # Morpholine variants
    "[*:1]N1CCO1", "[*:1]N1CC(C)OCC1",

    # Benzimidazole
    "[*:1]c1nc2ccccc2[nH]1", "[*:1]c1nc2ccccc2n1C",

    # Quinoline/isoquinoline variants
    "[*:1]c1ccnc2ccccc12", "[*:1]c1cnc2ccccc2c1",

    # Pyrazine variants
    "[*:1]c1cnccn1", "[*:1]c1nccnc1",

    # Thiophene and furan variants
    "[*:1]c1ccsc1", "[*:1]c1sccc1C", "[*:1]c1sc(C)cc1",
    "[*:1]c1ccoc1", "[*:1]c1occc1C", "[*:1]c1oc(C)cc1",

    # Pyridone variants
    "[*:1]C1=CC(=O)N=CC1", "[*:1]C1=CNC(=O)C=C1",

    # === LINKERS & FUNCTIONAL GROUPS (104 fragments) ===
    # Original simple linkers
    "[*:1]C", "[*:1]CC", "[*:1]CCC", "[*:1]CO", "[*:1]COC", "[*:1]OCC",

    # Extended alkyl chains
    "[*:1]CCCC", "[*:1]CCCCC", "[*:1]CCCCCC",

    # Branched alkyl
    "[*:1]C(C)C", "[*:1]C(C)(C)C", "[*:1]CC(C)C",
    "[*:1]C(C)CC", "[*:1]CC(C)CC",

    # Oxygen linkers
    "[*:1]CCOC", "[*:1]CCOCC", "[*:1]OCCOC",
    "[*:1]COCCO", "[*:1]OCCCO",

    # Nitrogen linkers
    "[*:1]CCN", "[*:1]NCCN", "[*:1]CCNCC",
    "[*:1]NCC", "[*:1]CNCC", "[*:1]NCCNC",

    # Carbonyl groups
    "[*:1]C(=O)N", "[*:1]C(=O)O", "[*:1]C(=O)OC",
    "[*:1]C(=O)CC", "[*:1]CCC(=O)C", "[*:1]C(=O)CCC",
    "[*:1]CC(=O)N", "[*:1]NC(=O)C", "[*:1]C(=O)NC",

    # Amide variants
    "[*:1]NC(=O)N", "[*:1]OC(=O)N", "[*:1]NC(=O)O",

    # Sulfonyl groups
    "[*:1]S(=O)(=O)N", "[*:1]S(=O)(=O)C", "[*:1]CS(=O)(=O)", 
    "[*:1]S(=O)(=O)CC", "[*:1]CCS(=O)(=O)", "[*:1]S(=O)(=O)O",

    # Phosphonyl groups
    "[*:1]P(=O)(O)O", "[*:1]P(=O)(C)C", "[*:1]CP(=O)(O)O",
    "[*:1]P(=O)(OC)(OC)", "[*:1]CP(=O)(OC)OC",

    # Nitrile variants
    "[*:1]C#N", "[*:1]CC#N", "[*:1]CCC#N", "[*:1]C(C#N)C",

    # Ester variants
    "[*:1]OC(=O)CC", "[*:1]C(=O)OCC", "[*:1]COC(=O)C",

    # Ether variants
    "[*:1]COCC", "[*:1]CCOCC", "[*:1]OCCOC",

    # Amine variants
    "[*:1]N", "[*:1]NCC", "[*:1]CCN(C)C", "[*:1]N(C)CC",
    "[*:1]NC", "[*:1]N(C)C", "[*:1]NNC",

    # Thioether variants
    "[*:1]S", "[*:1]SCC", "[*:1]CSCC", "[*:1]SCCSC",

    # Halogen variants
    "[*:1]CF", "[*:1]CCF", "[*:1]CF3", "[*:1]CCl", "[*:1]CBr", "[*:1]CI",
    "[*:1]CCCF", "[*:1]CCF", "[*:1]CCCCF", "[*:1]C(F)(F)C", "[*:1]C(F)(F)CC",

    # === MEDICINAL CHEMISTRY BIOISOSTERES (16 fragments) ===
    # Carboxylic acid bioisosteres
    "[*:1]C(=O)NHOH", "[*:1]C(=O)NHSO2CF3", "[*:1]C1NNC(=O)O1",
    "[*:1]c1nnn[nH]1", "[*:1]S(=O)(=O)NH2",

    # Amide bioisosteres
    "[*:1]CNHSO2", "[*:1]C=NNH2", "[*:1]CNHCO",
    "[*:1]COC=N", "[*:1]C=NOH",

    # Ether bioisosteres
    "[*:1]CNHC", "[*:1]C(=O)NHC", "[*:1]CSC",

    # Benzene bioisosteres
    "[*:1]C1=CC=NC=C1", "[*:1]c1nccnc1", "[*:1]C1CCC=CC1",

    # Special functional groups
    "[*:1]C(=O)N(O)C", "[*:1]C(=O)N(O)CC",  # Hydroxamic acids
    "[*:1]C(=O)NNC", "[*:1]C(=O)NNCC",  # Hydrazides
    "[*:1]NC(=N)N", "[*:1]NC(=N)NC", "[*:1]N=C(N)NC",  # Guanidines
    "[*:1]C=NNC(=O)N", "[*:1]CC=NNC(=O)N",  # Semicarbazones
    "[*:1]C1(C)CCCC1", "[*:1]C1(CC)CCCC1",  # Spiro centers
]

DECORATION_SMILES = [
    "[*:1]F", "[*:1]Cl", "[*:1]Br", "[*:1]I",
    "[*:1]C(F)(F)F",      # CF3
    "[*:1]CO",            # OMe
    "[*:1][N+][O-]",  # nitro
    "[*:1]S(=O)(=O)C"     # metil-sulfonile
]

# =========================
# Librerie nuovi gruppi (POLAR / LIPOPHILIC)
# =========================
POLAR_GROUP_SMILES = [
    "[*:1]O",             # -OH (alcol/fenolo)
    "[*:1]N",             # -NH2 (ammina)
    "[*:1]C(=O)O",        # -C(=O)OH (carbossilico)
    "[*:1]C(=O)N",        # -CONH2 (ammide primaria)
    "[*:1]NC(=O)C",       # -NHCOCH3 (acetammide)
    "[*:1]S(=O)(=O)N",    # -SO2NH2 (sulfonammide)
    "[*:1]P(=O)(O)O",     # -PO3H2 (fosfonico)
]

LIPOPHILIC_GROUP_SMILES = [
    # Alkyl
    "[*:1]C",          # Me
    "[*:1]CC",         # Et
    "[*:1]CCC",        # n-Pr
    "[*:1]C(C)C",      # i-Pr
    "[*:1]C(C)(C)C",   # t-Bu
    # Aryl
    "[*:1]c1ccccc1",   # fenile
]


REFERENCE_SMILES = "brics_10_top01.csv"

# =========================
# Configurazione (aggiunta)
# =========================
MAX_PROTECT_FRAG_HEAVY_ATOMS = 14  # Boc/Cbz richiedono una soglia > DECOR (6)

# =========================
# Libreria gruppi protettivi (nuova)
# Nota: l'atomo fittizio [*:1] è posizionato sul C acilico o sullo S solfonile
# in modo che l'aggancio su N formi N–C(=O)–O–R (carbammati) o N–S(=O)2–R (solfonammidi).
# =========================
PROTECTING_GROUP_SMILES = [
    "[*:1]C(=O)OC(C)(C)C",            # Boc
    "[*:1]NC(=O)OCc1ccccc1",         # Cbz
    "[*:1]C(=O)C",                    # Ac (N-acetile)
    "[*:1]S(=O)(=O)c1ccc(C)cc1",      # Ts (tosile)
    "[*:1]S(=O)(=O)C"                 # Ms (mesile)
]


_USE_TRUE_SA = False
def calc_sa_proxy(m: Chem.Mol) -> float:
    nHvy = m.GetNumHeavyAtoms()
    nAromRings = rdmd.CalcNumAromaticRings(m)
    nBridge = rdmd.CalcNumBridgeheadAtoms(m)
    nSpiro = rdmd.CalcNumSpiroAtoms(m)
    nHetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    score = 0.10*nHvy + 0.60*nAromRings + 1.20*nBridge + 1.00*nSpiro + 0.15*nHetero
    return float(max(1.0, min(10.0, 1.0 + score)))

try:
    import sascorer
    def calc_sa(m: Chem.Mol) -> float:
        return float(sascorer.calculateScore(m))
    _USE_TRUE_SA = True
except Exception:
    def calc_sa(m: Chem.Mol) -> float:
        return calc_sa_proxy(m)

def mol_allowed_elements(m: Chem.Mol, allowed: Set[str]) -> bool:
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:  
            continue
        if a.GetSymbol() not in allowed:
            return False
    return True

def normalize_dummy_isotopes(m: Chem.Mol) -> Chem.Mol:
    em = Chem.RWMol(m)
    for a in em.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
    return em.GetMol()

def canonical_fragment_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(normalize_dummy_isotopes(m), canonical=True)

def fragment_ok(m: Chem.Mol, max_heavy: int, allowed: Set[str], forbid_O_polyvalent_anchor: bool = True) -> bool:
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False
    if not mol_allowed_elements(m, allowed):
        return False
    if m.GetNumHeavyAtoms() > max_heavy:
        return False
    dummies = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetDegree() != 1:
        return False
    if forbid_O_polyvalent_anchor:
        nb = list(dummies[0].GetNeighbors())[0]
        if nb.GetSymbol() == "O" and nb.GetDegree() >= 2:
            return False
    return True

def load_fragments(smiles_list: List[str], max_heavy: int) -> List[Chem.Mol]:
    out = []
    seen: Set[str] = set()
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if fragment_ok(m, max_heavy, ALLOWED_ELEMENTS):
            key = canonical_fragment_smiles(m)
            if key not in seen:
                seen.add(key)
                out.append(m)
    if not out:
        raise RuntimeError("Nessun frammento valido caricato dalla lista.")
    return out

def brics_monoanchor_variants(frag: Chem.Mol) -> List[Chem.Mol]:
    dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) <= 1:
        return [frag]
    out = []
    for keep in dummies:
        em = Chem.RWMol(frag)
        for di in sorted(dummies, reverse=True):
            if di == keep:
                continue
            try:
                em.RemoveAtom(di)
            except Exception:
                pass
        out.append(em.GetMol())
    return out

def build_brics_fragments(infile: str) -> List[Chem.Mol]:
    f = open(infile,"r")
    l = f.read().split("\n")[:-1]
    pool: List[Chem.Mol] = []
    for i in range(len(l)):
        pool.append(Chem.MolFromSmiles(l[i].split(",")[0]))
    return pool

CORE_FRAGMENTS: List[Chem.Mol] = []
DECOR_FRAGMENTS: List[Chem.Mol] = []
PROTECT_FRAGMENTS: List[Chem.Mol] = []  
POLAR_FRAGMENTS: List[Chem.Mol] = []       
LIPOPHILIC_FRAGMENTS: List[Chem.Mol] = []  

def init_fragment_libraries():
    global CORE_FRAGMENTS, DECOR_FRAGMENTS, PROTECT_FRAGMENTS
    global POLAR_FRAGMENTS, LIPOPHILIC_FRAGMENTS
    base = load_fragments(BASE_CORE_FRAGMENTS_SMILES, MAX_CORE_FRAG_HEAVY_ATOMS)
    brics_ext = build_brics_fragments(REFERENCE_SMILES)

    combined = []
    seen: Set[str] = set()
    for m in base + brics_ext:
        key = canonical_fragment_smiles(m)
        if key not in seen:
            seen.add(key)
            combined.append(m)
    CORE_FRAGMENTS = combined

    DECOR_FRAGMENTS = load_fragments(DECORATION_SMILES, MAX_DECOR_FRAG_HEAVY_ATOMS)
    PROTECT_FRAGMENTS = load_fragments(PROTECTING_GROUP_SMILES, MAX_PROTECT_FRAG_HEAVY_ATOMS)    
    POLAR_FRAGMENTS = load_fragments(POLAR_GROUP_SMILES, MAX_DECOR_FRAG_HEAVY_ATOMS)
    LIPOPHILIC_FRAGMENTS = load_fragments(LIPOPHILIC_GROUP_SMILES, MAX_DECOR_FRAG_HEAVY_ATOMS)

init_fragment_libraries()

@dataclass
class Action:
    op: str
    frag_idx: Optional[int] = None

@dataclass
class Program:
    actions: List[Action] = field(default_factory=list)
    def copy(self) -> "Program":
        return Program(actions=[Action(a.op, a.frag_idx) for a in self.actions])

def get_attachment_sites(mol: Chem.Mol) -> List[int]:
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() in ALLOWED_ELEMENTS and a.GetNumImplicitHs() > 0]

def get_aromatic_attachment_sites(mol: Chem.Mol) -> List[int]:
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and a.GetIsAromatic() and a.GetNumImplicitHs() > 0]

def get_hetero_nucleophile_sites(mol: Chem.Mol) -> List[int]:
    sites = []
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("N", "O") and a.GetNumImplicitHs() > 0:
            sites.append(a.GetIdx())
    return sites

def insert_ch2_linker(mol: Chem.Mol) -> Optional[Chem.Mol]:
    cand_bonds = []
    for b in mol.GetBonds():
        if b.IsInRing(): 
            continue
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a = b.GetBeginAtom()
        c = b.GetEndAtom()
        if a.GetAtomicNum() > 1 and c.GetAtomicNum() > 1:  
            cand_bonds.append((a.GetIdx(), c.GetIdx()))
    if not cand_bonds:
        return None

    a_idx, b_idx = random.choice(cand_bonds)
    rw = Chem.RWMol(mol)
    try:
        new_c = rw.AddAtom(Chem.Atom("C"))  
        rw.RemoveBond(a_idx, b_idx)
        rw.AddBond(a_idx, new_c, Chem.BondType.SINGLE)
        rw.AddBond(new_c, b_idx, Chem.BondType.SINGLE)
        new = rw.GetMol()
        Chem.SanitizeMol(new)
        return new
    except Exception:
        return None

def delete_ch2_linker(mol: Chem.Mol) -> Optional[Chem.Mol]:
    cands = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        if a.GetDegree() != 2:
            continue
        if a.IsInRing():
            continue
        nbs = [n.GetIdx() for n in a.GetNeighbors()]
        if len(nbs) != 2:
            continue
        b1 = mol.GetBondBetweenAtoms(a.GetIdx(), nbs[0])
        b2 = mol.GetBondBetweenAtoms(a.GetIdx(), nbs[1])
        if b1 is None or b2 is None:
            continue
        if b1.GetBondType() != Chem.BondType.SINGLE or b2.GetBondType() != Chem.BondType.SINGLE:
            continue
        if mol.GetAtomWithIdx(nbs[0]).GetAtomicNum() > 1 and mol.GetAtomWithIdx(nbs[1]).GetAtomicNum() > 1:
            cands.append((a.GetIdx(), nbs[0], nbs[1]))
    if not cands:
        return None

    c_idx, n1, n2 = random.choice(cands)
    rw = Chem.RWMol(mol)
    try:
        rw.AddBond(n1, n2, Chem.BondType.SINGLE)
        rw.RemoveAtom(c_idx)
        new = rw.GetMol()
        Chem.SanitizeMol(new)
        return new
    except Exception:
        return None

_HALOGENS_Z = [9, 17, 35, 53]
def swap_halogen(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    hal_idxs = [a.GetIdx() for a in mol.GetAtoms() 
                if a.GetAtomicNum() in _HALOGENS_Z and a.GetDegree() == 1]
    if not hal_idxs:
        return None
    for _ in range(max_retries):
        idx = random.choice(hal_idxs)
        curZ = mol.GetAtomWithIdx(idx).GetAtomicNum()
        choices = [z for z in _HALOGENS_Z if z != curZ]
        newZ = random.choice(choices)
        rw = Chem.RWMol(mol)
        try:
            rw.GetAtomWithIdx(idx).SetAtomicNum(newZ)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None


def find_dummy_idx(frag: Chem.Mol) -> Optional[int]:
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:
            return a.GetIdx()
    return None

def attach_fragment_at_atom(base: Chem.Mol, base_atom_idx: int, frag: Chem.Mol) -> Optional[Chem.Mol]:
    dummy_idx = find_dummy_idx(frag)
    if dummy_idx is None:
        return None
    dummy_atom = frag.GetAtomWithIdx(dummy_idx)
    if dummy_atom.GetDegree() != 1:
        return None
    frag_neighbor = list(dummy_atom.GetNeighbors())[0].GetIdx()

    em = Chem.RWMol(frag)
    try:
        em.RemoveAtom(dummy_idx)
    except Exception:
        return None
    frag_no_dummy = em.GetMol()

    combo = Chem.CombineMols(base, frag_no_dummy)
    rw = Chem.RWMol(combo)

    base_atoms = base.GetNumAtoms()
    frag_attach_idx = frag_neighbor if frag_neighbor < dummy_idx else frag_neighbor - 1
    frag_attach_idx_global = base_atoms + frag_attach_idx

    try:
        rw.AddBond(int(base_atom_idx), int(frag_attach_idx_global), Chem.BondType.SINGLE)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None

def _largest_heavy_fragment(m: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        return max(frags, key=lambda x: x.GetNumHeavyAtoms())
    except Exception:
        return None

def oxidize_alcohol(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    
    candidates: List[Tuple[int, int]] = []  
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        pairs = [(a1, a2), (a2, a1)]
        for C, O in pairs:
            if C.GetSymbol() != "C" or O.GetSymbol() != "O":
                continue
            if O.GetNumImplicitHs() <= 0:
                continue
            if C.GetHybridization() not in (Chem.HybridizationType.SP3,):
                continue
            if C.GetTotalNumHs() <= 0:
                continue
            candidates.append((C.GetIdx(), O.GetIdx()))
    if not candidates:
        return None

    attempts = 0
    while attempts < max_retries:
        attempts += 1
        c_idx, o_idx = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            if rw.GetBondBetweenAtoms(c_idx, o_idx) is None:
                continue
            rw.RemoveBond(c_idx, o_idx)
            rw.AddBond(c_idx, o_idx, Chem.BondType.DOUBLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def _is_simple_ketone_or_aldehyde(m: Chem.Mol, c_idx: int, o_idx: int) -> bool:
    
    c_at = m.GetAtomWithIdx(c_idx)
    o_at = m.GetAtomWithIdx(o_idx)
    if c_at.GetSymbol() != "C" or o_at.GetSymbol() != "O":
        return False
    b = m.GetBondBetweenAtoms(c_idx, o_idx)
    if b is None or b.GetBondType() != Chem.BondType.DOUBLE:
        return False
    for nb in c_at.GetNeighbors():
        if nb.GetIdx() == o_idx:
            continue
        if nb.GetSymbol() in ("O", "N"):
            b2 = m.GetBondBetweenAtoms(c_idx, nb.GetIdx())
            if b2 is not None and b2.GetBondType() == Chem.BondType.SINGLE:
                return False
    return True

def reduce_carbonyl(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
   
    candidates: List[Tuple[int, int]] = []  # (c_idx, o_idx)
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        pairs = [(a1, a2), (a2, a1)]
        for C, O in pairs:
            if _is_simple_ketone_or_aldehyde(mol, C.GetIdx(), O.GetIdx()):
                candidates.append((C.GetIdx(), O.GetIdx()))
    if not candidates:
        return None

    attempts = 0
    while attempts < max_retries:
        attempts += 1
        c_idx, o_idx = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            if rw.GetBondBetweenAtoms(c_idx, o_idx) is None:
                continue
            rw.RemoveBond(c_idx, o_idx)
            rw.AddBond(c_idx, o_idx, Chem.BondType.SINGLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception as e:
            continue
    return None

def _find_ester_sites(m: Chem.Mol) -> List[Tuple[int, int, int]]:
    
    sites: List[Tuple[int, int, int]] = []
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        for Oa, Ca in ((a1, a2), (a2, a1)):
            if Oa.GetSymbol() != "O" or Ca.GetSymbol() != "C":
                continue
            Ocarbonyl_idx = -1
            for nb in Ca.GetNeighbors():
                if nb.GetIdx() == Oa.GetIdx():
                    continue
                if nb.GetSymbol() == "O":
                    b2 = m.GetBondBetweenAtoms(Ca.GetIdx(), nb.GetIdx())
                    if b2 is not None and b2.GetBondType() == Chem.BondType.DOUBLE:
                        Ocarbonyl_idx = nb.GetIdx()
                        break
            if Ocarbonyl_idx < 0:
                continue
            o_nbs = [nb.GetIdx() for nb in Oa.GetNeighbors() if nb.GetIdx() != Ca.GetIdx()]
            if not o_nbs:
                continue
            if not any(m.GetAtomWithIdx(i).GetSymbol() == "C" for i in o_nbs):
                continue
            
            sites.append((Ca.GetIdx(), Oa.GetIdx(), Ocarbonyl_idx))
    return sites

def hydrolyze_ester(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:

    candidates = _find_ester_sites(mol)
    if not candidates:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        c_idx, oalk_idx, _ = random.choice(candidates)
        rw = Chem.RWMol(mol)
        if rw.GetBondBetweenAtoms(c_idx, oalk_idx) is None:
            continue
        try:
            rw.RemoveBond(c_idx, oalk_idx)
        except Exception:
            continue
        try:
            newO = rw.AddAtom(Chem.Atom("O"))
            rw.AddBond(c_idx, newO, Chem.BondType.SINGLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            new = _largest_heavy_fragment(new)
            return new
        except Exception:
            continue
    return None

def _find_acid_sites(m: Chem.Mol) -> List[Tuple[int, int]]:
    out = []
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        for Oa, Ca in ((a1, a2), (a2, a1)):
            if Oa.GetSymbol() != "O" or Oa.GetNumImplicitHs() <= 0:
                continue
            if Ca.GetSymbol() != "C":
                continue
            is_carbonyl = any(
                (nb.GetSymbol() == "O" and m.GetBondBetweenAtoms(Ca.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)
                for nb in Ca.GetNeighbors()
            )
            if not is_carbonyl:
                continue
            out.append((Oa.GetIdx(), Ca.GetIdx()))
    return out

def form_ester_intramolecular(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
   
    _EXT_ALCOHOL_SMILES = ["[*:1]OC", "[*:1]OCC", "[*:1]Oc1ccccc1"]
    _EXT_ALCOHOL_FRAGS = [Chem.MolFromSmiles(s) for s in _EXT_ALCOHOL_SMILES if Chem.MolFromSmiles(s) is not None]

    acid_sites = _find_acid_sites(mol)  
    if not acid_sites or not _EXT_ALCOHOL_FRAGS:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        Oacid, Cacyl = random.choice(acid_sites)
        rw = Chem.RWMol(mol)

        if rw.GetBondBetweenAtoms(Oacid, Cacyl) is None:
            continue
        try:
            rw.RemoveBond(Oacid, Cacyl)
            rw.RemoveAtom(Oacid)
        except Exception:
            continue

        def _adj(idx: int, removed: int) -> int:
            return idx - 1 if idx > removed else idx
        Cacyl2 = _adj(Cacyl, Oacid)

        try:
            intermediate = rw.GetMol()
            Chem.SanitizeMol(intermediate)
        except Exception:
            continue

        frag = random.choice(_EXT_ALCOHOL_FRAGS)
        new_mol = attach_fragment_at_atom(intermediate, Cacyl2, frag)
        if new_mol is not None:
            return new_mol

    return None



def form_amide_intramolecular(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    
    _EXT_AMINE_SMILES = ["[*:1]N", "[*:1]NC", "[*:1]NCC"]
    _EXT_AMINE_FRAGS = [Chem.MolFromSmiles(s) for s in _EXT_AMINE_SMILES if Chem.MolFromSmiles(s) is not None]

    acid_sites = _find_acid_sites(mol)  
    if not acid_sites or not _EXT_AMINE_FRAGS:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        Oacid, Cacyl = random.choice(acid_sites)
        rw = Chem.RWMol(mol)

        if rw.GetBondBetweenAtoms(Oacid, Cacyl) is None:
            continue
        try:
            rw.RemoveBond(Oacid, Cacyl)
            rw.RemoveAtom(Oacid)
        except Exception:
            continue
        def _adj(idx: int, removed: int) -> int:
            return idx - 1 if idx > removed else idx
        Cacyl2 = _adj(Cacyl, Oacid)

        try:
            intermediate = rw.GetMol()
            Chem.SanitizeMol(intermediate)
        except Exception:
            continue

        frag = random.choice(_EXT_AMINE_FRAGS)
        new_mol = attach_fragment_at_atom(intermediate, Cacyl2, frag)
        if new_mol is not None:
            return new_mol

    return None

def saturate_aromatic_ring(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
   
    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    if not atom_rings:
        return None

    def _ring_bonds_for_ring(ring_atoms: Tuple[int, ...]) -> List[Tuple[int, int]]:
        n = len(ring_atoms)
        return [(ring_atoms[i], ring_atoms[(i+1) % n]) for i in range(n)]

    candidates: List[Tuple[int, ...]] = []
    for r in atom_rings:
        ok = True
        for a, b in _ring_bonds_for_ring(r):
            bd = mol.GetBondBetweenAtoms(a, b)
            if bd is None or not bd.GetIsAromatic():
                ok = False
                break
        if ok:
            candidates.append(r)

    if not candidates:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        ring = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            for a in ring:
                rw.GetAtomWithIdx(a).SetIsAromatic(False)
            for i in range(len(ring)):
                a = ring[i]
                b = ring[(i + 1) % len(ring)]
                bd = rw.GetBondBetweenAtoms(a, b)
                if bd is None:
                    break
                bd.SetIsAromatic(False)
                bd.SetBondType(Chem.BondType.SINGLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def aromatize_ring(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    if not atom_rings:
        return None

    cand: List[Tuple[int, ...]] = []
    for r in atom_rings:
        if len(r) != 6:
            continue
        all_c = all(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in r)
        if not all_c:
            continue
        ok_single = True
        for i in range(6):
            a, b = r[i], r[(i+1) % 6]
            bd = mol.GetBondBetweenAtoms(a, b)
            if bd is None or bd.GetBondType() != Chem.BondType.SINGLE:
                ok_single = False
                break
        if ok_single:
            cand.append(r)

    if not cand:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        r = random.choice(cand)
        for parity in (0, 1):
            rw = Chem.RWMol(mol)
            try:
                for i in range(6):
                    a, b = r[i], r[(i+1) % 6]
                    bd = rw.GetBondBetweenAtoms(a, b)
                    if bd is None:
                        raise RuntimeError("bond missing")
                    if (i + parity) % 2 == 0:
                        bd.SetBondType(Chem.BondType.DOUBLE)
                    else:
                        bd.SetBondType(Chem.BondType.SINGLE)
                    bd.SetIsAromatic(False)
                new = rw.GetMol()
                Chem.SanitizeMol(new)
                return new
            except Exception:
                continue
    return None

def expand_ring(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    ring_bonds: List[Tuple[int, int]] = []
    for b in mol.GetBonds():
        if not b.IsInRing():
            continue
        a = b.GetBeginAtom()
        c = b.GetEndAtom()
        if a.GetAtomicNum() > 1 and c.GetAtomicNum() > 1:
            ring_bonds.append((a.GetIdx(), c.GetIdx()))
    if not ring_bonds:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        a_idx, b_idx = random.choice(ring_bonds)
        rw = Chem.RWMol(mol)
        try:
            if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
                continue
            rw.RemoveBond(a_idx, b_idx)
            new_c = rw.AddAtom(Chem.Atom("C"))
            rw.AddBond(a_idx, new_c, Chem.BondType.SINGLE)
            rw.AddBond(new_c, b_idx, Chem.BondType.SINGLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def contract_ring(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    ri = mol.GetRingInfo()
    rings = list(ri.AtomRings())
    if not rings:
        return None

    candidates: List[Tuple[int, int, int]] = []
    for r in rings:
        n = len(r)
        if n <= 5:
            continue  
        for i in range(n):
            center = r[i]
            left = r[(i - 1) % n]
            right = r[(i + 1) % n]
            at_center = mol.GetAtomWithIdx(center)
            if at_center.GetDegree() != 2:
                continue  
            if mol.GetBondBetweenAtoms(left, right) is not None:
                continue
            candidates.append((left, center, right))

    if not candidates:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        left, center, right = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            rw.AddBond(left, right, Chem.BondType.SINGLE)
            rw.RemoveAtom(center)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

from typing import Deque
from collections import deque

def _unique_ring_edges(m: Chem.Mol) -> List[Tuple[int,int]]:
    ri = m.GetRingInfo()
    rings = list(ri.AtomRings())
    seen = set()
    edges: List[Tuple[int,int]] = []
    for r in rings:
        n = len(r)
        for t in range(n):
            a = r[t]
            b = r[(t+1) % n]
            key = (a,b) if a < b else (b,a)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)
    return edges

def _bfs_nonring_singles(m: Chem.Mol, start: int, max_depth: int = 4) -> Dict[int,int]:
    out: Dict[int,int] = {}
    Q: Deque[Tuple[int,int]] = deque()
    Q.append((start, 0))
    visited: Set[int] = {start}
    while Q:
        v, d = Q.popleft()
        if d >= max_depth:
            continue
        at = m.GetAtomWithIdx(v)
        for nb in at.GetNeighbors():
            u = nb.GetIdx()
            if u in visited:
                continue
            if nb.GetAtomicNum() <= 1:
                continue
            if nb.IsInRing():
                continue
            b = m.GetBondBetweenAtoms(v, u)
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            visited.add(u)
            out[u] = d + 1
            Q.append((u, d + 1))
    return out

def fuse_rings(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    ri = mol.GetRingInfo()
    if not ri or not ri.AtomRings():
        return None

    edges = _unique_ring_edges(mol)
    if not edges:
        return None

    def edge_weight(a: int, b: int) -> int:
        bd = mol.GetBondBetweenAtoms(a,b)
        if bd is not None and bd.GetIsAromatic():
            return 2
        return 1

    candidates: List[Tuple[int,int]] = []  

    ring_atoms_set: Set[int] = set()
    for r in ri.AtomRings():
        ring_atoms_set.update(r)

    for (i,k) in edges:
        for s, other in ((i,k),(k,i)):
            s_atom = mol.GetAtomWithIdx(s)
            other_atom = mol.GetAtomWithIdx(other)
            if other_atom.GetNumImplicitHs() <= 0:
                continue
            for nb in s_atom.GetNeighbors():
                alpha = nb.GetIdx()
                if nb.IsInRing():
                    continue
                if mol.GetBondBetweenAtoms(s, alpha).GetBondType() != Chem.BondType.SINGLE:
                    continue
                if nb.GetFormalCharge() != 0:
                    continue
                depth_map = _bfs_nonring_singles(mol, alpha, max_depth=4)
                for j, d_alpha_j in depth_map.items():
                    if d_alpha_j not in (2,3,4):
                        continue
                    if j in ring_atoms_set:
                        continue
                    if mol.GetBondBetweenAtoms(other, j) is not None:
                        continue
                    jat = mol.GetAtomWithIdx(j)
                    if jat.GetAtomicNum() not in (6,7):  
                        continue
                    if jat.GetFormalCharge() != 0:
                        continue
                    if jat.GetNumImplicitHs() <= 0:
                        continue
                    if jat.GetDegree() >= 4:
                        continue
                    ring_size = d_alpha_j + 3  
                    if ring_size < 5 or ring_size > 7:
                        continue
                    w = edge_weight(i,k)
                    for _ in range(w):  
                        candidates.append((other, j))

    if not candidates:
        return None

    tries = 0
    while tries < max_retries and candidates:
        tries += 1
        other, j = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            if rw.GetBondBetweenAtoms(other, j) is not None:
                continue
            rw.AddBond(int(other), int(j), Chem.BondType.SINGLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            if not mol_allowed_elements(new, ALLOWED_ELEMENTS):
                continue
            if new.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
                continue
            return new
        except Exception:
            continue
    return None

_HALOGENS_Z = [9, 17, 35, 53]  

def _carbon_sites_with_H(mol: Chem.Mol, aromatic_only: bool = False) -> List[int]:
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        if aromatic_only and not a.GetIsAromatic():
            continue
        if a.GetNumImplicitHs() > 0:
            out.append(a.GetIdx())
    return out

def halogenate(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _carbon_sites_with_H(mol, aromatic_only=True)
    if not sites:
        sites = _carbon_sites_with_H(mol, aromatic_only=False)
    if not sites:
        return None

    for _ in range(max_retries):
        site = random.choice(sites)
        X = random.choice(["F", "Cl", "Br", "I"])
        rw = Chem.RWMol(mol)
        try:
            x_idx = rw.AddAtom(Chem.Atom(X))
            rw.AddBond(site, x_idx, Chem.BondType.SINGLE)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def dehalogenate(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    hal_idxs = [a.GetIdx() for a in mol.GetAtoms()
                if a.GetAtomicNum() in _HALOGENS_Z and a.GetDegree() == 1]
    if not hal_idxs:
        return None
    for _ in range(max_retries):
        idx = random.choice(hal_idxs)
        rw = Chem.RWMol(mol)
        try:
            rw.RemoveAtom(idx)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def nitrate(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _carbon_sites_with_H(mol, aromatic_only=True)
    if not sites:
        sites = _carbon_sites_with_H(mol, aromatic_only=False)
    if not sites:
        return None

    for _ in range(max_retries):
        c_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            n_idx  = rw.AddAtom(Chem.Atom("N"))
            o1_idx = rw.AddAtom(Chem.Atom("O"))
            o2_idx = rw.AddAtom(Chem.Atom("O"))

            rw.AddBond(int(c_idx), int(n_idx), Chem.BondType.SINGLE)
            rw.AddBond(int(n_idx), int(o1_idx), Chem.BondType.DOUBLE)
            rw.AddBond(int(n_idx), int(o2_idx), Chem.BondType.SINGLE)

            n_atom = rw.GetAtomWithIdx(int(n_idx))
            n_atom.SetFormalCharge(1)
            rw.GetAtomWithIdx(int(o2_idx)).SetFormalCharge(-1)

            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue

    return None

def _find_free_amine_sites_sulfonyl(m: Chem.Mol) -> List[int]:
    
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "N":
            continue
        if a.GetNumImplicitHs() > 0:
            out.append(a.GetIdx())
    return out

def sulfonylate(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    frag_smiles = ["[*:1]S(=O)(=O)C", "[*:1]S(=O)(=O)c1ccccc1"]
    frags = [Chem.MolFromSmiles(s) for s in frag_smiles if Chem.MolFromSmiles(s) is not None]
    if not frags:
        return None

    n_sites = _find_free_amine_sites_sulfonyl(mol)
    if n_sites:
        for _ in range(max_retries):
            n_site = random.choice(n_sites)
            frag = random.choice(frags)
            new_mol = attach_fragment_at_atom(mol, n_site, frag)
            if new_mol is not None:
                return new_mol

    sites = _carbon_sites_with_H(mol, aromatic_only=True)
    if not sites:
        sites = _carbon_sites_with_H(mol, aromatic_only=False)
    if not sites:
        return None

    for _ in range(max_retries):
        site = random.choice(sites)
        frag = random.choice(frags)
        new_mol = attach_fragment_at_atom(mol, site, frag)
        if new_mol is not None:
            return new_mol
    return None

def _find_phenol_sites(m: Chem.Mol) -> List[int]:
    out: List[int] = []
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        for Oa, Ca in ((a1, a2), (a2, a1)):
            if Oa.GetSymbol() != "O" or Oa.GetNumImplicitHs() <= 0:
                continue
            if Ca.GetSymbol() != "C" or not Ca.GetIsAromatic():
                continue
            out.append(Oa.GetIdx())
    return out


def _find_simple_acetyl_sites(m: Chem.Mol) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for c in m.GetAtoms():
        if c.GetSymbol() != "C":
            continue
        Cacyl = c.GetIdx()
        has_carbonyl_O = False
        Ocarbonyl_idx = -1
        for nb in c.GetNeighbors():
            if nb.GetSymbol() != "O":
                continue
            b = m.GetBondBetweenAtoms(Cacyl, nb.GetIdx())
            if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                has_carbonyl_O = True
                Ocarbonyl_idx = nb.GetIdx()
                break
        if not has_carbonyl_O:
            continue

        het_candidates: List[int] = []
        methyl_candidates: List[int] = []
        for nb in c.GetNeighbors():
            if nb.GetIdx() == Ocarbonyl_idx:
                continue
            b = m.GetBondBetweenAtoms(Cacyl, nb.GetIdx())
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if nb.GetSymbol() in ("O", "N"):
                het_candidates.append(nb.GetIdx())
            elif nb.GetSymbol() == "C":
                if nb.GetDegree() == 1 and not nb.IsInRing():
                    methyl_candidates.append(nb.GetIdx())

        if het_candidates and methyl_candidates:
            out.append((het_candidates[0], Cacyl))
    return out

def methylate_amine(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    def _is_amide_like_nitrogen(m: Chem.Mol, n_idx: int) -> bool:
        n = m.GetAtomWithIdx(n_idx)
        for c in n.GetNeighbors():
            if c.GetSymbol() == "C":
                for nb in c.GetNeighbors():
                    if nb.GetIdx() == n_idx:
                        continue
                    if nb.GetSymbol() == "O":
                        b = m.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
                        if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                            return True
            if c.GetSymbol() == "S":
                o_doubles = 0
                for nb in c.GetNeighbors():
                    if nb.GetSymbol() == "O":
                        b = m.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
                        if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                            o_doubles += 1
                if o_doubles >= 2:
                    return True
        return False

    sites_H = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() == "N"
               and a.GetNumImplicitHs() > 0
               and a.GetFormalCharge() == 0
               and not _is_amide_like_nitrogen(mol, a.GetIdx())]

    sites_tert = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetSymbol() == "N"
                  and a.GetNumImplicitHs() == 0
                  and a.GetFormalCharge() == 0
                  and a.GetDegree() == 3
                  and not a.GetIsAromatic()
                  and not _is_amide_like_nitrogen(mol, a.GetIdx())]

    frag = Chem.MolFromSmiles("[*:1]C")
    if frag is None:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        choose_tert = (not sites_H) and bool(sites_tert) or (random.random() < 0.30 and bool(sites_tert))
        if choose_tert:
            n_site = random.choice(sites_tert)
            rw = Chem.RWMol(mol)
            try:
                c_idx = rw.AddAtom(Chem.Atom("C"))
                rw.AddBond(int(n_site), int(c_idx), Chem.BondType.SINGLE)
                n_atom = rw.GetAtomWithIdx(int(n_site))
                n_atom.SetFormalCharge(1)   
                new = rw.GetMol()
                Chem.SanitizeMol(new)
                return new
            except Exception:
                continue
        else:
            if not sites_H:
                return None
            n_site = random.choice(sites_H)
            new_mol = attach_fragment_at_atom(mol, n_site, frag)  
            if new_mol is not None:
                return new_mol
    return None



def demethylate_amine(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    candidates: List[Tuple[int, int, int]] = []  

    for n in mol.GetAtoms():
        if n.GetSymbol() != "N":
            continue
        Nidx = n.GetIdx()
        Nq = n.GetFormalCharge()  
        for nb in n.GetNeighbors():
            if nb.GetSymbol() != "C":
                continue
            if nb.GetDegree() == 1 and not nb.IsInRing():
                b = mol.GetBondBetweenAtoms(Nidx, nb.GetIdx())
                if b is not None and b.GetBondType() == Chem.BondType.SINGLE:
                    candidates.append((Nidx, nb.GetIdx(), Nq))

    if not candidates:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        Nidx, cme, Nq = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            rw.RemoveAtom(int(cme))
            Nidx2 = Nidx - 1 if Nidx > cme else Nidx
            n_atom = rw.GetAtomWithIdx(int(Nidx2))
            if Nq > 0:
                n_atom.SetFormalCharge(0)
            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None



def acetylate_phenol(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_phenol_sites(mol)
    if not sites:
        return None
    frag = Chem.MolFromSmiles("[*:1]C(=O)C") 
    if frag is None:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        o_site = random.choice(sites)
        new_mol = attach_fragment_at_atom(mol, o_site, frag)
        if new_mol is not None:
            return new_mol
    return None


def deacetylate(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_simple_acetyl_sites(mol)
    if not sites:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        het_idx, cacyl_idx = random.choice(sites)

        side_atoms = _bfs_side_atoms(mol, base_idx=het_idx, side_root_idx=cacyl_idx)
        if not side_atoms:
            continue

        heavy = [i for i in side_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if len(heavy) != 3:
            continue

        new_mol = delete_atoms_set(mol, side_atoms)
        if new_mol is not None:
            return new_mol
    return None

def _find_ring_o_sites(m: Chem.Mol) -> List[int]:
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "O": 
            continue
        if not a.IsInRing(): 
            continue
        if a.GetFormalCharge() != 0: 
            continue
        if a.GetDegree() != 2: 
            continue
        out.append(a.GetIdx())
    return out

def _find_ring_nh_sites(m: Chem.Mol) -> List[int]:
    out: List[int] = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "N":
            continue
        if not a.IsInRing():
            continue
        if a.GetFormalCharge() != 0:
            continue
        if a.GetTotalNumHs() <= 0:
            continue
        out.append(a.GetIdx())
    return out


def ring_nh_to_o(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_ring_nh_sites(mol)
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        n_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            n_atom = rw.GetAtomWithIdx(n_idx)
            h_nbs = [nb.GetIdx() for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() == 1]
            for h in sorted(h_nbs, reverse=True):
                rw.RemoveAtom(h)
                if n_idx > h:
                    n_idx -= 1
            n_atom = rw.GetAtomWithIdx(n_idx)
            n_atom.SetAtomicNum(8)   # O
            n_atom.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def ring_o_to_nh(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_ring_o_sites(mol)
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        o_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(o_idx)
            at.SetAtomicNum(7)     
            at.SetFormalCharge(0)   
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def _find_linker_sites_by_symbol(m: Chem.Mol, symbol: str) -> List[int]:
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != symbol:
            continue
        if a.IsInRing():
            continue
        if a.GetDegree() != 2:
            continue
        ok = True
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum() <= 1:
                ok = False; break
            b = m.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                ok = False; break
        if not ok:
            continue
        if symbol == "C":
            if a.GetHybridization() not in (Chem.HybridizationType.SP3,):
                continue
            if a.GetTotalNumHs() < 2:
                continue
        if symbol == "N":
            if a.GetFormalCharge() != 0:
                continue
            if a.GetTotalNumHs() < 1:
                continue
        out.append(a.GetIdx())
    return out

def _linker_replace_center(mol: Chem.Mol, center_idx: int, new_symbol: str) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    try:
        at = rw.GetAtomWithIdx(center_idx)
        at.SetFormalCharge(0)
        if new_symbol == "C":
            at.SetAtomicNum(6)
        elif new_symbol == "N":
            at.SetAtomicNum(7)
        elif new_symbol == "O":
            at.SetAtomicNum(8)
        elif new_symbol == "S":
            at.SetAtomicNum(16)
        else:
            return None
        new = rw.GetMol(); Chem.SanitizeMol(new)
        return new
    except Exception:
        return None

def swap_linker(mol: Chem.Mol, from_symbol: str, to_symbol: str, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_linker_sites_by_symbol(mol, from_symbol)
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        idx = random.choice(sites)
        new = _linker_replace_center(mol, idx, to_symbol)
        if new is not None:
            return new
    return None

def hydrazide_to_ester(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites: List[Tuple[int,int,int,int]] = []  # (Cacyl, N1, N2, Rprime)
    for c in mol.GetAtoms():
        if c.GetSymbol() != "C":
            continue
        Cacyl = c.GetIdx()
        ocarb = -1
        for nb in c.GetNeighbors():
            if nb.GetSymbol() == "O":
                b = mol.GetBondBetweenAtoms(Cacyl, nb.GetIdx())
                if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                    ocarb = nb.GetIdx(); break
        if ocarb < 0:
            continue
        for nb in c.GetNeighbors():
            if nb.GetIdx() == ocarb:
                continue
            if nb.GetSymbol() != "N":
                continue
            b = mol.GetBondBetweenAtoms(Cacyl, nb.GetIdx())
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            N1 = nb
            N2 = None
            for nb2 in N1.GetNeighbors():
                if nb2.GetIdx() == Cacyl:
                    continue
                if nb2.GetSymbol() == "N":
                    N2 = nb2; break
            if N2 is None:
                continue
            if N1.GetDegree() != 2 or N2.GetDegree() != 2:
                continue
            Rprime = None
            for nb3 in N2.GetNeighbors():
                if nb3.GetIdx() == N1.GetIdx():
                    continue
                if nb3.GetAtomicNum() > 1:
                    Rprime = nb3; break
            if Rprime is None:
                continue
            sites.append((Cacyl, N1.GetIdx(), N2.GetIdx(), Rprime.GetIdx()))
    if not sites:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        Cacyl, N1, N2, Rprime = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            if rw.GetBondBetweenAtoms(Cacyl, N1) is not None:
                rw.RemoveBond(Cacyl, N1)
            if rw.GetBondBetweenAtoms(N1, N2) is not None:
                rw.RemoveBond(N1, N2)
            if rw.GetBondBetweenAtoms(N2, Rprime) is not None:
                rw.RemoveBond(N2, Rprime)
            for rm in sorted([N1, N2], reverse=True):
                rw.RemoveAtom(rm)
                def _adj(i: int, rmv: int) -> int: return i - 1 if i > rmv else i
                Cacyl = _adj(Cacyl, rm)
                Rprime = _adj(Rprime, rm)
            oidx = rw.AddAtom(Chem.Atom("O"))
            rw.AddBond(Cacyl, oidx, Chem.BondType.SINGLE)
            rw.AddBond(oidx, Rprime, Chem.BondType.SINGLE)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def add_f_to_c_h(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() == "C" and a.GetNumImplicitHs() > 0]
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        c_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            f_idx = rw.AddAtom(Chem.Atom("F"))
            rw.AddBond(c_idx, f_idx, Chem.BondType.SINGLE)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def remove_f_terminal(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() == "F" and a.GetDegree() == 1]
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        f_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            rw.RemoveAtom(f_idx)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def carbonyl_o_to_s(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites: List[int] = []  
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if (a.GetSymbol() == "C" and c.GetSymbol() == "O"):
            sites.append(c.GetIdx())
        elif (a.GetSymbol() == "O" and c.GetSymbol() == "C"):
            sites.append(a.GetIdx())
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        o_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(o_idx)
            at.SetAtomicNum(16)  # S
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def thiocarbonyl_s_to_o(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites: List[int] = []  
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if (a.GetSymbol() == "C" and c.GetSymbol() == "S"):
            sites.append(c.GetIdx())
        elif (a.GetSymbol() == "S" and c.GetSymbol() == "C"):
            sites.append(a.GetIdx())
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        s_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(s_idx)
            at.SetAtomicNum(8)   # O
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def swap_heavy_halogen(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    heavy = {"Cl": 17, "Br": 35, "I": 53}
    sites = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() in heavy and a.GetDegree() == 1]
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        idx = random.choice(sites)
        cur = mol.GetAtomWithIdx(idx).GetSymbol()
        choices = [z for z in ["Cl","Br","I"] if z != cur]
        newZ = heavy[random.choice(choices)]
        rw = Chem.RWMol(mol)
        try:
            rw.GetAtomWithIdx(idx).SetAtomicNum(newZ)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def _find_oh_sites_general(m: Chem.Mol) -> List[int]:
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "O":
            continue
        if a.GetFormalCharge() != 0:
            continue
        if a.GetNumImplicitHs() <= 0:
            continue
        if not any(nb.GetAtomicNum() > 1 for nb in a.GetNeighbors()):
            continue
        out.append(a.GetIdx())
    return out

def _find_nh2_sites_general(m: Chem.Mol) -> List[int]:
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "N":
            continue
        if a.GetFormalCharge() != 0:
            continue
        if a.GetNumImplicitHs() < 2:
            continue
        if not any(nb.GetAtomicNum() > 1 for nb in a.GetNeighbors()):
            continue
        out.append(a.GetIdx())
    return out

def oh_to_nh2(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_oh_sites_general(mol)
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        o_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(o_idx)
            at.SetAtomicNum(7)     # N
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def nh2_to_oh(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_nh2_sites_general(mol)
    if not sites:
        return None
    tries = 0
    while tries < max_retries:
        tries += 1
        n_idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(n_idx)
            at.SetAtomicNum(8)     # O
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def _find_terminal_ch3_sites(m: Chem.Mol) -> List[int]:
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        if a.IsInRing():
            continue
        if a.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        if a.GetDegree() != 1:
            continue
        if a.GetTotalNumHs() < 3:
            continue
        out.append(a.GetIdx())
    return out

def ch3_to_nh2(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_terminal_ch3_sites(mol)
    if not sites:
        return None
    for _ in range(max_retries):
        idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(idx)
            at.SetAtomicNum(7)  
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def ch3_to_oh(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = _find_terminal_ch3_sites(mol)
    if not sites:
        return None
    for _ in range(max_retries):
        idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(idx)
            at.SetAtomicNum(8)  
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def nh2_to_ch3(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = [i for i in _find_nh2_sites_general(mol)
             if mol.GetAtomWithIdx(i).GetDegree() == 1 and not mol.GetAtomWithIdx(i).IsInRing()]
    if not sites:
        return None
    for _ in range(max_retries):
        idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(idx)
            at.SetAtomicNum(6)  
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None

def oh_to_ch3(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    sites = [i for i in _find_oh_sites_general(mol)
             if mol.GetAtomWithIdx(i).GetDegree() == 1 and not mol.GetAtomWithIdx(i).IsInRing()]
    if not sites:
        return None
    for _ in range(max_retries):
        idx = random.choice(sites)
        rw = Chem.RWMol(mol)
        try:
            at = rw.GetAtomWithIdx(idx)
            at.SetAtomicNum(6)  
            at.SetFormalCharge(0)
            new = rw.GetMol(); Chem.SanitizeMol(new)
            return new
        except Exception:
            continue
    return None


def replace_coo_with_conhnh(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    candidates = _find_ester_sites(mol)
    if not candidates:
        return None

    tries = 0
    while tries < max_retries:
        tries += 1
        Cacyl, Oalk, _Ocarb = random.choice(candidates)

        Oa = mol.GetAtomWithIdx(Oalk)
        r_neighbors = [nb.GetIdx() for nb in Oa.GetNeighbors() if nb.GetIdx() != Cacyl]
        if not r_neighbors:
            continue
        Rprime = r_neighbors[0]

        rw = Chem.RWMol(mol)
        try:
            if rw.GetBondBetweenAtoms(Cacyl, Oalk) is not None:
                rw.RemoveBond(Cacyl, Oalk)

            rw.RemoveAtom(Oalk)

            def _adj(i: int, removed: int) -> int:
                return i - 1 if i > removed else i
            Cacyl2 = _adj(Cacyl, Oalk)
            Rprime2 = _adj(Rprime, Oalk)

            n1 = rw.AddAtom(Chem.Atom("N"))
            n2 = rw.AddAtom(Chem.Atom("N"))
            rw.AddBond(Cacyl2, n1, Chem.BondType.SINGLE)  
            rw.AddBond(n1, n2, Chem.BondType.SINGLE)      
            rw.AddBond(n2, Rprime2, Chem.BondType.SINGLE) 

            new = rw.GetMol()
            Chem.SanitizeMol(new)
            return new
        except Exception:
            continue

    return None

def bioisosteric_swap(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    candidates = [
        lambda m: ring_nh_to_o(m, max_retries),
        lambda m: ring_o_to_nh(m, max_retries),

        lambda m: swap_linker(m, "C", "N", max_retries),
        lambda m: swap_linker(m, "N", "C", max_retries),
        lambda m: swap_linker(m, "C", "O", max_retries),
        lambda m: swap_linker(m, "O", "C", max_retries),
        lambda m: swap_linker(m, "C", "S", max_retries),
        lambda m: swap_linker(m, "S", "C", max_retries),
        lambda m: swap_linker(m, "N", "O", max_retries),
        lambda m: swap_linker(m, "O", "N", max_retries),
        lambda m: swap_linker(m, "N", "S", max_retries),
        lambda m: swap_linker(m, "S", "N", max_retries),
        lambda m: swap_linker(m, "O", "S", max_retries),
        lambda m: swap_linker(m, "S", "O", max_retries),

        lambda m: replace_coo_with_conhnh(m, max_retries),
        lambda m: hydrazide_to_ester(m, max_retries),

        lambda m: add_f_to_c_h(m, max_retries),
        lambda m: remove_f_terminal(m, max_retries),

        lambda m: carbonyl_o_to_s(m, max_retries),
        lambda m: thiocarbonyl_s_to_o(m, max_retries),

        lambda m: swap_heavy_halogen(m, max_retries),

        lambda m: oh_to_nh2(m, max_retries),
        lambda m: nh2_to_oh(m, max_retries),

        lambda m: ch3_to_nh2(m, max_retries),
        lambda m: ch3_to_oh(m, max_retries),
        lambda m: nh2_to_ch3(m, max_retries),
        lambda m: oh_to_ch3(m, max_retries),
    ]

    random.shuffle(candidates)

    tried = 0
    for fn in candidates:
        if tried >= max_retries:
            break
        tried += 1
        new_mol = fn(mol)
        if new_mol is None:
            continue
        if not mol_allowed_elements(new_mol, ALLOWED_ELEMENTS):
            continue
        if new_mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            continue
        return new_mol
    return None

def _side_is_polar(m: Chem.Mol, side_atoms: Set[int], max_heavy: int = 6) -> bool:
    heavy = [i for i in side_atoms if m.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if not heavy or len(heavy) > max_heavy:
        return False
    return any(m.GetAtomWithIdx(i).GetSymbol() in ("O", "N") for i in heavy)

def _side_is_lipophilic(m: Chem.Mol, side_atoms: Set[int], max_heavy: int = 10) -> bool:
    heavy = [i for i in side_atoms if m.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if not heavy or len(heavy) > max_heavy:
        return False
    only_c = all(m.GetAtomWithIdx(i).GetSymbol() == "C" for i in heavy)
    if not only_c:
        return False
    any_arom_c = any(m.GetAtomWithIdx(i).GetSymbol() == "C" and m.GetAtomWithIdx(i).GetIsAromatic() for i in heavy)
    return True if only_c else False

def add_polar_group(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    if not POLAR_FRAGMENTS:
        return None

    sites = [i for i in get_aromatic_attachment_sites(mol)]
    if not sites:
        sites = [i for i in get_attachment_sites(mol) if mol.GetAtomWithIdx(i).GetSymbol() == "C"]
    if not sites:
        sites = get_attachment_sites(mol)
    if not sites:
        return None

    for _ in range(max_retries):
        site = random.choice(sites)
        frag = random.choice(POLAR_FRAGMENTS)
        new_mol = attach_fragment_at_atom(mol, site, frag)  
        if new_mol is not None:
            return new_mol
    return None


def add_lipophilic_group(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    if not LIPOPHILIC_FRAGMENTS:
        return None

    sites = [i for i in get_aromatic_attachment_sites(mol)]
    if not sites:
        sites = [i for i in get_attachment_sites(mol) if mol.GetAtomWithIdx(i).GetSymbol() == "C"]
    if not sites:
        sites = get_attachment_sites(mol)
    if not sites:
        return None

    for _ in range(max_retries):
        site = random.choice(sites)
        frag = random.choice(LIPOPHILIC_FRAGMENTS)
        new_mol = attach_fragment_at_atom(mol, site, frag)  
        if new_mol is not None:
            return new_mol
    return None


def swap_polar_lipophilic(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    candidates = cuttable_bonds(mol, MAX_DECOR_DELETE_SIZE, aromatic_base_only=True)
    if not candidates:
        candidates = cuttable_bonds(mol, MAX_GENERAL_DELETE_SIZE, aromatic_base_only=False)
    if not candidates:
        return None

    for _ in range(max_retries):
        base_idx, side_root, side_atoms = random.choice(candidates)

        is_pol = _side_is_polar(mol, side_atoms)
        is_lip = _side_is_lipophilic(mol, side_atoms)

        if not is_pol and not is_lip:
            continue  

        target_lib = LIPOPHILIC_FRAGMENTS if is_pol else POLAR_FRAGMENTS
        if not target_lib:
            continue

        pruned = delete_atoms_set(mol, side_atoms)  
        if pruned is None:
            continue

        frag = random.choice(target_lib)
        new_mol = attach_fragment_at_atom(pruned, base_idx, frag)  
        if new_mol is not None:
            return new_mol

    return None

def cyclize_chain(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
    MIN_RING = 5
    MAX_RING = 8
    ALLOWED_END_SYMBOLS = {"C", "N"}  
    atoms = mol.GetAtoms()
    end_idxs = []
    for a in atoms:
        if a.GetAtomicNum() <= 1:
            continue
        if a.IsInRing():
            continue
        if a.GetSymbol() not in ALLOWED_END_SYMBOLS:
            continue
        if a.GetFormalCharge() != 0:
            continue
        if a.GetNumImplicitHs() <= 0:  
            continue
        if a.GetDegree() >= 4:  
            continue
        if a.GetSymbol() == "N":
            amide_like = False
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "C":
                    for nb2 in nb.GetNeighbors():
                        if nb2.GetIdx() == a.GetIdx():
                            continue
                        if nb2.GetSymbol() == "O":
                            b = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                            if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                                amide_like = True
                                break
                    if amide_like:
                        break
                if nb.GetSymbol() == "S":
                    o_doubles = 0
                    for nb2 in nb.GetNeighbors():
                        if nb2.GetSymbol() == "O":
                            b = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                            if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                                o_doubles += 1
                    if o_doubles >= 2:
                        amide_like = True
                        break
            if amide_like:
                continue
        end_idxs.append(a.GetIdx())

    if len(end_idxs) < 2:
        return None

    candidates: List[Tuple[int, int]] = []
    end_set = set(end_idxs)
    ring_atoms_set = set()
    for r in mol.GetRingInfo().AtomRings():
        ring_atoms_set.update(r)

    end_list = list(end_set)
    n = len(end_list)
    for ii in range(n):
        i = end_list[ii]
        for jj in range(ii + 1, n):
            j = end_list[jj]
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue

            path = Chem.GetShortestPath(mol, i, j)
            if not path:
                continue

            ring_size = len(path)  
            if ring_size < MIN_RING or ring_size > MAX_RING:
                continue

            internal = path[1:-1]
            if any(k in ring_atoms_set for k in internal):
                continue

            single_ok = True
            for k in range(len(path) - 1):
                bd = mol.GetBondBetweenAtoms(path[k], path[k + 1])
                if bd is None or bd.GetBondType() != Chem.BondType.SINGLE:
                    single_ok = False
                    break
            if not single_ok:
                continue

            candidates.append((i, j))

    if not candidates:
        return None

    tries = 0
    while tries < max_retries and candidates:
        tries += 1
        a_idx, b_idx = random.choice(candidates)
        rw = Chem.RWMol(mol)
        try:
            rw.AddBond(int(a_idx), int(b_idx), Chem.BondType.SINGLE)
            new_m = rw.GetMol()
            Chem.SanitizeMol(new_m)
            if not mol_allowed_elements(new_m, ALLOWED_ELEMENTS):
                continue
            if new_m.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
                continue
            return new_m
        except Exception:
            try:
                candidates.remove((a_idx, b_idx))
            except Exception:
                pass
            continue

    return None

def reduce_chirality_aggressive(mol: Chem.Mol, max_retries: int = MAX_ACTION_RETRIES) -> Optional[Chem.Mol]:
   
    def _count_chiral(m: Chem.Mol) -> int:
        try:
            return len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
        except Exception:
            return 0

    def _chiral_centers(m: Chem.Mol) -> List[int]:
        try:
            return [i for (i, lab) in Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)]
        except Exception:
            return []

    def _try_fallbacks(m: Chem.Mol, before_count: int, local_retries: int = 3) -> Optional[Chem.Mol]:
        for op in (contract_ring, aromatize_ring, saturate_aromatic_ring):
            try:
                new = op(m, max_retries=local_retries)
            except Exception:
                new = None
            if new is None:
                continue
            if not mol_allowed_elements(new, ALLOWED_ELEMENTS):
                continue
            if new.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
                continue
            if _count_chiral(new) < before_count:
                return new
        return None

    before = _count_chiral(mol)
    if before == 0:
        return None

    max_inring_heavy = globals().get("CHIRAL_INRING_MAX_HEAVY", 3)
    min_ring_c_eq_c = globals().get("MIN_RING_SIZE_FOR_INRING_CEQC", 6)

    centers = _chiral_centers(mol)
    tries = 0

    while tries < max_retries and centers:
        tries += 1
        c_idx = random.choice(centers)
        at = mol.GetAtomWithIdx(c_idx)

        if at.GetTotalNumHs() > 0:
            candidate_sides = []
            for nb in at.GetNeighbors():
                if nb.GetAtomicNum() <= 1:
                    continue
                nb_idx = nb.GetIdx()
                b = mol.GetBondBetweenAtoms(c_idx, nb_idx)
                if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                    continue
                side_atoms = _bfs_side_atoms(mol, base_idx=c_idx, side_root_idx=nb_idx)
                heavy = [i for i in side_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]

                if not heavy:
                    continue

                if b.IsInRing():
                    if len(heavy) > max_inring_heavy:
                        continue
                else:
                    if len(heavy) > min(MAX_GENERAL_DELETE_SIZE, 8):
                        continue

                candidate_sides.append(side_atoms)

            random.shuffle(candidate_sides)
            for side in candidate_sides:
                new_m = delete_atoms_set(mol, side)
                if new_m is None:
                    continue
                if _count_chiral(new_m) < before:
                    return new_m

        for nb in at.GetNeighbors():
            if nb.GetSymbol() == "O" and nb.GetNumImplicitHs() > 0:
                b = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
                if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                    continue
                rw = Chem.RWMol(mol)
                try:
                    rw.GetBondBetweenAtoms(c_idx, nb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                    new = rw.GetMol(); Chem.SanitizeMol(new)
                    if _count_chiral(new) < before:
                        return new
                except Exception:
                    pass

        for nb in at.GetNeighbors():
            if nb.GetSymbol() == "N" and nb.GetNumImplicitHs() > 0 and nb.GetFormalCharge() == 0:
                b = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
                if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                    continue
                rw = Chem.RWMol(mol)
                try:
                    rw.GetBondBetweenAtoms(c_idx, nb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                    new = rw.GetMol(); Chem.SanitizeMol(new)
                    if _count_chiral(new) < before:
                        return new
                except Exception:
                    pass

        for nb in at.GetNeighbors():
            if nb.GetSymbol() != "C":
                continue
            b = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if b.IsInRing():
                continue
            if at.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if nb.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if at.GetTotalNumHs() <= 0 and nb.GetTotalNumHs() <= 0:
                continue
            rw = Chem.RWMol(mol)
            try:
                rw.GetBondBetweenAtoms(c_idx, nb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                new = rw.GetMol(); Chem.SanitizeMol(new)
                if _count_chiral(new) < before:
                    return new
            except Exception:
                pass

        for nb in at.GetNeighbors():
            if nb.GetSymbol() != "C":
                continue
            b = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if not b.IsInRing():
                continue
            if b.GetIsAromatic():
                continue
            if at.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if nb.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            ok_ring = False
            ri = mol.GetRingInfo()
            for ring in ri.BondRings():
                if b.GetIdx() in ring and len(ring) >= min_ring_c_eq_c:
                    ok_ring = True
                    break
            if not ok_ring:
                continue
            if at.GetDegree() > 3 or nb.GetDegree() > 3:
                continue
            rw = Chem.RWMol(mol)
            try:
                rw.GetBondBetweenAtoms(c_idx, nb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                new = rw.GetMol(); Chem.SanitizeMol(new)
                if _count_chiral(new) < before:
                    return new
            except Exception:
                pass

        if at.GetTotalNumHs() == 0:
            out_of_ring_nbs = [nb for nb in at.GetNeighbors()
                               if nb.GetAtomicNum() > 1
                               and (mol.GetBondBetweenAtoms(c_idx, nb.GetIdx()).IsInRing() is False)]
            random.shuffle(out_of_ring_nbs)
            for nb in out_of_ring_nbs:
                side = _bfs_side_atoms(mol, base_idx=c_idx, side_root_idx=nb.GetIdx())
                heavy = [i for i in side if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                if not heavy or len(heavy) > min(6,
                                                 MAX_GENERAL_DELETE_SIZE):
                    continue
                pruned = delete_atoms_set(mol, side)
                if pruned is None:
                    continue
                new = reduce_chirality_aggressive(pruned, max_retries=2)
                if new is not None and _count_chiral(new) < before:
                    return new

        centers = _chiral_centers(mol)

    fb = _try_fallbacks(mol, before_count=before, local_retries=3)
    if fb is not None:
        return fb

    return None


def _bfs_side_atoms(mol: Chem.Mol, base_idx: int, side_root_idx: int) -> Set[int]:
    visited: Set[int] = {base_idx}
    stack = [side_root_idx]
    side_atoms: Set[int] = set()
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        side_atoms.add(v)
        atom = mol.GetAtomWithIdx(v)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in visited:
                continue
            stack.append(nb_idx)
    return side_atoms

def cuttable_bonds(mol: Chem.Mol, max_side_heavy: int, aromatic_base_only: bool = False) -> List[Tuple[int, int, Set[int]]]:
    out = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a = bond.GetBeginAtom().GetIdx()
        b = bond.GetEndAtom().GetIdx()
        for base_idx, side_idx in ((a, b), (b, a)):
            base_atom = mol.GetAtomWithIdx(base_idx)
            if aromatic_base_only:
                if base_atom.GetSymbol() != "C" or not base_atom.GetIsAromatic():
                    continue
            side_atoms = _bfs_side_atoms(mol, base_idx, side_idx)
            heavy = sum(1 for i in side_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
            total_heavy = mol.GetNumHeavyAtoms()
            if heavy == 0 or heavy >= total_heavy:
                continue
            if heavy <= max_side_heavy:
                out.append((base_idx, side_idx, side_atoms))
    return out

def delete_atoms_set(mol: Chem.Mol, atoms_to_delete: Set[int]) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    for idx in sorted(atoms_to_delete, reverse=True):
        try:
            rw.RemoveAtom(idx)
        except Exception:
            return None
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None


def apply_action(mol: Chem.Mol, action: Action, max_retries: int = MAX_ACTION_RETRIES) -> Chem.Mol:
    op = action.op

    if op == "ADD_FRAG":
        frag = CORE_FRAGMENTS[action.frag_idx]
        for _ in range(max_retries):
            sites = get_attachment_sites(mol)
            if not sites:
                break
            site = random.choice(sites)
            new_mol = attach_fragment_at_atom(mol, site, frag)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "DECORATE_AROM":
        frag = DECOR_FRAGMENTS[action.frag_idx]
        for _ in range(max_retries):
            sites = get_aromatic_attachment_sites(mol)
            if not sites:
                break
            site = random.choice(sites)
            new_mol = attach_fragment_at_atom(mol, site, frag)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "DELETE_SIDECHAIN":
        for _ in range(max_retries):
            cands = cuttable_bonds(mol, MAX_GENERAL_DELETE_SIZE, aromatic_base_only=False)
            if not cands:
                break
            base_idx, side_root, side_atoms = random.choice(cands)
            new_mol = delete_atoms_set(mol, side_atoms)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "REPLACE_SIDECHAIN_CORE":
        frag = CORE_FRAGMENTS[action.frag_idx]
        for _ in range(max_retries):
            cands = cuttable_bonds(mol, MAX_GENERAL_DELETE_SIZE, aromatic_base_only=False)
            if not cands:
                break
            base_idx, side_root, side_atoms = random.choice(cands)
            pruned = delete_atoms_set(mol, side_atoms)
            if pruned is None:
                continue
            new_mol = attach_fragment_at_atom(pruned, base_idx, frag)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "DELETE_DECOR_AROM":
        for _ in range(max_retries):
            cands = cuttable_bonds(mol, MAX_DECOR_DELETE_SIZE, aromatic_base_only=True)
            if not cands:
                break
            base_idx, side_root, side_atoms = random.choice(cands)
            new_mol = delete_atoms_set(mol, side_atoms)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "REPLACE_DECOR_AROM":
        frag = DECOR_FRAGMENTS[action.frag_idx]
        for _ in range(max_retries):
            cands = cuttable_bonds(mol, MAX_DECOR_DELETE_SIZE, aromatic_base_only=True)
            if not cands:
                break
            base_idx, side_root, side_atoms = random.choice(cands)
            pruned = delete_atoms_set(mol, side_atoms)
            if pruned is None:
                continue
            new_mol = attach_fragment_at_atom(pruned, base_idx, frag)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "INSERT_CH2_LINKER":
        for _ in range(max_retries):
            new_mol = insert_ch2_linker(mol)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "DELETE_CH2_LINKER":
        for _ in range(max_retries):
            new_mol = delete_ch2_linker(mol)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)

    if op == "SWAP_HALOGEN":
        new_mol = swap_halogen(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "ALKYLATE_HETERO":
        for _ in range(max_retries):
            sites = get_hetero_nucleophile_sites(mol)
            if not sites:
                break
            site = random.choice(sites)
            frag_smiles = random.choice(["[*:1]C", "[*:1]CC"])
            frag = Chem.MolFromSmiles(frag_smiles)
            if frag is None:
                break
            new_mol = attach_fragment_at_atom(mol, site, frag)
            if new_mol is not None:
                return get_largest_fragment(new_mol)
        return get_largest_fragment(mol)
    
    if op == "OXIDIZE_ALCOHOL":
        new_mol = oxidize_alcohol(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "REDUCE_CARBONYL":
        new_mol = reduce_carbonyl(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol
    
    if op == "HYDROLYZE_ESTER":
        new_mol = hydrolyze_ester(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "FORM_ESTER":
        new_mol = form_ester_intramolecular(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "FORM_AMIDE":
        new_mol = form_amide_intramolecular(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "SATURATE_RING":
        new_mol = saturate_aromatic_ring(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "AROMATIZE_RING":
        new_mol = aromatize_ring(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "EXPAND_RING":
        new_mol = expand_ring(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "CONTRACT_RING":
        new_mol = contract_ring(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "FUSE_RINGS":
        new_mol = fuse_rings(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "HALOGENATE":
        new_mol = halogenate(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "DEHALOGENATE":
        new_mol = dehalogenate(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "NITRATE":
        new_mol = nitrate(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "SULFONYLATE":
        new_mol = sulfonylate(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "METHYLATE_AMINE":
        new_mol = methylate_amine(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "DEMETHYLATE_AMINE":
        new_mol = demethylate_amine(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "ACETYLATE_PHENOL":
        new_mol = acetylate_phenol(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "DEACETYLATE":
        new_mol = deacetylate(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "BIOISOSTERIC_SWAP":
        new_mol = bioisosteric_swap(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "ADD_POLAR_GROUP":
        new_mol = add_polar_group(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "ADD_LIPOPHILIC_GROUP":
        new_mol = add_lipophilic_group(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "SWAP_POLAR_LIPOPHILIC":
        new_mol = swap_polar_lipophilic(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    if op == "CYCLIZE_CHAIN":
        new_mol = cyclize_chain(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol
    
    if op == "REDUCE_CHIRALITY":
        new_mol = reduce_chirality_aggressive(mol, max_retries)
        return get_largest_fragment(new_mol) if new_mol is not None else mol

    return mol

def get_largest_fragment(mol):
    if mol is None:
        raise ValueError("SMILES non valido")
    frags = Chem.GetMolFrags(mol, asMols=True)
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    return largest














