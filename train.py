import os
import math
import json
import csv
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem import Descriptors, Crippen

from mut_all import (
    init_fragment_libraries,
    CORE_FRAGMENTS,
    DECOR_FRAGMENTS,
    apply_action,
    Action as MutAction,
)

from rdkit.Chem import MACCSkeys
try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP
    _has_avalon = True
except Exception:
    _has_avalon = False

def _bv_to_tensor(bv, nBits: int) -> torch.Tensor:
    arr = torch.zeros(nBits, dtype=torch.float32)
    DataStructs.ConvertToNumpyArray(bv, arr.numpy())
    return arr

def make_fp(m: Chem.Mol, kind: str = "morgan", nBits: int = 2048, radius: int = 2) -> torch.Tensor:
    if m is None:
        return torch.zeros(nBits, dtype=torch.float32)

    kind = (kind or "morgan").lower()

    if kind == "morgan":
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
        return _bv_to_tensor(bv, nBits)

    if kind == "feat_morgan":  
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits, useFeatures=True)
        return _bv_to_tensor(bv, nBits)

    if kind == "rdk":
        bv = Chem.RDKFingerprint(m, fpSize=nBits)  
        return _bv_to_tensor(bv, nBits)

    if kind == "pattern":
        bv = Chem.PatternFingerprint(m, fpSize=nBits)
        return _bv_to_tensor(bv, nBits)

    if kind == "layered":
        bv = rdMD.GetHashedLayeredFingerprintAsBitVect(m, nBits=nBits)
        return _bv_to_tensor(bv, nBits)

    if kind == "atompair":
        bv = rdMD.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nBits)
        return _bv_to_tensor(bv, nBits)

    if kind == "torsion":
        bv = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nBits)
        return _bv_to_tensor(bv, nBits)

    if kind == "maccs":
        bv = MACCSkeys.GenMACCSKeys(m)  
        arr = torch.zeros(167, dtype=torch.float32)
        DataStructs.ConvertToNumpyArray(bv, arr.numpy())
        return arr

    if kind == "avalon" and _has_avalon:
        bv = GetAvalonFP(m, nBits)
        return _bv_to_tensor(bv, nBits)

    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
    return _bv_to_tensor(bv, nBits)

ALL_OPS = [
    "ADD_FRAG",
    "DECORATE_AROM",
    "DELETE_SIDECHAIN",
    "DELETE_DECOR_AROM",
    "REPLACE_SIDECHAIN_CORE",
    "REPLACE_DECOR_AROM",
    "INSERT_CH2_LINKER",
    "DELETE_CH2_LINKER",
    "SWAP_HALOGEN",
    "HALOGENATE",
    "DEHALOGENATE",
    "NITRATE",
    "SULFONYLATE",
    "ADD_POLAR_GROUP",
    "ADD_LIPOPHILIC_GROUP",
    "SWAP_POLAR_LIPOPHILIC",
    "ALKYLATE_HETERO",
    "OXIDIZE_ALCOHOL",
    "REDUCE_CARBONYL",
    "HYDROLYZE_ESTER",
    "FORM_ESTER",
    "FORM_AMIDE",
    "METHYLATE_AMINE",
    "DEMETHYLATE_AMINE",
    "ACETYLATE_PHENOL",
    "DEACETYLATE",
    "BIOISOSTERIC_SWAP",
    "SATURATE_RING",
    "AROMATIZE_RING",
    "EXPAND_RING",
    "CONTRACT_RING",
    "FUSE_RINGS",
    "CYCLIZE_CHAIN",
    "REDUCE_CHIRALITY",
]
STOP_OP = "STOP"
ACTION_VOCAB = ALL_OPS + [STOP_OP]
ACTION2ID = {op: i for i, op in enumerate(ACTION_VOCAB)}
ID2ACTION = {i: op for op, i in ACTION2ID.items()}

CORE_SET = {"ADD_FRAG", "REPLACE_SIDECHAIN_CORE"}
DECOR_SET = {"DECORATE_AROM", "REPLACE_DECOR_AROM"}


def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def mol_to_smiles(m: Optional[Chem.Mol]) -> str:
    if m is None:
        return "None"
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return "None"

def tanimoto(m1: Optional[Chem.Mol], m2: Optional[Chem.Mol], fp_kind: str, fp_nbits: int, fp_radius: int) -> float:
    if m1 is None or m2 is None:
        return 0.0
    kind = fp_kind.lower()
    if kind == "morgan":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, fp_radius, nBits=fp_nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, fp_radius, nBits=fp_nbits)
    elif kind == "feat_morgan":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, fp_radius, nBits=fp_nbits, useFeatures=True)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, fp_radius, nBits=fp_nbits, useFeatures=True)
    elif kind == "rdk":
        fp1 = Chem.RDKFingerprint(m1, fpSize=fp_nbits); fp2 = Chem.RDKFingerprint(m2, fpSize=fp_nbits)
    elif kind == "pattern":
        fp1 = Chem.PatternFingerprint(m1, fpSize=fp_nbits); fp2 = Chem.PatternFingerprint(m2, fpSize=fp_nbits)
    elif kind == "layered":
        fp1 = rdMD.GetHashedLayeredFingerprintAsBitVect(m1, nBits=fp_nbits); fp2 = rdMD.GetHashedLayeredFingerprintAsBitVect(m2, nBits=fp_nbits)
    elif kind == "atompair":
        fp1 = rdMD.GetHashedAtomPairFingerprintAsBitVect(m1, nBits=fp_nbits); fp2 = rdMD.GetHashedAtomPairFingerprintAsBitVect(m2, nBits=fp_nbits)
    elif kind == "torsion":
        fp1 = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(m1, nBits=fp_nbits); fp2 = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(m2, nBits=fp_nbits)
    elif kind == "maccs":
        fp1 = MACCSkeys.GenMACCSKeys(m1); fp2 = MACCSkeys.GenMACCSKeys(m2)
    elif kind == "avalon" and _has_avalon:
        fp1 = GetAvalonFP(m1, fp_nbits); fp2 = GetAvalonFP(m2, fp_nbits)
    else:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, fp_radius, nBits=fp_nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, fp_radius, nBits=fp_nbits)

    return float(DataStructs.TanimotoSimilarity(fp1, fp2))

try:
    from rdkit.Chem.QED import qed as _qed_fn
except Exception:
    _qed_fn = None

def qed_score(m: Optional[Chem.Mol]) -> float:
    if m is None or _qed_fn is None:
        return 0.0
    try:
        return float(_qed_fn(m))
    except Exception:
        return 0.0

try:
    import sascorer as _sa_mod  # type: ignore
except Exception:
    _sa_mod = None

def sa_score(m: Optional[Chem.Mol]) -> float:
    if m is None:
        return 0.0
    if _sa_mod is not None:
        try:
            return 1.0-(float(_sa_mod.calculateScore(m))-1.0)/9.0
        except Exception:
            return 0.0
    try:
        ha = m.GetNumHeavyAtoms()
        rings = rdMD.CalcNumRings(m)
        fr_sp3 = rdMD.CalcFractionCSP3(m)
        hetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (6, 1))
        chiral = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
        score = 1.0 + 0.10 * ha + 0.50 * rings + 0.30 * len(chiral) + 0.20 * hetero - 0.50 * fr_sp3
        return 1.0-(float(max(1.0, min(10.0, score)))-1.0)/9.0
    except Exception:
        return 0.0

def marginal_internal_diversity(gen_m, prev_mols, k: int, fp_kind: str, fp_nbits: int, fp_radius: int) -> float:
    if gen_m is None or len(prev_mols) == 0:
        return 0.0
    sims = [tanimoto(gen_m, pm, fp_kind, fp_nbits, fp_radius) for pm in prev_mols]
    if k is not None and k > 0:
        sims.sort(reverse=True)                 
        sims = sims[:min(k, len(sims))]         
    mean_sim = sum(sims) / len(sims)
    return max(0.0, min(1.0, 1.0 - float(mean_sim)))

@dataclass
class StepResult:
    mol: Optional[Chem.Mol]
    done: bool
    changed: bool
    logp: torch.Tensor
    ent: torch.Tensor

class MutEnv:
    def __init__(self, src_mol: Chem.Mol, max_steps: int = 12, fp_kind: str = "morgan", fp_nbits: int = 2048, fp_radius: int = 2):
        self.src_mol = src_mol
        self.mol = Chem.Mol(src_mol)
        self.t = 0
        self.max_steps = max_steps
        self._last_smi = mol_to_smiles(self.mol)
        self._did_any_change = False
        self.kind = fp_kind
        self.nbits = fp_nbits
        self.radius = fp_radius
        self.actions = [] 

    def state_fp(self) -> torch.Tensor:        
        return make_fp(self.mol, self.kind, self.nbits, self.radius)  

    def step(self, policy, temperature=1.0, top_p=1.0) -> StepResult:
        self.t += 1
        device = next(policy.parameters()).device
        x = self.state_fp().unsqueeze(0).to(device)
        h = policy.features(x)
        logits_action = policy.action_head(h).squeeze(0)

        if self.t == 1:
            stop_id = ACTION2ID[STOP_OP]
            logits_action[stop_id] = -1e9

        if len(CORE_FRAGMENTS) == 0:
            for op in ("ADD_FRAG", "REPLACE_SIDECHAIN_CORE"):
                logits_action[ACTION2ID[op]] = -1e9
        if len(DECOR_FRAGMENTS) == 0:
            for op in ("DECORATE_AROM", "REPLACE_DECOR_AROM"):
                logits_action[ACTION2ID[op]] = -1e9

        chosen_a, logp_a, ent_a = sample_categorical(logits_action, temperature, top_p)
        op = ID2ACTION[int(chosen_a)]

        if op == STOP_OP and self.t >= 2:
            self.actions.append({"op": op, "frag": None})
            return StepResult(self.mol, True, False, logp_a, ent_a)

        _dev = logits_action.device
        frag_logp = torch.tensor(0.0, device=_dev)
        frag_ent = torch.tensor(0.0, device=_dev)
        frag_idx: Optional[int] = None
        if op in CORE_SET:
            logits_core = policy.core_head(h).squeeze(0)
            frag_idx, frag_logp, frag_ent = sample_categorical(logits_core, temperature, top_p)
            frag_idx = int(frag_idx)
        elif op in DECOR_SET:
            logits_decor = policy.decor_head(h).squeeze(0)
            frag_idx, frag_logp, frag_ent = sample_categorical(logits_decor, temperature, top_p)
            frag_idx = int(frag_idx)

        before = self._last_smi
        try:
            new_mol = apply_action(self.mol, MutAction(op, frag_idx))
        except Exception:
            new_mol = self.mol
        new_smi = mol_to_smiles(new_mol)
        changed = (new_smi != before)
        if changed:
            self._did_any_change = True
            self.mol = new_mol
            self._last_smi = new_smi

        self.actions.append({"op": op, "frag": frag_idx})

        done = (self.t >= self.max_steps)
        return StepResult(self.mol, done, changed, logp_a + frag_logp, ent_a + frag_ent)

class PolicyNet(nn.Module):
    def __init__(self, fp_dim: int, n_actions: int, n_core: int, n_decor: int, hidden: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        
        self.core_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_core),
        )
        self.decor_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_decor),
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.action_head(h), self.core_head(h), self.decor_head(h)

    def features(self, x):
        return self.backbone(x)

    def action_logits(self, x):
        return self.action_head(self.backbone(x))

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    return logits / max(1e-6, temperature)

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    if logits.numel() == 0 or logits.shape[-1] == 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    shifted = cutoff.clone()
    if shifted.shape[-1] > 0:
        shifted[..., 1:] = cutoff[..., :-1]
    shifted[..., 0] = False
    indices_to_remove = torch.zeros_like(shifted, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_idx, src=shifted)
    new_logits = logits.masked_fill(indices_to_remove, float("-inf"))
    if torch.isneginf(new_logits).all():
        best = torch.argmax(logits)
        new_logits = logits.clone()
        new_logits[best] = logits[best]
    return new_logits

def sample_categorical(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    if logits.numel() == 0 or logits.shape[-1] == 0:
        raise RuntimeError("Tried to sample from empty logits vector.")
    logits_orig = logits
    logits = apply_temperature(logits, temperature)
    logits = top_p_filter(logits, top_p)
    if torch.isneginf(logits).all():
        logits = logits_orig
    dist = Categorical(logits=logits)
    idx = dist.sample()
    logp = dist.log_prob(idx)
    ent = dist.entropy()
    return idx, logp, ent

@dataclass
class TrainConfig:
    max_steps: int = 12
    episodes_per_src: int = 4
    epochs: int = 5
    lr: float = 1e-3
    entropy_coef: float = 0.01
    baseline_decay: float = 0.9
    temperature: float = 1.0
    top_p: float = 0.95
    seed: Optional[int] = 42
    src_per_epoch: Optional[int] = None
    fp_kind: str = "morgan"
    fp_nbits: int = 2048
    fp_radius: int = 2
    w_sa: float = 0.5
    w_qed: float = 0.5
    log_dirname: str = "logs"
    training_csv: str = "train_set.csv"   
    append_unique: bool = True            
    fixed_training_set: bool = False
    resume_ckpt: str = None

class ReinforceTrainer:
    def __init__(self, policy: PolicyNet, cfg: TrainConfig, device: str = "cpu", out_dir: str = "ckpts"):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.baseline = 0.0
        self.out_dir = out_dir
        if cfg.resume_ckpt:
            state = torch.load(cfg.resume_ckpt, map_location=device)
            if "model" in state:
                self.policy.load_state_dict(state["model"])
                print(f"[RESUME] Loaded weights from {cfg.resume_ckpt}")
            if "optimizer" in state:
                try:
                    self.opt.load_state_dict(state["optimizer"])
                    print(f"[RESUME] Loaded optimizer state from {cfg.resume_ckpt}")
                except Exception as e:
                    print(f"[RESUME] Impossibile caricare ottimizzatore: {e}")

        os.makedirs(self.out_dir, exist_ok=True)
        self.log_dir = os.path.join(self.out_dir, self.cfg.log_dirname)
        os.makedirs(self.log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"epoch_log_{ts}.csv")
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([
                "epoch","n_src","n_episodes",
                "avg_R_total","avg_R_base","avg_R_div",
                "avg_tanimoto_src","std_tanimoto_src",
                "avg_SA","std_SA",
                "avg_QED","std_QED",
                "avg_SA_src","std_SA_src",
                "avg_QED_src","std_QED_src",
                "avg_deltaSA","std_deltaSA",
                "avg_deltaQED","std_deltaQED",
                "avg_mid","avg_steps",
            ])


    def _episode(self, src_mol: Chem.Mol) -> Tuple[float, List[torch.Tensor], List[torch.Tensor], str, int, Dict[str,float], str]:
        env = MutEnv(src_mol, max_steps=self.cfg.max_steps, fp_kind=self.cfg.fp_kind, fp_nbits=self.cfg.fp_nbits, fp_radius=self.cfg.fp_radius)
        logps: List[torch.Tensor] = []
        ents: List[torch.Tensor] = []
        done = False

        sa_src = sa_score(src_mol)
        q_src  = qed_score(src_mol)

        while not done:
            step_res = env.step(self.policy, self.cfg.temperature, self.cfg.top_p)
            logps.append(step_res.logp)
            ents.append(step_res.ent)
            done = step_res.done

        gen_mol = env.mol
        gen_smi = mol_to_smiles(gen_mol)

        sim_src = tanimoto(gen_mol, src_mol, self.cfg.fp_kind, self.cfg.fp_nbits, self.cfg.fp_radius)
        sa_gen  = sa_score(gen_mol)
        q_gen   = qed_score(gen_mol)

        ##delta_sa  = (sa_gen - sa_src)/(sa_src)*100.0
        #delta_qed = (q_gen - q_src)/(q_src)*100.0
        delta_sa  = (sa_gen - sa_src)
        delta_qed = (q_gen - q_src)

        R_base = (self.cfg.w_sa  * float(sa_gen) +
                  self.cfg.w_qed * float(q_gen))

        if gen_smi == mol_to_smiles(src_mol):
            R_base -= 0.05

        actions_json = json.dumps(env.actions, ensure_ascii=False)
        aux = {
            "sim_src": float(sim_src),
            "sa": float(sa_gen),
            "qed": float(q_gen),
            "sa_src": float(sa_src),
            "qed_src": float(q_src),
            "delta_sa": float(delta_sa),
            "delta_qed": float(delta_qed),
            "R_base": float(R_base),
        }
        steps = env.t
        return float(R_base), logps, ents, gen_smi, steps, aux, actions_json

    def train(self, src_list: List[str], src_epoch: List[str]):
        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / max(1, len(xs)))
        def _std(xs: List[float]) -> float:
             n = len(xs)
             if n == 0:
                 return 0.0
             mu = _mean(xs)
             var = sum((x - mu) ** 2 for x in xs) / n  
             return float(var ** 0.5)


        for epoch in range(1, self.cfg.epochs + 1):
            epoch_generated: list[str] = []
            seen_epoch: set[str] = set()
            self.policy.train()
            
            if self.cfg.fixed_training_set:
                pool = src_list  
                if self.cfg.src_per_epoch is not None and self.cfg.src_per_epoch > 0:
                    k = min(self.cfg.src_per_epoch, len(pool))
                    selected_srcs = random.sample(pool, k)
                else:
                    selected_srcs = pool  
            else:
                src_list_ep = []
                if src_epoch:
                    for i in range(len(src_epoch)):
                        if str(src_epoch[i]) == str(epoch - 1):
                            src_list_ep.append(src_list[i])
                else:
                    src_list_ep = src_list
            
                if self.cfg.src_per_epoch is not None and self.cfg.src_per_epoch > 0:
                    k = min(self.cfg.src_per_epoch, len(src_list_ep))
                    selected_srcs = random.sample(src_list_ep, k)
                else:
                    selected_srcs = src_list_ep

            ep_R_total: List[float] = []
            ep_R_base:  List[float] = []
            ep_R_div:   List[float] = []
            ep_sim_src: List[float] = []
            ep_SA:      List[float] = []
            ep_QED:     List[float] = []
            ep_steps:   List[float] = []
            ep_mid:     List[float] = []
            ep_SA_src: List[float] = []
            ep_QED_src: List[float] = []
            ep_deltaSA: List[float] = []
            ep_deltaQED: List[float] = []

            total_R_sum = 0.0
            total_count = 0

            prog_bar = tqdm(enumerate(selected_srcs, 1), total=len(selected_srcs), desc=f"[Epoch {epoch}]")

            for src_idx, src_smi in prog_bar:
                src_m = mol_from_smiles(src_smi)
                if src_m is None:
                    continue

                prev_gen_mols: List[Chem.Mol] = []

                batch_loss = 0.0

                for _ep in range(1, self.cfg.episodes_per_src + 1):
                    R_base, logps, ents, gen_smi, steps, aux, actions_json = self._episode(src_m)
                    gen_m = mol_from_smiles(gen_smi)

                    mid = marginal_internal_diversity(gen_m, prev_gen_mols, 5, self.cfg.fp_kind, self.cfg.fp_nbits, self.cfg.fp_radius) 
                    #R_div = self.cfg.w_div * mid
                    R_total = float(R_base) #+ float(R_div)
                    
                    self.baseline = self.cfg.baseline_decay * self.baseline + (1 - self.cfg.baseline_decay) * R_total
                    advantage = R_total - self.baseline

                    logp_sum = torch.stack(logps).sum()
                    ent_sum = torch.stack(ents).sum()
                    loss = -(logp_sum * advantage) - self.cfg.entropy_coef * ent_sum
                    batch_loss += loss

                    # Accumula statistiche per epoca
                    ep_R_total.append(R_total)
                    ep_R_base.append(float(R_base))
                    ep_R_div.append(float(0.0))
                    ep_sim_src.append(float(aux["sim_src"]))
                    ep_SA.append(float(aux["sa"]))
                    ep_QED.append(float(aux["qed"]))
                    ep_steps.append(float(steps))
                    ep_mid.append(float(mid))
                    ep_SA_src.append(float(aux["sa_src"]))
                    ep_QED_src.append(float(aux["qed_src"]))
                    ep_deltaSA.append(float(aux["delta_sa"]))
                    ep_deltaQED.append(float(aux["delta_qed"]))

                    total_R_sum += R_total
                    total_count += 1

                    if gen_m is not None:
                        prev_gen_mols.append(gen_m)
                        smi_out = mol_to_smiles(gen_m)
                        if smi_out not in seen_epoch:
                            seen_epoch.add(smi_out)
                            epoch_generated.append({
                                "smiles": smi_out,
                                "src": src_smi,
                                "sim_src": float(aux.get("sim_src", 0.0))
                            })

                if self.cfg.episodes_per_src > 0:
                    self.opt.zero_grad()
                    (batch_loss / max(1, self.cfg.episodes_per_src)).backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.opt.step()

                prog_bar.set_postfix({
                    "avgR(ep)": f"{_mean(ep_R_total):.3f}",
                    "baseline": f"{self.baseline:.3f}"
                })

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([
                    epoch,
                    len(selected_srcs),
                    total_count,
                    f"{_mean(ep_R_total):.6f}",
                    f"{_mean(ep_R_base):.6f}",
                    f"{_mean(ep_R_div):.6f}",
                    f"{_mean(ep_sim_src):.6f}",
                    f"{_std(ep_sim_src):.6f}",
                    f"{_mean(ep_SA):.6f}",
                    f"{_std(ep_SA):.6f}",
                    f"{_mean(ep_QED):.6f}",
                    f"{_std(ep_QED):.6f}",
                    f"{_mean(ep_SA_src):.6f}",
                    f"{_std(ep_SA_src):.6f}",
                    f"{_mean(ep_QED_src):.6f}",
                    f"{_std(ep_QED_src):.6f}",
                    f"{_mean(ep_deltaSA):.6f}",
                    f"{_std(ep_deltaSA):.6f}",
                    f"{_mean(ep_deltaQED):.6f}",
                    f"{_std(ep_deltaQED):.6f}",
                    f"{_mean(ep_mid):.6f}",
                    f"{_mean(ep_steps):.3f}",
                ])

            ckpt_path = os.path.join(self.out_dir, f"policy_epoch_{epoch}.pt")
            torch.save({
                "model": self.policy.state_dict(),
                "optimizer": self.opt.state_dict(),
                "cfg": self.cfg.__dict__,
                "action_vocab": ACTION_VOCAB,
            }, ckpt_path)
            print(f"[Epoch {epoch}] saved {ckpt_path} avgR={_mean(ep_R_total):.4f}")
            if not self.cfg.fixed_training_set:
                added = self._append_to_training_set(epoch_generated, epoch)
                print(f"[Epoch {epoch}] appended {added} new molecules to {self.cfg.training_csv}")
                if epoch < self.cfg.epochs:
                    try:
                        src_list, src_epoch = load_srcs(self.cfg.training_csv, src_col="smiles")
                    except Exception:
                        pass
            else:
                pass

        final_path = os.path.join(self.out_dir, "policy_final.pt")
        torch.save({
                "model": self.policy.state_dict(),
                "optimizer": self.opt.state_dict(),
                "cfg": self.cfg.__dict__,
                "action_vocab": ACTION_VOCAB,
            }, ckpt_path)
        print(f"[DONE] Saved final checkpoint: {final_path}")
        print(f"[LOG] Epoch-level log saved to: {self.log_path}")

    def _compute_metrics(self, smi: str) -> dict:
        m = mol_from_smiles(smi)
        if m is None:
            return {"smiles": smi, "qed": 0.0, "sa": 0.0, "logp": 0.0, "chiral": 0, "mw": 0.0}
        try:
            chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
        except Exception:
            chiral = 0
        return {
            "smiles": smi,
            "qed": float(qed_score(m)),
            "sa": float(sa_score(m)),
            "logp": float(Crippen.MolLogP(m)),
            "chiral": int(chiral),
            "mw": float(Descriptors.MolWt(m)),
        }

    def _append_to_training_set(self, items: list[dict], epoch: int) -> int:
        if not items:
            return 0

        tmp = []
        seen_batch = set()
        for it in items:
            smi = str(it.get("smiles", "")).strip()
            if not smi:
                continue
            if smi in seen_batch:
                continue
            seen_batch.add(smi)
            tmp.append({
                "smiles": smi,
                "src": str(it.get("src", "")),
                "sim_src": float(it.get("sim_src", 0.0)),
            })
        if not tmp:
            return 0

        rows = []
        for it in tmp:
            met = self._compute_metrics(it["smiles"])
            rows.append({
                "smiles": it["smiles"],
                "src": it["src"],
                "sim_src": it["sim_src"],
                "qed": met["qed"],
                "sa": met["sa"],
                "logp": met["logp"],
                "chiral": met["chiral"],
                "mw": met["mw"],
                "epoch": int(epoch),
            })
        df_new = pd.DataFrame(rows).drop_duplicates(subset=["smiles"])

        if not os.path.exists(self.cfg.training_csv):
            df_new.to_csv(self.cfg.training_csv, index=False)
            return len(df_new)

        try:
            df_old = pd.read_csv(self.cfg.training_csv, usecols=["smiles"])
            seen_old = set(df_old["smiles"].astype(str).tolist())
        except Exception:
            seen_old = set()
        df_new = df_new[~df_new["smiles"].astype(str).isin(seen_old)]
        if len(df_new) == 0:
            return 0

        df_new.to_csv(self.cfg.training_csv, mode="a", header=False, index=False)
        return len(df_new)

@torch.no_grad()
def generate(
    policy: PolicyNet,
    src_smiles: str,
    n_samples: int = 20,
    max_steps: int = 12,
    temperature: float = 1.1,
    top_p: float = 0.95,
    fp_kind: str = "morgan",
    fp_nbits: int = 2048,
    fp_radius: int = 2,
    unique: bool = True,
    seed: Optional[int] = None,
    out_prefix: Optional[str] = None,   # <-- nuovo
) -> List[str]:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    src_m = mol_from_smiles(src_smiles)
    if src_m is None:
        raise ValueError("SMILES sorgente invalido")
    policy.eval()

    generated = []
    seen = set()

    # per logging azioni
    action_logs = []

    for i in range(n_samples):
        env = MutEnv(src_m, max_steps=max_steps, fp_kind=fp_kind, fp_nbits=fp_nbits, fp_radius=fp_radius)
        done = False
        while not done:
            step_res = env.step(policy, temperature, top_p)
            done = step_res.done

        smi = mol_to_smiles(env.mol)
        if unique and smi in seen:
            continue
        seen.add(smi)
        generated.append(smi)

        sa = sa_score(env.mol)
        q = qed_score(env.mol)
        sim_src = tanimoto(env.mol, src_m, fp_kind, fp_nbits, fp_radius)

        action_logs.append({
            "id": f"mol_{len(generated):03d}",
            "smiles": smi,
            "steps": env.t,
            "actions_json": json.dumps(env.actions, ensure_ascii=False),
            "SA": float(sa),
            "QED": float(q),
            "Tanimoto_src": float(sim_src),
        })

    if out_prefix is not None:
        act_path = f"{out_prefix}.csv"
        import csv as _csv
        with open(act_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["id","smiles","steps","actions_json","SA","QED","Tanimoto_src","src"])
            for row in action_logs:
                writer.writerow([
                    row["id"], row["smiles"], row["steps"], row["actions_json"],
                    f"{row['SA']:.6f}", f"{row['QED']:.6f}", f"{row['Tanimoto_src']:.6f}",
                    src_smiles
                ])

        print(f"[GENERATE] Saved: {act_path}")

    return generated

def load_srcs(csv_path: str, src_col: str | None = None):
    df = pd.read_csv(csv_path)
    col = None
    if src_col and src_col in df.columns:
        col = src_col
    elif "smiles" in df.columns:
        col = "smiles"
    elif "src" in df.columns:
        col = "src"
    else:
        raise AssertionError(f"Colonne attese: 'smiles' o 'src' in {csv_path}")
    return [str(s) for s in df[col].dropna().astype(str).tolist()], [str(s) for s in df['epoch'].dropna().astype(str).tolist()]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "sample"], required=True)
    parser.add_argument("--data", type=str, default="start.csv",
                    help="Starting CSV (accepted column: 'smiles' or 'src'). Default: start.csv")
    parser.add_argument("--out_dir", type=str, default="ckpts")

    # train args
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episodes_per_src", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src_per_epoch", type=int, default=None, help="Number of compounds per epoch (None = all)")
    parser.add_argument("--device", type=str, default="cuda")    
    parser.add_argument("--fp_kind", type=str, default="morgan", choices=["morgan","feat_morgan","rdk","pattern","layered","maccs","atompair","torsion","avalon"])
    parser.add_argument("--fp_nbits", type=int, default=2048)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--training_csv", type=str, default=None,
                    help="Cumulative training set (default: <out_dir>/train_set.csv)")
    parser.add_argument("--append_unique", action="store_true", default=True)
    parser.add_argument("--fixed_training_set", action="store_true", default=False,
                    help="Use a fixed training set (not curriculum learning)")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                    help="Checkpoint model to restart the training")
    
    # reward weights
    parser.add_argument("--w_sa", type=float, default=0.5)
    parser.add_argument("--w_qed", type=float, default=0.5)

    # sample args
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--src", type=str)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--gen_max_steps", type=int, default=16)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--out_prefix", type=str, default=None)

    args = parser.parse_args()

    fp_kind = args.fp_kind
    fp_nbits = (167 if args.fp_kind=="maccs" else args.fp_nbits)
    fp_radius = args.fp_radius

    init_fragment_libraries()
    n_core = len(CORE_FRAGMENTS)
    n_decor = len(DECOR_FRAGMENTS)

    device = args.device

    if args.mode == "train":
        training_csv = args.training_csv or os.path.join(args.out_dir, "train_set.csv")
        device = args.device    
        os.makedirs(args.out_dir, exist_ok=True)
        if args.fixed_training_set:
            df0 = pd.read_csv(args.data)
            if "smiles" in df0.columns:
                base_src_col = "smiles"
            elif "src" in df0.columns:
                base_src_col = "src"
            else:
                raise AssertionError(f"Colonne attese in {args.data}: 'smiles' o 'src'")
            base_src_list = [str(s) for s in df0[base_src_col].dropna().astype(str).tolist()]
            src_list, src_epoch = base_src_list, ["0"] * len(base_src_list)
        else:
            if not os.path.exists(training_csv):
                df0 = pd.read_csv(args.data)
                if "smiles" in df0.columns:
                    base = df0.rename(columns={"smiles": "smiles"})
                elif "src" in df0.columns:
                    base = df0.rename(columns={"src": "smiles"})
                else:
                    raise AssertionError(f"Colonne attese in {args.data}: 'smiles' o 'src'")
                if "epoch" not in base.columns:
                    base["epoch"] = 0
                base.to_csv(training_csv, index=False)
                print(f"[BOOTSTRAP] Created {training_csv} from {args.data}")
            src_list, src_epoch = load_srcs(training_csv, src_col="smiles")

        policy = PolicyNet(fp_dim=fp_nbits, n_actions=len(ACTION_VOCAB), n_core=n_core, n_decor=n_decor)

        cfg = TrainConfig(
            max_steps=args.max_steps,
            episodes_per_src=args.episodes_per_src,
            epochs=args.epochs,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            src_per_epoch=args.src_per_epoch,
            fp_kind=args.fp_kind,
            fp_nbits=args.fp_nbits,
            fp_radius=args.fp_radius,
            w_sa=args.w_sa,
            w_qed=args.w_qed,
            training_csv=training_csv,
            append_unique=args.append_unique,
            fixed_training_set=args.fixed_training_set,
            resume_ckpt=args.resume_ckpt,
        )
        if cfg.seed is not None:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        trainer = ReinforceTrainer(policy, cfg, device=device, out_dir=args.out_dir)
        trainer.train(src_list, src_epoch)

    elif args.mode == "sample":
        assert args.ckpt is not None and args.src is not None, "--ckpt e --src needed in mode=sample"
        policy = PolicyNet(fp_dim=fp_nbits, n_actions=len(ACTION_VOCAB), n_core=n_core, n_decor=n_decor).to(device)
        state = torch.load(args.ckpt, map_location=device)
        policy.load_state_dict(state["model"])
        gens = generate(
            policy=policy,
            src_smiles=args.src,
            n_samples=args.n_samples,
            max_steps=args.gen_max_steps,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            fp_kind=args.fp_kind,
            fp_nbits=args.fp_nbits,
            fp_radius=args.fp_radius,
            unique=True,
            seed=None,
            out_prefix=args.out_prefix,   
        )
        #for i, smi in enumerate(gens, 1):
        #    print(f"{i:03d}\t{smi}")

if __name__ == "__main__":
    main()

