import random
import subprocess
import os
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import rdkit.Chem.AllChem
import rdkit.Chem.QED
import sascorer
import rdkit.Chem.Crippen as cri
from train import *
from random import shuffle
import csv
import torch
import shutil
import sys
import secrets

device = 'cuda'            #'cpu' or 'cuda'
generations = 1000         #number of generations
g = 0                      #start generation
FP_KIND = "morgan"         #fingerprint type
FP_NBITS = 2048            #fingerprint bits
FP_RADIUS = 2              #fingerprint radius
CHILDRENS_PER_MOL = 10     #number of molecules generated per parent mol
GLOBAL_SIM_CUTOFF = 0.40   #similarity thrashold for elite pool molecules
TOP_N = 20                 #elite pool size
SCAFFOLD = None            #scaffold smilses for lead optimization or None
OUTPUT_FOLDER = "results"  #output folder

if os.path.exists(OUTPUT_FOLDER):
    print("The output folder already exists")
else:
    os.makedirs(OUTPUT_FOLDER)
    print("Output Folder created")

random.seed(secrets.randbelow(10000000000000000000))

LOWER_IS_BETTER = True
FP_RADIUS = 2
FP_NBITS = 2048

n_core = len(CORE_FRAGMENTS)
n_decor = len(DECOR_FRAGMENTS)
policy = PolicyNet(fp_dim=FP_NBITS, n_actions=len(ACTION_VOCAB), n_core=n_core, n_decor=n_decor).to(device)

def cleanup():
    try:
        subprocess.run(
            "rm -f smi.smi smi.log smi.sd glide.log glide_subjobs.log "
            "glide_subjob_poses.zip glide_subjobs.tar.gz glide_pv.maegz "
            "glide.csv glide_skip.csv glide.vsdb out smi_actions.smi",
            shell=True, check=False
        )
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass

def copy_outputs(g: int, bbb: int):
    try:
        if os.path.exists("glide_pv.maegz"):
            shutil.copyfile("glide_pv.maegz", os.path.join(OUTPUT_FOLDER, f"{g}_{bbb}_pv.maegz"))
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass

def doca():

    f = open("smi.smi","r")
    sss = f.read().split('\n')[:-1]
    f.close()

    f = open("smi_actions.smi","r")
    sss_actions = f.read().split('\n')[:-1]
    f.close()
    
    sons_dict = {}

    command = """./ligprep.sh"""
    subprocess.run(command, stdout=subprocess.PIPE,universal_newlines=True, shell=True)

    command = """./glide.sh"""
    subprocess.run(command, stdout=subprocess.PIPE,universal_newlines=True, shell=True)

    f = open("glide.csv","r")
    lines = f.read().split("\n")[1:-1]
    f.close()

    for l in lines:
        ll = l.split(",")
        name = ll[1].replace("\"","").split(":")[1]
        qed = rdkit.Chem.QED.default(Chem.MolFromSmiles(sss[int(name)-1]))
        saMol = sascorer.calculateScore(Chem.MolFromSmiles(sss[int(name)-1]))
        logp = cri.MolLogP(Chem.MolFromSmiles(sss[int(name)-1]))
        chiral = len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(sss[int(name)-1]), force=True, includeUnassigned=True))
        ds = float(ll[5])
        le = float(ll[5])/np.sqrt(float(Chem.MolFromSmiles(sss[int(name)-1]).GetNumHeavyAtoms()))
        actions = sss_actions[int(name)-1]
        if name in sons_dict.keys():
            if le<float(sons_dict[name][2]):
                sons_dict[name]=[sss[int(name)-1],str(ds),str(le),str(qed),str(saMol),str(logp),str(chiral),str(actions)]
        else:
            sons_dict[name]=[sss[int(name)-1],str(ds),str(le),str(qed),str(saMol),str(logp),str(chiral),str(actions)]
    
    return sons_dict

def write_done_so_far(sons_dict, so_far, g, s, bbb, b):
    f = open("done_so_far","a")
    for k in sons_dict.keys():
        vec = sons_dict[k]
        f.write(f"{vec[0]} = GEN{g}_{s} 0 {vec[1]} {vec[2]} {vec[3]} {vec[4]} {vec[5]} {vec[6]} {g}_{str(bbb)} {str(vec[7])} {str(b)}\n")
        so_far.append(vec[0])
    f.close()

def write_done_so_far_batched(sons_dict, meta_by_index, so_far, g):
    with open("done_so_far", "a") as f:
        for name, vec in sons_dict.items():
            idx = int(name) - 1
            s, bbb1, b = meta_by_index[idx]
            f.write(f"{vec[0]} = GEN{g}_{s} 0 {vec[1]} {vec[2]} {vec[3]} {vec[4]} {vec[5]} {vec[6]} {g}_{str(bbb1)} {str(vec[7])} {str(b)}\n")
            so_far.append(vec[0])

elements = [1, 2, 3]
probabilities = [0.334,0.333,0.333]
models = []
for i in range(1,4):
    policy = PolicyNet(fp_dim=FP_NBITS, n_actions=len(ACTION_VOCAB), n_core=n_core, n_decor=n_decor).to(device)
    state = torch.load("models/policy_final_"+str(i)+".pt", map_location=device)
    policy.load_state_dict(state["model"])
    models.append(policy)

def generate_mols(model,maxa,n,smi,FP_KIND,FP_NBITS,FP_RADIUS):
    i = 0
    it = 0
    maxiter = n*10
    mols = []
    while i<n and it<maxiter:
        mol = generate_genetic(policy=model,src_smiles=smi,max_steps=maxa,n_samples=1,temperature=1.0,top_p=0.95,fp_kind=FP_KIND,fp_nbits=FP_NBITS,fp_radius=FP_RADIUS,unique=True,seed=None,out_prefix=None)[0]
        if mol["smiles"] not in mols:
            if SCAFFOLD!=None and Chem.MolFromSmiles(mol["smiles"]).HasSubstructMatch(Chem.MolFromSmiles(SCAFFOLD)):
                mols.append(mol)
                i = i + 1
            if SCAFFOLD==None:
                mols.append(mol)
                i = i + 1 
        it = it + 1
    return mols

while (g<generations):
    g=g+1
    
    f = open("done_so_far","r")
    mols = f.read().split('\n')[:-1]
    so_far = []
    origin_vec = []
    all_vec = []
    for mol in mols:
        ms = mol.split()
        all_vec.append(ms)
        so_far.append(ms[0])
        origin_vec.append(ms[-1])
    f.close()
    
    f = open("best_so_far","r")
    bests = f.read().split('\n')[:-1]
    bests_smi = []
    bests_ds = []
    bests_pos = []
    for best in bests:
        for mol in mols:
            if (best.split(" ")[0]==mol.split(" ")[0]):
                bests_smi.append(best.split(" ")[0])
                bests_ds.append(mol.split(" ")[5])
                bests_pos.append(best.split(" ")[-1])
                break
    
    bests_ds = np.array(bests_ds).astype('float')
    bests_smi = np.array(bests_smi)
    bests_pos = np.array(bests_pos)
    idx_sorted = np.argsort(bests_ds)
    best_smi_sorted = bests_smi[idx_sorted]
    best_pos_sorted = bests_pos[idx_sorted]

    f.close()
    print ("Generation "+str(g))

    all_smiles = []
    meta_by_index = []  
    seen_new = set()    
    for bbb in range(len(best_smi_sorted)):
        try:
            s = 1 if (g == 1) else np.random.choice(elements, 1, p=probabilities)[0]
            smiles = generate_mols(models[s - 1], s, CHILDRENS_PER_MOL, best_smi_sorted[bbb], FP_KIND, FP_NBITS, FP_RADIUS)

            valid = 0
            for m in smiles:
                if (m["smiles"] not in so_far) and (m["smiles"] not in seen_new):
                    all_smiles.append(m)
                    meta_by_index.append((s, bbb + 1, best_pos_sorted[bbb]))
                    seen_new.add(m["smiles"])
                    valid += 1

        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            continue

    if len(all_smiles) < 1:
        pass
    else:
        cleanup()  
        with open("smi.smi", "w") as fc:
            for i in range(len(all_smiles)):
                fc.write(all_smiles[i]["smiles"] + "\n")
        with open("smi_actions.smi", "w") as fc:
            for i in range(len(all_smiles)):
                fc.write(all_smiles[i]["actions_json"] + "\n")

        sons_dict = doca()  
        copy_outputs(g, 0)

        write_done_so_far_batched(sons_dict, meta_by_index, so_far, g)

    with open("done_so_far", "r") as f:
        mols_lines = f.read().split('\n')[:-1]

    records = []
    fps_cache = []  

    for i, line in enumerate(mols_lines):
        toks = line.split()
        if len(toks) < 6:
            continue
        smi = toks[0]
        try:
            ds = float(toks[5])  
        except Exception:
            continue

        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=FP_RADIUS, nBits=FP_NBITS)

        records.append({
            'smi': smi,
            'ds': ds,
            'line': line,
            'pos': i + 1
        })
        fps_cache.append(fp)

    if not records:
        open("best_so_far", "w").close()
        command = f"cp best_so_far {OUTPUT_FOLDER}/best_so_far_{g}"
        subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        command = f"cp done_so_far {OUTPUT_FOLDER}/done_so_far_{g}"
        subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    else:
        order = sorted(range(len(records)), key=lambda i: records[i]['ds'], reverse=not LOWER_IS_BETTER)

        selected_idx = []  

        for idx in order:
            keep = True
            fp_i = fps_cache[idx]
            for j in selected_idx:
                sim = DataStructs.TanimotoSimilarity(fp_i, fps_cache[j])
                if sim > GLOBAL_SIM_CUTOFF:
                    keep = False
                    break
            if keep:
                selected_idx.append(idx)
                if TOP_N is not None and len(selected_idx) >= TOP_N:
                    break

        selected_idx.sort(key=lambda i: records[i]['ds'], reverse=not LOWER_IS_BETTER)

        with open("best_so_far", "w") as f:
            for i_sel in selected_idx:
                r = records[i_sel]
                f.write(r['line'] + " " + str(r['pos']) + "\n")

        command = f"cp best_so_far {OUTPUT_FOLDER}/best_so_far_{g}"
        subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        command = f"cp done_so_far {OUTPUT_FOLDER}/done_so_far_{g}"
        subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)


