import re
import numpy as np
from ase.calculators.vasp import Vasp

def regular_kpoint_mesh(cell, scale:int=1, kpts_max:int|None=None):
    vector_lengths = np.linalg.norm(cell, axis=1)
    ratios = vector_lengths.max() / vector_lengths
    sampling = np.max(np.stack([scale*ratios, np.ones(3)], axis=-1), axis=1)
    sampling = np.round(sampling, 0).astype(int)

    if kpts_max and kpts_max < sampling.max():
        kpts = np.round(sampling/sampling.max()*kpts_max)
        kpts = np.array([k if k > 1 else 1 for k in kpts], dtype=int)
    else:
        kpts = sampling

    return kpts


def get_resources(atoms, calc: Vasp, target_nbands_per_core, ncores_per_node, ncore=1):
    # Get nions
    atoms_string = str(atoms.symbols)
    species_counts = np.array([int(n) for n in re.findall(r"[^a-zA-Z]+", atoms_string)])
    nions = np.sum(species_counts)

    # Get nelect
    calc.write_input(atoms)
    zvals = np.zeros_like(species_counts)
    i = 0
    with open(f"{calc.directory}/POTCAR", "r") as file:
        line = file.readline()
        while line:
            if "ZVAL" in line:
                zvals[i] = float(line.split()[5])
            line = file.readline()
    nelect = np.sum(species_counts * zvals)

    # Get the number of bands
    nbands = np.max([int((nelect+2)/2 + np.max([nions/2,3])), int(3/5*nelect)])

    # Get the appropriate core-count that is still a multiple of ncore
    # This will be a range of values, with the greatest number being picked
    ncores = ncore * int(np.max([np.round(nbands / target_nbands_per_core / ncore, 0), 1]))

    # Get the appropriate node count
    nodes = int(np.ceil(ncores / ncores_per_node))

    # If more than one node would be used, just use the entire node
    # regardless of the optimal core count
    if nodes > 1:
        ncores = nodes * ncores_per_node

    # Recompute nbands_per_core and ncores_per_node
    nbands_per_core = nbands / ncores
    ncores_per_node = int(ncores/nodes)

    return dict(
        ntasks=ncores,
        nodes=nodes,
        ntasks_per_node=ncores_per_node,
        nbands=nbands,
        nbands_per_task=nbands_per_core
    )


def write_resources(calc: Vasp, resources: dict):
    path = f"{calc.directory}/job_resources.env"
    with open(path, "w") as file:
        try:
            file.write(f"NTASKS={resources["ntasks"]}\n")
        except KeyError:
            pass
        try:
            file.write(f"NODES={resources["nodes"]}\n")
        except KeyError:
            pass
        try:
            file.write(f"NTASKS_PER_NODE={resources["ntasks_per_node"]}\n")
        except KeyError:
            pass
        try:
            file.write(f"NBANDS_PER_TASK={resources["nbands_per_task"]}\n")
        except KeyError:
            pass
        try:
            file.write(f"NBANDS={resources["nbands"]}\n")
        except KeyError:
            pass
