import numpy as np

def regular_kpoint_mesh(cell, scale:int=1, kpts_max:int|None=None):
    vector_lengths = np.linalg.norm(cell, axis=1)
    ratios = vector_lengths.max() / vector_lengths
    sampling = np.max(np.stack([scale*ratios, np.ones(3)], axis=-1), axis=1)
    sampling = np.round(sampling, 0).astype(int)

    if kpts_max and kpts_max < sampling.max():
        kpts = np.round(sampling/sampling.max()*kpts_max)
        kpts = np.array([k if k > 1 else 1 for k in kpts])
    else:
        kpts = sampling

    return kpts
