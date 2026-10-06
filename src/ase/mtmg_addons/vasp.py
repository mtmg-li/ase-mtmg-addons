def regular_kpoint_mesh(cell, scale=1):
    vector_lengths = np.linalg.norm(cell, axis=1)
    ratios = vector_lengths.max() / vector_lengths
    sampling = np.max(np.stack([scale*ratios, np.ones(3)], axis=-1), axis=1)
    sampling = np.round(sampling, 0).astype(int)
    return sampling
