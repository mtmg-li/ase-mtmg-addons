from ase.atoms import Atoms
from ase.geometry import get_distances
from scipy.optimize import linear_sum_assignment
import numpy as np


def best_fit_axis(
    atoms: Atoms,
    positive_component: int | None = None
):
    centroid = np.mean(atoms.positions, axis=0)
    _, _, Vt = np.linalg.svd(atoms.positions - centroid)
    n = Vt[0],
    if positive_component and n[positive_component] < 0:
        n *= -1.0
    return n


def best_fit_plane(
    atoms: Atoms,
    positive_component: int | None = None
):
    centroid = np.mean(atoms.positions, axis=0)
    _, _, Vt = np.linalg.svd(atoms.positions - centroid)
    n = Vt[-1]
    if positive_component and n[positive_component] < 0:
        n *= -1.0
    return n


def gen_chain_mask(
    atoms: Atoms,
    start_index: int,
    search_radius: float,
    stop_at: list[str] | str = ["H"],
    exclude: list[str] | str | None = None
):
    chain_indices = [start_index]
    if not isinstance(stop_at, list):
        stop_at = [stop_at]
    for current_index in chain_indices:
        # First check if this is a termination point
        if atoms[current_index].symbol in stop_at:
            continue

        # Get the distances from the current index
        _, distances = get_distances(atoms.positions[current_index], atoms.positions, atoms.cell, True)
        distances = distances[0]

        # Convert to a mask for easier manipulation
        selection_mask = Mask(distances <= search_radius)
        selection_mask[current_index] = False

        # Get the symbols indices for later comparison
        selection_symbols = atoms[selection_mask].symbols

        # For everything that was selected
        for i, s in zip(selection_mask.indices(), selection_symbols):
            # See if it is already in the list, if so, ignore it
            if i in chain_indices:
                selection_mask[i] = False

        # Add the selection to the chain indices
        for i in selection_mask.indices():
            chain_indices.append(int(i))

    # Now, exclude anything from the selection that shouldn't be there
    if exclude is not None:
        # Cast as a list if it isn't, this allows for simpler single-species selection
        if not isinstance(exclude, list):
            exclude = [exclude]
        # Check everything in the chain indices against the list
        # This kind of paradigm is dangerous, but works because it only removes
        for i in chain_indices:
            if atoms[i].symbol in exclude:
                chain_indices.remove(i)

    return Mask([i in chain_indices for i in range(len(atoms))])


def match_atoms(
    atoms_1: Atoms,
    atoms_2: Atoms,
    triplets: dict[int, int] | None = None,
    manual: dict[int, int] | None = None
):
    """
    Returns an array `mapping` such that mapping[i] gives the index in atoms2
    corresponding to atom i in atoms1, found via optimal assignment.
    Matching is done per-chemical-species and respects PBC if cell is periodic.
    """
    n1, n2 = len(atoms_1), len(atoms_2)
    if n1 != n2:
        raise ValueError("Atoms objects must have the same number of atoms")

    mapping = np.full(n1, -1, dtype=int)
    symbols_1 = np.array(atoms_1.get_chemical_symbols())
    symbols_2 = np.array(atoms_2.get_chemical_symbols())

    # Use atoms1's cell/pbc as the reference for minimum-image convention
    cell = atoms_1.cell
    pbc = atoms_1.pbc

    # Get copies to work with instead of originals
    atoms_1, atoms_2 = atoms_1.copy(), atoms_2.copy()

    # Align the cells based on the triplets
    if triplets is not None:
        # Create the normal vector to the plane defined by the three in each Atoms
        triplet_positions = np.asarray([
            [atoms_1.positions[i] for i in triplets.keys()],
            [atoms_2.positions[i] for i in triplets.values()]]
        )
        normal_vectors = np.asarray([
            np.cross(*(np.diff(triplet_positions[0], axis=0).T*[1, -1]).T),
            np.cross(*(np.diff(triplet_positions[1], axis=0).T*[1, -1]).T)
        ])
        centroids = triplet_positions.mean(axis=1)
        # Rotate the first atoms object to match the second
        atoms_1.rotate(normal_vectors[0], normal_vectors[1], centroids[0])
        # Translate the geometric center of the first to match the second
        atoms_1 = translate_atoms(atoms_1, np.diff(centroids, axis=0))

    for symbol in set(symbols_1):
        idx1 = np.where(symbols_1 == symbol)[0]
        idx2 = np.where(symbols_2 == symbol)[0]

        if len(idx1) != len(idx2):
            raise ValueError(f"Species count mismatch for {symbol}: "
                             f"{len(idx1)} vs {len(idx2)}")

        positions_1 = atoms_1.positions[idx1]
        positions_2 = atoms_2.positions[idx2]

        # Pairwise distance matrix respecting periodic boundaries
        _, dist_matrix = get_distances(positions_1, positions_2, cell=cell, pbc=pbc)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        mapping[idx1[row_ind]] = idx2[col_ind]

    return mapping


def reorder_atoms(atoms1: Atoms, atoms2: Atoms, triplet: dict[int, int] | None = None):
    """Return a copy of atoms2 reordered to best match atoms1's atom order."""
    mapping = match_atoms(atoms1, atoms2, triplet)
    return atoms2[mapping]


def rotate_atoms_to_plane(
    atoms:Atoms,
    plane_atoms:Atoms,
    alignment_vector:tuple[float,float,float],
    center:tuple[float,float,float],
    mask:list[bool]|None = None
):
    all_atoms = atoms.copy()

    if mask is not None:
        mask_indices = np.flatnonzero(mask)
        atoms = atoms[mask]
    else:
        atoms = atoms.copy()

    v1 = np.array(plane_atoms[0].position) - np.array(plane_atoms[1].position)
    v2 = np.array(plane_atoms[2].position) - np.array(plane_atoms[1].position)
    plane_vector = np.cross(v1, v2)
    plane_unit_vector = plane_vector/np.linalg.norm(plane_vector)

    alignment_vector = np.array(alignment_vector)
    alignment_unit_vector = alignment_vector/np.linalg.norm(alignment_vector)

    angle = np.acos(np.dot(plane_unit_vector, alignment_unit_vector))

    if angle > np.pi/2:
        plane_vector *= -1.0
        plane_unit_vector *= -1.0
        angle -= np.pi/2

    atoms.rotate(
        a=plane_unit_vector,
        v=alignment_unit_vector,
        center=center
    )

    if mask is not None:
        for i, p in zip(mask_indices, atoms.positions):
            all_atoms.positions[i] = p
    else:
        all_atoms.positions = atoms.positions

    return all_atoms


def translate_atoms(
    atoms: Atoms,
    displacement: tuple[float, float, float],
    mask: list[bool] | None = None
):
    all_atoms = atoms.copy()

    if mask is not None:
        # Use the nx3 displacement method
        displacement = np.vstack([displacement if m else np.zeros(3) for m in mask])

    all_atoms.translate(displacement)

    return all_atoms
