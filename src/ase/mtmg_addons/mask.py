import numpy as np


class Mask(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array)
        if obj.dtype != bool:
            raise RuntimeError("Mask dtype can only be bool")
        return obj.view(cls)

    def indices(self):
        return np.flatnonzero(self)

    @classmethod
    def from_indices(cls, atoms, indices):
        if not isinstance(indices, (list, tuple, np.ndarray)):
            indices = [indices]
        return cls([i in indices for i in range(len(atoms))])

    def encode_indices(self, indices: list[int] | int):
        contained_indices = self.indices()
        # Single index handler
        if isinstance(indices, int):
            if indices not in contained_indices:
                raise RuntimeError("Provided index is not within the mask")
            return np.where(indices == contained_indices)[0][0]

        # Index list handler
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices)
        if not all(np.isin(indices, contained_indices)):
            raise RuntimeError("Provided indices do not all lie within mask")
        locations = np.asarray([np.where(i == contained_indices)[0][0] for i in indices])
        return locations

    def decode_indices(self, indices: list[int] | int):
<<<<<<< HEAD
        return self.indices()[indices]"
=======
        return self.indices()[indices]
>>>>>>> 5558597 (Initial commit of pre-existing functions and classes)
