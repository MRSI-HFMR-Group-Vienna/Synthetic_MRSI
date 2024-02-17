import default
import numpy as np



class Model:
    # TODO
    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_mask(self):
        # TODO. Create masks is in the simulator. Right place there?
        raise NotImplementedError("This method is not yet implemented")

    def add_T1_image(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_subject_variability(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_pathological_alterations(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


class MetabolicAtlas:
    # TODO
    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def transform_to_T1(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def load(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


class MetabolicMap:
    """
    This includes the data of the metabolic map as numpy.array, thus the concentration at each spatial position.
    """
    def __init__(self):
        self.concentration: np.ndarray = None
        self.name: str = None
        self.unit: str = None


class Simulator:
    # TODO

    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def transform_metabolic_atlas_to_T1(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def create_masks(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")




if __name__ == "__main__":
    pass