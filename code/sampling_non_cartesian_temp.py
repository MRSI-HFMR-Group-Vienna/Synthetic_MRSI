
# =============== ABOUT MAIN sampling.py ==================
# Model

# -> has already apply_coil_sensitivity_maps
# -> cartesian FFT
# -> cartesian IFFT
# -> apply_gaussian_noise
# -> coil_combination
# =========================================================

# --> non cartesian trajectories FT quite complex -> thus extra class?

# trajectory
# density compensation
# simulate cartesian trajectory
# Operator class (?) with:
#     -> trajectory_in
#     -> trajectory_out

# in tools.py
# NUFT (slow FT)
# density compensation

class DummyClass:
    def __init__(self):
        self.traj_data = None

    def prepare_crt_trajectories(apply_girf: bool = True):
        #if apply_girf
        # Apply girf
        # Interpolate to ADC
        # Normalise CRTTrajectoriesMAT
        # Calculate k-space radii
        # maybe post processing...
        pass

