# Project Name: Synthetic MRSI

### Description
This project is dedicated to simulating **Magnetic Resonance Spectroscopy Imaging (MRSI)** processes and replicating the artifacts commonly introduced by MR devices. The core objective is to simulate the reverse process of the imaging system, transitioning from the image and spectral domain back to the non-Cartesian k-space.
The project is written in _Python_ and primarily uses the _Dask_ library and CUDA for acceleration.

### Current Status
This project is under active development.

---

## How to Use

### 1. Installation

#### Prerequisites:<br>
- **Operating System**: Tested on Linux
- **NVIDIA CUDA**: Version 12.0+ (for GPU support)

#### Steps:<br>
1. **Clone the repository**:
    ```bash
    git clone https://github.com/MRSI-HFMR-Group-Vienna/Synthetic_MRSI
    cd Synthetic_MRSI
    ```
    
2. **Install Anaconda**:<br>
   Follow the instructions at the following link to install Anaconda: 
   https://docs.anaconda.com/anaconda/install/

3. **Set up the environment**:<br>
    Navigate to the environment folder: 
    ```bash
    cd Synthetic_MRSI/environment
    ```
    Choose the latest environment version and create the conda environment (e.g., `MRSI_simulation_5.yml`):
    ```bash
    conda env create -f MRSI_simulation.yml
    ```
    Activate the environment:
    ```bash
    conda activate MRSI_simulation
    ```
    > ℹ️
    > If creating the environment from the `.yml` file is slow with `conda`, you can speed it up by using **mamba**.  
    > Install mamba in the base environment and then create the environment with:
    >
    > ```bash
    > conda install -n base -c conda-forge mamba
    > mamba env create -f MRSI_simulation.yml
    > ```



### 2. How to Start
Once the environment is set up, you can start the simulation. Please note that the project is still under development.

#### To start the simulation:
```bash
cd code
python main.py
```

### 3. What to Consider
- **Hardware Requirements**:
    - **CPU**: Not yet specified.
    - **GPU**: At least one NVIDIA GPU with CUDA 12.0+
    - **RAM**: Not yet specified.
    - **Memory**: Not yet specified.
- **Performance Tips**:
    - Use the GPU whenever possible to accelerate the simulation. Even better to use multiple GPUs!


---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Contact

For further inquiries or support, contact <br>
> **Markus Schuster** at markus.schuster[at]meduniwien.ac.at <br>
> **Berhard Strasser** at bernhard.strasser[at]meduniwien.ac.at <br>
> **Alireza Vasfi** (Ali) at alireza.vasfi[at]meduniwien.ac.at <br>
> **Wolfgang Bogner** at wolfgang.bogner[at]meduniwien.ac.at
