# Project Name: Synthetic MRSI

## Description
This project is dedicated to simulating **Magnetic Resonance Spectroscopy Imaging (MRSI)** processes and replicating the artifacts commonly introduced by MR devices. The core objective is to simulate the reverse process of the imaging system, transitioning from the image and spectral domain back to the non-Cartesian k-space.
The project is written in _Python_ and primarily uses the _Dask_ library and CUDA for acceleration.

## Current Status
🚧 This project is under active development.

## Hardware Requirements
- **CPU**: ............ Not yet specified.
- **RAM**: ........... Not yet specified.
- **Memory**: .... Not yet specified.
- **GPU**:
  - NVIDIA CUDA: .... Version 12.0+ (for GPU support)
  - VRAM: .................. At least one GPU with 32 GB VRAM (e.g., NVIDIA A100)


## This project includes
* Prepared [docker image](https://hub.docker.com/r/boraborapalm/synthetic_mrsi)
* Conda environment
* Jupyter Lab
  
* The python modules (.py)
* A main jupyter notebook

<dl>
<dd>
      
> [!NOTE]
> The container alrady includes the installed conda enviroment and the jupyter lab. The python modules in the background are used and imported in the main notebook. The conda environment can also be installed separately, and respective yml file can be found in the env/MRSI_simulation.yml. 
> The MRSI_simulation.yml should be the same as installed in the latest docker container.
</dd>
</dl>

## This project includes not
* A completed paths file.

<dl>
<dd>
    
> [!NOTE]
> Please check the template regrading the required structure: config/paths_template.json. Further, make sure that the given units are conform with the python library [pint](https://pint.readthedocs.io/en/stable/).

</dd>
</dl>

## How to use
1. **Clone the repository:**
    ```bash
    git clone https://github.com/MRSI-HFMR-Group-Vienna/Synthetic_MRSI
    cd Synthetic_MRSI
    ```

2. **Start the docker container either on a cluster (e.g., HPC cluster) or a local workstation:**

    For local workstation:
    ```bash
    docker run boraborapalm/synthetic_mrsi:<tag> -p XXXX:8888 --gpus all -v /path/to/cloned/Synthetic_MRSI /bin/bash 
    ```

    <dl>
    <dd>

        
    > Check the latest docker image [here](https://hub.docker.com/r/boraborapalm/synthetic_mrsi). Further, make sure that at least one GPU is forwarded (all with --gpus all), the necessary paths to this project are forwarded and that the jupyter lab port 8888 is forwarded from the inside of the container to the ouside in order to access it from outside (-p XXXX:8888, with XXXX beeing a free port ousite the container). 
    

    </dd>
    </dl>


    For hcp cluster: <br>
    
    &nbsp;&nbsp;&nbsp;&nbsp; If you using a slurm workload manager, the check out the following: [Link](https://slurm.schedmd.com/overview.html).

    <dl>
    <dd>
        
    > Further, make sure that the CPUs are not dynamically allocated since it might results in an issue regrading the GPU <-> CPU transfer via dask (e.g., [slurm](https://docs.rapids.ai/api/dask-cuda/nightly/troubleshooting/))
    </dd>
    </dl>

3. **Start jupyter lab:**
   ```bash
    jupyter lab   --no-browser   --ip=0.0.0.0   --port=8888   --allow-root   --notebook-dir=/
    ```

4. **Set up paths.json:**

    <dl>
    <dd>
        
   Use the config/paths_template.json to insert your paths and also insert units if possible that are conform with [pint](https://pint.readthedocs.io/en/stable/).
   Otherwise, insert "unknown".
       
    </dd>
    </dl>

5. **Run the jupyter notebook:**
    <dl>
    <dd>
        
   Open the main_notebook.ipynb and specify the GPUs _target_gpu_smaller_tasks_, _target_gpus_big_tasks_, where the first one is an integer and the lst one an list of integers. Further, the device_memory_limit can be set and also the desired blocksize, which dask uses. Please note: since this project is under active developement, things might change.

       
    </dd>
    </dl>


## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.



## Contact

For further inquiries or support, contact <br>
> **Markus Schuster** at markus.schuster[at]meduniwien.ac.at <br>
> **Berhard Strasser** at bernhard.strasser[at]meduniwien.ac.at <br>
> **Alireza Vasfi** (Ali) at alireza.vasfi[at]meduniwien.ac.at <br>
> **Wolfgang Bogner** at wolfgang.bogner[at]meduniwien.ac.at
