from spectral_spatial_simulation import Model as SpectralSpatialModel
import matplotlib.pyplot as plt
from file import Console
from tqdm import tqdm
import numpy as np
import torch

# Class FourierTransform
def cartesian_FT(model: SpectralSpatialModel):
    Console.printf_section("Sampling -> FFT")

    #print(f"mmemap hape: {model.volume.shape}")

    # TODO: Maybe better: just select one time point from memmap, transform to tensor -> to cuda -> fft -> back to numpy memmap and next!
    volume: np.memmap = model.volume


    # Be aware mmemap has shape (x,y,z,1,time)
    print(volume.shape)
    print(volume.squeeze(axis=-2).shape)
    input("============================")


    if torch.cuda.is_available():

        indent = " " * 22
        for t in tqdm(range(volume[0,0,0,:].size), desc=indent + "3D fft of each time point (on CUDA)"):
            volume_part = volume[:, :, :, 0, t]
            volume[:, :, :, 0, t] = torch.fft.fftn(torch.from_numpy(volume[:, :, :, 0, t]).cuda()).cpu().numpy()

       #Console.printf("info", "fft of volume")
       #indent = " " * 22
       #for t, _ in enumerate(tqdm(volume[0, 0, 0, :], desc=indent + "3D fft of each time point (on CUDA)")):  # total=volume[0,0,0,:].size(), # volume[0,0,0,:].size())
       #    #volume_part = torch.from_numpy(volume[:, :, :, t]).to(device='cuda')
       #    #volume_part = torch.fft.fftn(volume_part)
       #    #volume[:, :, :, t] = volume_part.to(device='cpu').numpy()
       #    volume[:, :, :, t] = torch.fft.fftn(torch.from_numpy(volume[:, :, :, t])).numpy()



        #volume_part = volume[:, :, :, t].to(device='cpu')
        # volume[:, :, :, t] = torch.fft.fftn(volume[:, :, :, t])
        #volume_part = volume[:, :, :, t].to(device='cpu')


    #volume = torch.squeeze(torch.from_numpy(model.volume)) # TODO


    #if torch.cuda.is_available():
    #    space_required_mb_cuda = np.prod(model.volume.shape) * np.dtype(model.volume.dtype).itemsize * 1 / (1024 * 1024)
    #    space_free_cuda = torch.cuda.mem_get_info()[0] * 1 / (1024 * 1024)  # free memory on GPU [MB]
    #    space_total_cuda = torch.cuda.mem_get_info()[1] * 1 / (1024 * 1024)  # total memory on GPU [MB]
    #    Console.printf("info", f"Require total on GPU [MB]: {space_required_mb_cuda}. Free [MB]: {space_free_cuda}")
#
    #    if space_required_mb_cuda <= space_free_cuda:
    #    else
#
#
    #    if False:#space_required_mb_cuda <= space_free_cuda:
    #        Console.printf("info", f"Start putting whole tensor on GPU ({torch.cuda.get_device_name(0)})")
    #        Console.start_timer()
    #        volume = volume.to(device='cuda')
    #        Console.stop_timer()
    #        Console.printf("success", f"Put tensor to GPU {torch.cuda.get_device_name(0)}")
#
    #        Console.printf("info", "Start spatial FFT")
    #        indent = " " * 22
    #        for t, _ in enumerate(tqdm(volume[0, 0, 0, :], desc=indent + "3D fft of each time point (on CUDA)")):  # total=volume[0,0,0,:].size(), # volume[0,0,0,:].size())
    #            volume[:, :, :, t] = torch.fft.fftn(volume[:, :, :, t])
#
    #        Console.printf("success", f"Transformed volume to frequency domain. Shape: {volume.shape}")
#
    #    else:
    #        Console.printf("info", f"Just putting stepwise all parts of tensor on GPU ({torch.cuda.get_device_name(0)})")
#
    #        indent = " " * 22
    #        for t, _ in enumerate(tqdm(volume[0, 0, 0, :], desc=indent + "3D fft of each time point (on CUDA)")):  # total=volume[0,0,0,:].size(), # volume[0,0,0,:].size())
    #            volume_part = volume[:, :, :, t].to(device='cpu')
    #            volume_part_fft = torch.fft.fftn(volume_part).to(device='cpu')
    #            volume[:, :, :, t] = volume_part_fft
#
    #        Console.printf("success", f"Transformed volume to frequency domain. Shape: {volume.shape}")
    #        # just put the parts required on GPU
#
#
    #else:
    #    Console.printf("info", "No supported GPU available")



    ##volume: np.memmap = model.volume
    #volume = torch.squeeze(torch.from_numpy(model.volume)) # now torch tensor! & remove dimension: [x,y,z,1,t] -> [x,y,z,t]
#
    #if torch.cuda.is_available():
    #    Console.printf("info", f"Start putting tensor on GPU ({torch.cuda.get_device_name(0)})")
    #    Console.start_timer()
    #    volume = volume.to(device='cuda')
    #    Console.stop_timer()
    #    Console.printf("success", f"Put tensor to GPU {torch.cuda.get_device_name(0)}")
#
    #else:
    #    Console.printf("info", "No supported GPU available")
#
    #Console.printf("info", "Start spatial FFT")
    #indent = " " * 22
    #for t, _ in enumerate(tqdm(volume[0,0,0,:], desc=indent + "3D fft of each time point (on CUDA)")): # total=volume[0,0,0,:].size(), # volume[0,0,0,:].size())
    #    volume[:,:,:,t] = torch.fft.fftn(volume[:,:,:,t])
#
    #Console.printf("success", f"Transformed volume to frequency domain. Shape: {volume.shape}")
#
    #volume = volume.cpu()
    #plt.imshow(torch.abs(volume[:,50, :, 1]))
    #plt.show()
#
#
    ## take all x,y,z,1 and t=0
    ##volume[x,y,z,1,t]
#
    ## FT of each time point
    ## FT of each metabolite!
    ## (x,y,z,t)

def non_cartesian_FT():
    pass
# def non_cartesian

# Requirement:
# -> Cartesian FFT
# -> Non cartesian FFT