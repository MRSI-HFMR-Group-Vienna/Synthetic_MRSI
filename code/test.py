import torch

def get_least_used_GPU() -> torch.cuda.device:
    num_devices = torch.cuda.device_count()

    free_space_devices = []
    for device in range(num_devices):
        torch.cuda.mem_get_info(device)

        space_total_cuda = torch.cuda.mem_get_info(device)[1]
        space_free_cuda = torch.cuda.mem_get_info(device)[0]
        percentage_free_space = space_free_cuda/space_total_cuda
        free_space_devices.append(percentage_free_space)
        print(f"Free space on GPu {device}: {space_free_cuda / (1024**2)} / {space_total_cuda / (1024**2)}")

    index_most_free_space = free_space_devices.index(max(free_space_devices))

    print(f"On GPU {index_most_free_space} most free space at moment!")
    return torch.cuda.device(index_most_free_space)


if __name__ == "__main__":
    get_gpu_free_memory()
