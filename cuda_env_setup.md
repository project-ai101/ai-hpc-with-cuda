# Setup Nvidia CUDA Environment

The platform is Ubuntu 22.04.

### Check Nvidia GPU device
To use cuBLAS APIs, a Nvidia GPU device with tensor cores shall be available. To check if there is a Nvidia GPU card attached, 
use the following command,

```
$ lspci | grep -i Nvidia
```

The example output looks like

```
$ 04:00.0 VGA compatible controller: NVIDIA Corporation GA104 [GeForce RTX 3060] (rev a1)
$ 04:00.1 Audio device: NVIDIA Corporation GA104 High Definition Audio Controller (rev a1)
```

### Check and Install Nvidia Driver
Now, use the following command to check if a proper Nvidia driver has been installed. 

```
$ nvidia-smi
```

The output should look like

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:04:00.0 Off |                  N/A |
|  0%   50C    P8              18W / 170W |      9MiB / 12288MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1091      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

If the command is not found, the driver can be downloaded from [Nvidia Driver Download](https://www.nvidia.com/en-us/geforce/drivers/) site.

### Check and Install CUDA Toolkit

Use the following command to check if a CUDA Toolkit has been installed,

```
$ nvcc --version
```

The output should look like

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```

If nvcc command does not exist, use following command to install a CUDA toolkit

```
sudo apt install nvidia-cuda-toolkit
```

Now the environment is properly setup and ready for CUDA programming!
