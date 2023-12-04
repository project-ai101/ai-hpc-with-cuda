# CUDA Thread, Warp, Block and Stream
A NVidia GPU consists of an array of SM (Stream Multiprocessor). For example, a Nvidia RTX 3060 GPU has 28 SMs. 
Each SM has many CUDA cores which actually perform the instruction level computation. For example, a Nvidia RTX 3060 SM
has 128 CUDA cores. All these cores are scheduled by the Cuda Thread, Warp, Block and Stream framework. 
