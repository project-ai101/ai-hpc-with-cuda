# MPI Overview
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Author: Bin Tan

MPI (Message Passing Interface) is a well known distributed computation framework for HPC for 
many years. One implementation is [Open MPI](https://www.open-mpi.org). In this tutorial, 
basic and fundamental concepts are discussed and several examples are developed to help
understand the concepts in depth. They are grouped into following 6 categories

- Communicator and Rank
- Group and InterCommunicator
- Connector and Acceptor
- Blocking and Barrier
- Point-to-Point Communication
- Collective Operations

### Communicator and Rank
In a distributed computation world, how to communicate among the processes is essential. 
In MPI, Communicator is the object representing the network which connects a group 
computation processes. The id of each process is given and differentiated by the rank. 

The default and also the biggest communicator is MPI_COMM_WORLD which contains all the
processes for an application. This default communicator is created when the mpirun command
is executed. For example,

```
$ mpirun -n 2 -host node1:1,node2:1 example_application
```
This command shall create two processes. One is on the node1 and one is on the node2. Each 
process shall execute example_application. The following code shows how to access the MPI_COMM_WORLD
within the example_application

```cpp
    int nRanks;    // the size of the default communicator
    int myRank;    // the rank of this process in the default communicator

    MPI_init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
```
After the rank is identified, example_application knows which process it is running in.

### Group and InterCommunicator

### Connector and Acceptor

### Blocking and Barrier

### Point-to-Point Communication

### Collective Operations
