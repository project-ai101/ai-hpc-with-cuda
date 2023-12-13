# MPI Overview
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Author: Bin Tan

MPI (Message Passing Interface) is a well known distributed computation standard for HPC for 
many years. One popular implementation is [Open MPI](https://www.open-mpi.org). In this tutorial, 
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
$ mpirun -n 2 -host node1:1,node2:1 ./mpi_hello_world
```
This command shall create two processes. One is on the node1 and one is on the node2. Each 
process shall execute application mpi_hello_world. The following code shows how to access the MPI_COMM_WORLD
within the example_application

```cpp
    int nRanks;    // the size of the default communicator
    int myRank;    // the rank of this process in the default communicator

    MPI_init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
```
After the rank is identified, example_application can know which process it is running with. The option
"-n 2" means mpirun shall create two processes and the option "-host node1:1,node2:1" tells mpirun to
create one process on each node1 and node2. Here is the code of the simple [Hello World example](./mpi_hello_world.cpp).

### Group and InterCommunicator
The default communicator, MPI_COMM_WORLD, contains all processes created by the mpirun command. For point-to-point communication,
it seems sufficient. However, for collective operation communication, in many situations, all processes  is not
preferrable. Therefore, a concept to capture a subset of all processes is desirable. The new concept is called group. 
A group is defined as an ordered set of process identifiers and each process associated with the group is assigned an additional
integer id as rank of the process in the group. This means that the process could have multiple ranks, for example, one is associated
with a group and another is in the default communicator, MPI_COMM_WORLD. 

There are two pre-defined groups, MPI_GROUP_EMPTY and MPI_GROUP_NULL. MPI_GROUP_EMPTY is a valid group with no processes 
and MPI_GROUP_NULL is an invalid group.

The key difference between group and communicator is that group is a conceptual representation of a set of processes and does not
support any communication functionality like a communicator. To allow processes within a group be able to communicate to each other
like a MPI communicator, a communicator shall be created with the group. The API to create a 
new MPI communicator associated with the group is

```c
     MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm* newconn)
```

One important thing in MPI is that a group can not be created from scratch. It has to be created (or formed) from an exist
group via group (aka set) operations. There is a base group created by the MPI, which is associated with the 
default communicator, MPI_COMM_GROUP. It can be retrieved by API,

```c
     MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
```

The above two APIs reflect one important implementation fact which is that for a MPI communicator, there always is a group associated with it and
can be retrieved by MPI_Comm_group. However, for a group, there may not be a MPI communicator associated with it. If so, it has to
explicitly create one via MPI_Comm_create_group.

This is [an example](./mpi_group_example.cpp) to demonstrate above discussion about the group concept.


### Connector and Acceptor

### Blocking and Barrier

### Point-to-Point Communication

### Collective Operations
