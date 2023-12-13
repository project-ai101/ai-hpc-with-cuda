/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <iostream>
#include <mpi.h>


int main(int argc, char** argv) {
    int world_size;
    int world_rank;

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Get_processor_name(host_name, &name_len);
    std::cout << "Hello world from host " << host_name
              << " with rank " << world_rank
              << " and communicator size " << world_size
              << " processes"
              << std::endl;

    MPI_Finalize();

    return EXIT_SUCCESS;
}
