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

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int TOTAL_SUB_GROUPS = 3;
    int SUB_GROUP_DIVIDER = world_size / TOTAL_SUB_GROUPS;
    MPI_Comm my_sub_comm;

    // create three disjoint subgroups and sub-communicators.
    int sub_group_id = world_rank / SUB_GROUP_DIVIDER;
    int sub_group_rank = world_rank % SUB_GROUP_DIVIDER;
    // broadcast the value within each group.
    int bc_val = 1;

    // ranks shall be excluded from the base group to form a new group 
    int exclusive_ranks[2] {0, 1};
    // the base group of MPI_COMM_WORLD
    MPI_Group base_group;
    // the sub group formed by the difference between the base group and exclusive_ranks
    MPI_Group diff_sub_group;
    // the sub communicator which shall be associated with the sub_group_0_1_union
    MPI_Comm diff_sub_comm;
    // my rank in hte diff sub group
    int my_diff_sub_rank = MPI_UNDEFINED;

    // Create three disjoint groups to verify that the collective function MPI_Bcast
    // only performs on a specific sub communicator associated with the specific sub
    // group.
    if (MPI_Comm_split(MPI_COMM_WORLD,
                       sub_group_id,
                       sub_group_rank,
                       &my_sub_comm) != MPI_SUCCESS) {
        std::cout << "Failed to invoke MPI_Comm_split" << std::endl;
        goto Exit_Entry;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (MPI_Comm_group(MPI_COMM_WORLD, &base_group) != MPI_SUCCESS) {
            std::cout << "Failed to invoke base_group" << std::endl;
            goto Exit_Entry;
    }


    // make sure the initial values are difference between the root rank
    // in each sub communicator and other ranks.
    if (sub_group_rank == 0) {
        // the root in the sub communicator.
        bc_val = sub_group_id + 1 + bc_val;
    }
    MPI_Bcast(&bc_val, 1, MPI_INT, 0, my_sub_comm);

    std::cout << "The process " << world_rank
              << " in sub group " << sub_group_id
              << " with sub group rank " << sub_group_rank
              << " received broadcast value " << bc_val
              << std::endl;

    // now, use MPI_Group_excl to create a new group
    if (MPI_Group_excl(base_group, 2, exclusive_ranks, &diff_sub_group) != MPI_SUCCESS) {
        std::cout << "Failed to invoke MPI_Group_excl" << std::endl;
        goto Exit_Entry;
    }

    if (MPI_Group_rank(diff_sub_group, &my_diff_sub_rank) != MPI_SUCCESS) {
        std::cout << "Failed to invoke MPI_Group_rank" << std::endl;
        goto Exit_Entry;
    }

    if (my_diff_sub_rank != MPI_UNDEFINED) {
        // this process is a member of the sub_group_0_1_union
        if (MPI_Comm_create_group(MPI_COMM_WORLD, diff_sub_group, 20, &diff_sub_comm) != MPI_SUCCESS) {
            std::cout << "Failed to create a sub communicator for diff sub group" << std::endl;
            goto Exit_Entry;
        }
    }

    // reset the broadcast value
    bc_val = 1;
    if (my_diff_sub_rank == 0) {
        bc_val = 101;
    }

    if (my_diff_sub_rank != MPI_UNDEFINED) {
        if (MPI_Bcast(&bc_val, 1, MPI_INT, 0, diff_sub_comm) != MPI_SUCCESS) {
            std::cout << "Failed to broadcast the value" << std::endl;
            goto Exit_Entry;
        }
    }

    // make sure the separation line is printed between two sets of outputs
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "-----------------------------------------------" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_diff_sub_rank != MPI_UNDEFINED) {
        std::cout << "The process " << world_rank
                  << " in sub group " << sub_group_id
                  << " with sub group rank " << sub_group_rank
                  << " and with new sub group rank " << my_diff_sub_rank
                  << " in sub group 0_1 union received broadcast value "
                  << bc_val << std::endl;
    } else {
        std::cout << "Process " << world_rank << " is not in the diff sub group" << std::endl;
    }
Exit_Entry:
    MPI_Finalize();

    return EXIT_SUCCESS;
}
