/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <iostream>
#include <mpi.h>

#define HANDLE_ERR(err_msg) \
        std::cout << err_msg << std::endl; \
        exit_status = EXIT_FAILURE; \
        goto Exit_Entry

int main(int argc, char** argv) {
    int exit_status = EXIT_SUCCESS;

    int world_size = 0;
    int world_rank = MPI_UNDEFINED;
    int sub_group_id = MPI_UNDEFINED;
    int sub_group_rank = MPI_UNDEFINED;;

    // The default group associated with the default communicator, MPI_COMM_WORLD 
    MPI_Group mpi_group_world;
    MPI_Info mpi_info_world;
    MPI_Errhandler mpi_err_handler_world;

    // The ranks in the mpi_group_world to form two disjoint groups
    int group_a_ranks[] {0, 1, 2};
    int group_b_ranks[] {3, 4, 5};

    // two disjoint groups and communicators
    MPI_Group group_a = MPI_GROUP_NULL;
    MPI_Group group_b = MPI_GROUP_NULL;
    MPI_Comm intercomm_ab = MPI_COMM_NULL;

    int send_val = 1;
    int recv_val = 0;

    MPI_Status recv_status;
    int comm_compare_result;

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (MPI_Comm_group(MPI_COMM_WORLD, &mpi_group_world) != MPI_SUCCESS) {
        HANDLE_ERR("Failed to retrieve the default group");
    }

    if (MPI_Comm_get_info(MPI_COMM_WORLD, &mpi_info_world) != MPI_SUCCESS) {
        HANDLE_ERR("Failed to get the default info");
    }

    if (MPI_Comm_get_errhandler(MPI_COMM_WORLD, &mpi_err_handler_world) != MPI_SUCCESS) {
        HANDLE_ERR("Failed to get the default err handler");
    }

    if (MPI_Group_incl(mpi_group_world, 3, group_a_ranks, &group_a) != MPI_SUCCESS) {
        HANDLE_ERR("Failed to form the group_a");
    }
    if (MPI_Group_incl(mpi_group_world, 3, group_b_ranks, &group_b) != MPI_SUCCESS) {
        HANDLE_ERR("Failed to form the group_b");
    }

    if (MPI_Group_rank(group_a, &sub_group_rank) != MPI_SUCCESS) {
        HANDLE_ERR("Failed to get sub_group_rank from group_a");
    }

    if (sub_group_rank == MPI_UNDEFINED) {
        if (MPI_Group_rank(group_b, &sub_group_rank) != MPI_SUCCESS) {
            HANDLE_ERR("Failed to get sub_group_rank in group_b");
        }
        sub_group_id = 1;
    } else {
        sub_group_id = 0;
    }

    if (sub_group_rank == MPI_UNDEFINED) {
        HANDLE_ERR("Failed to get the sub_group_rank");
    }

    if (sub_group_id == 0) {
        if (MPI_Intercomm_create_from_groups(group_a, 2,
                                         group_b, 2,
                                         "21",
                                         mpi_info_world,
                                         mpi_err_handler_world,
                                         &intercomm_ab) != MPI_SUCCESS) {
            HANDLE_ERR("Failed to create the inter communicator linking local group_a to remote group_b");
        }
    } else {
        if (MPI_Intercomm_create_from_groups(group_b, 2,
                                         group_a, 2,
                                         "21",
                                         mpi_info_world,
                                         mpi_err_handler_world,
                                         &intercomm_ab) != MPI_SUCCESS) {
            HANDLE_ERR("Failed to create the inter communicator linking local group_b to remote group_a");
        }
    }

    if (intercomm_ab == MPI_COMM_NULL) {
        std::cout << "The process " << world_rank
                  << " has null intercomm_ab in sub group " << sub_group_id
                  << " with sub group rank " << sub_group_rank << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (sub_group_id == 0) {
        // group a is considered as the sending domain
        send_val += sub_group_rank;
        // use my sub_group_rank as the dest rank in the remote group
        // use the same tag in the intercomm_ab creation
        if (MPI_Send(&send_val, 1, MPI_INT, sub_group_rank, 21, intercomm_ab) != MPI_SUCCESS) {
            HANDLE_ERR("Failed to send data from group_a to group_b");
        }
    } else {
        // group b is considered as the receiving domain
        // expect the process with the same rank, sub_group_rank, in the sending group, group_a
        // use the same tage in the intercomm_ab creation
        if (MPI_Recv(&recv_val, 1, MPI_INT, sub_group_rank, 21, intercomm_ab, &recv_status) != MPI_SUCCESS) {
            HANDLE_ERR("Failed to recv data from group_a to group_b");
        }

        // now let's check if the received values do match expected
        // first to make sure all processes in the receiving group, group_b, completed.
        std::cout << "The process " << world_rank
                  << " has received value " << recv_val
                  << " from the process with sub group rank " << sub_group_rank
                  << " from the sub group " << sub_group_id
                  << std::endl;
    }

Exit_Entry:
    if (intercomm_ab != MPI_COMM_NULL) {
        MPI_Comm_free(&intercomm_ab);
    }

    if (group_a != MPI_GROUP_NULL) {
        MPI_Group_free(&group_a);
    }

    if (group_b != MPI_GROUP_NULL) {
        MPI_Group_free(&group_b);
    }

    MPI_Finalize();

    return exit_status;
}
