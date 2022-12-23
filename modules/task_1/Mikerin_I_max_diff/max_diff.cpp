// Copyright 2022 Mikerin Ilya
#include <mpi.h>
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include "../../../modules/task_1/Mikerin_I_max_diff/max_diff.h"


int* getRandomVector(int  sz) {
    std::random_device dev;
    std::mt19937 gen(dev());
    int* vec = new int[sz];
    for (int  i = 0; i < sz; i++) { vec[i] = gen() % 100; }
    return vec;
}

int getSequentialOperations(int* vec, int sz) {
    int max_diff = 0;
    for ( int i = 0; i < sz-1; i++ ) {
        if ( abs(vec[i]-vec[i+1]) > max_diff)
            max_diff = abs(vec[i]-vec[i+1]);
    }

    return max_diff;
}

int getParallelOperations(int* global_vec, int count_size_vector) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int delta = count_size_vector / size;
    const int remainder = count_size_vector % size;

    if (rank == 0) {
        for (int proc = 1; proc < size-1; proc++) {
            MPI_Send(global_vec + proc * delta + remainder, delta+1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
        MPI_Send(global_vec + (size-1) * delta + remainder, delta, MPI_INT, (size-1), 0, MPI_COMM_WORLD);
    }

    int global_max_diff = 0;

    if (rank == size-1) {
        int* local_vec = new int[delta];
        MPI_Status status;
        MPI_Recv(local_vec, delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int local_max_diff = getSequentialOperations(local_vec,delta);

        MPI_Reduce(&local_max_diff, &global_max_diff, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            int* local_vec = new int[delta+1 + remainder];
            for (int i = 0; i < delta+ 1 + remainder; i++) {
                local_vec[i] = global_vec[i];
            }
            int local_max_diff = getSequentialOperations(local_vec,delta+1 + remainder);
            MPI_Reduce(&local_max_diff, &global_max_diff, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        } else {
            int* local_vec = new int[delta+1];
            MPI_Status status;
            MPI_Recv(local_vec, delta + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            int local_max_diff = getSequentialOperations(local_vec,delta+1);
            MPI_Reduce(&local_max_diff, &global_max_diff, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        }
    }

    return global_max_diff;
}

