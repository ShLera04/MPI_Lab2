#include <iostream>
#include <mpi.h>
#include <vector>

std::vector<int> matrixMultiplicate(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int maxSize, int count)
{
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < maxSize; j++) {
            C[i * maxSize + j] = 0;
            for (int k = 0; k < maxSize; k++) {
                C[i * maxSize + j] += A[i * maxSize + k] * B[k * maxSize + j];
            }
        }
    }
    return C;
}
std::vector<int> matrixMultiplicationParallel(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int maxSize, int rank, int size) {
    int portion = maxSize / size;
    int remainder = maxSize % size;
    std::vector<int> count, sendcount, displ;
    count.resize(size);
    sendcount.resize(size);
    displ.resize(size);
    for (int i = 0; i < size; i++)
    {
        count[i] = portion;
        if (i < remainder)
        {
            count[i]++;
        }
    }
    for (int i = 0; i < size; i++)
    {
        int tmp = count[i] * maxSize;
        sendcount[i] = tmp;
    }
    int _displ = 0;
    for (int i = 0; i < size; i++) {
        displ[i] = _displ;
        _displ += sendcount[i];
    }

    std::vector<int> localA(sendcount[rank]);

    MPI_Scatterv(A.data(), sendcount.data(), displ.data(), MPI_INT, localA.data(), sendcount[rank], MPI_INT, 0, MPI_COMM_WORLD);


    MPI_Bcast(B.data(), maxSize * maxSize, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> localC(sendcount[rank]);
    /* for (int i = 0; i < count[rank] && count[rank] != 0; i++) {
         for (int j = 0; j < maxSize; j++) {
             localC[i * maxSize + j] = 0;
             for (int k = 0; k < maxSize; k++) {
                 localC[i * maxSize + j] += localA[i * maxSize + k] * B[k * maxSize + j];
             }
         }
     }*/
    localC = matrixMultiplicate(localA, B, localC, maxSize, count[rank]);

    MPI_Gatherv(localC.data(), sendcount[rank], MPI_INT, C.data(), sendcount.data(), displ.data(), MPI_INT, 0, MPI_COMM_WORLD);
    return C;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int maxSize = 3;
    std::vector<int> A, B, C;
    A.resize(maxSize * maxSize);
    B.resize(maxSize * maxSize);
    C.resize(maxSize * maxSize);
    if (rank == 0) {
        A = { 2,4,6,8,10,12,14,16,18 };
        B = { 1,3,5,7,9,11,13,15,17 };

        for (int i = 0; i < maxSize * maxSize; i++)
        {
            C[i] = 0;
        }
    }
    matrixMultiplicationParallel(A, B, C, maxSize, rank, size);
    if (rank == 0) {
        std::cout << "Matrix C:" << std::endl;
        for (int i = 0; i < maxSize; i++) {
            for (int j = 0; j < maxSize; j++) {
                std::cout << C[i * maxSize + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}

