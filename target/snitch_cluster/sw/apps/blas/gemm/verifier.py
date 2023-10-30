import random
import time
import subprocess

import numpy as np
import os


def random_inputs():
    f = open("inc/main.h", "w")
    M = random.randint(1, 50)
    N = random.randint(1, 50)
    K = random.randint(1, 50)
    f.write("#define M " + str(M) +
            "\n#define N " + str(N) +
            "\n#define K " + str(K) + "\n")
    f.write("#ifndef DATA_TYPE\n" +
            "#define DATA_TYPE float\n" +
            "#endif\n")

    rand = random.randint(0, 1)
    transA = rand
    f.write("int transA=" + str(rand) + ";\n")

    rand = random.randint(0, 1)
    transB = rand
    f.write("int transB=" + str(rand) + ";\n")

    if transA:
        A = np.empty((K, M))
    else:
        A = np.empty((M, K))
    if transB:
        B = np.empty((N, K))
    else:
        B = np.empty((K, N))
    C = np.empty((M, N))

    f.write("DATA_TYPE A[]={")
    (I,J) = (K,M) if transA else (M,K)
    for i in range(0, I):
        for j in range(0, J):
            rand = np.float32(random.uniform(-1, 1))
            A[i][j] = rand
            if i == I - 1 and j == J - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    (I,J) = (N,K) if transB else (K,N)
    f.write("DATA_TYPE B[]={")
    for i in range(0, I):
        for j in range(0, J):
            rand = np.float32(random.uniform(-1, 1))
            B[i][j] = rand
            if i == I - 1 and j == J - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    f.write("DATA_TYPE C[]={")
    for i in range(0, M):
        for j in range(0, N):
            rand = np.float32(random.uniform(-1, 1))
            C[i][j] = rand
            if i == M - 1 and j == N - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    f.write("DATA_TYPE Y[M*N];\n")


    rand = random.uniform(-100, 100)
    alpha = round(np.float32(rand),4)
    f.write("float alpha=" + str(alpha) + ";\n")

    rand = random.uniform(-100, 100)
    beta = round(np.float32(rand),4)
    f.write("float beta=" + str(beta) + ";\n")

    return A, B, C, alpha, beta, transA, transB


def golden_model(A, B, C, alpha, beta, transA, transB):
    if transA:
        A = np.transpose(A)
    if transB:
        B = np.transpose(B)
    return np.float32(alpha * np.matmul(A, B) + beta * C)


if __name__ == '__main__':
    f=open("outputs/comparison.txt","w")
    for i in range(0,100):
        print("Iteration ",i)
        A, B, C, alpha, beta, transA, transB = random_inputs()
        Y = golden_model(A, B, C, alpha, beta, transA, transB)

        compile_process = subprocess.call('./exec.sh', shell=True)
        with open("outputs/output.txt", "r") as file:
            # Read the entire contents of the file
            file_content = file.read()
        numbers_as_strings = file_content.split(",")
        Y_c_file = [np.float32(number) for number in numbers_as_strings]
        f.write("EXP\tREAL\n")
        for exp, real in zip(np.ravel(Y), Y_c_file):
            absolute_diff = round(abs(abs(round(exp, 4))-abs(round(real, 4))),4)

            if absolute_diff > 0.0001:
                f.write(str(absolute_diff)+"\t"+str(round(exp, 4))+"\t"+str(round(real, 4))+"\n")
                # exit()
        f.write("---------------------------\n")
