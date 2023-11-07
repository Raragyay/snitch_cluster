import random
import time
import subprocess

import numpy as np
import os

import torch
from torch import nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.layer(x)
        return y


def random_inputs():
    f = open("inc/main.h", "w")

    M = 1
    N = random.randint(1, 1000)
    K = random.randint(1, 1000)

    f.write("#define M " + str(1) +
            "\n#define N " + str(N) +
            "\n#define K " + str(K) + "\n")
    f.write("#ifndef DATA_TYPE\n" +
            "#define DATA_TYPE float\n" +
            "#endif\n")

    I = np.empty((M, K))
    W = np.empty((K, N))
    B = np.empty((M, N))
    E = np.empty((M, N))

    f.write("DATA_TYPE I[]={")
    for i in range(0, M):
        for j in range(0, K):
            rand = round(np.float32(random.uniform(-1, 1)), 4)
            I[i][j] = rand
            if i == M - 1 and j == K - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    f.write("DATA_TYPE W[]={")
    for i in range(0, K):
        for j in range(0, N):
            rand = round(np.float32(random.uniform(-1, 1)), 4)
            W[i][j] = rand
            if i == K - 1 and j == N - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    f.write("DATA_TYPE B[]={")
    for i in range(0, M):
        for j in range(0, N):
            rand = round(np.float32(random.uniform(-1, 1)), 4)
            B[i][j] = rand
            if i == M - 1 and j == N - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    f.write("DATA_TYPE E[]={")
    for i in range(0, M):
        for j in range(0, N):
            rand = round(np.float32(random.uniform(-1, 1)), 4)
            E[i][j] = rand
            if i == M - 1 and j == N - 1:
                f.write(str(rand) + "};\n")
            else:
                f.write(str(rand) + ", ")

    rand = random.uniform(0.0001, 0.1)
    e = round(np.float32(rand), 4)
    f.write("float e=" + str(e) + ";\n")

    return I, W, B, E, e, N, K


def golden_model(I, W, B, E, e, K, N):
    model = NeuralNetwork(K, N)
    with torch.no_grad():
        for i in range(N):
            model.layer.weight[i] = torch.from_numpy(np.transpose(W))[i]
        for i in range(N):
            model.layer.bias[i] = torch.from_numpy(np.transpose(B))[i]

    real = model.forward(torch.from_numpy(I).type(torch.FloatTensor))

    model.zero_grad()
    optimizer = optim.SGD(model.parameters(), lr=e)
    real.backward(gradient=(torch.from_numpy(E).type(torch.FloatTensor)))

    with torch.no_grad():
        optimizer.step()
    return model.layer.weight, model.layer.bias
    # print("BIAS\n",model.layer.bias)
    # print("WEIGHT\n", torch.transpose(model.layer.weight,1,0))
    #
    # I = torch.from_numpy(I).type(torch.FloatTensor)
    # W = torch.from_numpy(W).type(torch.FloatTensor).requires_grad_()
    # B = torch.from_numpy(B).type(torch.FloatTensor).requires_grad_()
    # E = torch.from_numpy(E).type(torch.FloatTensor)
    #
    # D = torch.matmul(I, W)+B # This is your GEMM operation
    # D.backward(gradient=E)
    # with torch.no_grad():
    #     B -= e*B.grad
    #     W -= e*W.grad
    # print(B)
    # print(W)


if __name__ == '__main__':
    f = open("outputs/comparison.txt", "w")
    for i in range(0, 50):
        print("Iteration", i)
        I, W, B, E, e, N, K = random_inputs()
        W_golden, B_golden = golden_model(I, W, B, E, e, K, N)

        compile_process = subprocess.call('./exec.sh', shell=True)
        with open("outputs/output.txt", "r") as file:
            # Read the entire contents of the file
            file_content = file.read()
        sections = file_content.split(",\n\n")
        weights_as_strings = sections[0].split(",")
        W_real = [np.float32(number) for number in weights_as_strings]
        biases_as_strings = sections[1].split(",")
        B_real = [np.float32(number) for number in biases_as_strings]
        f.write("ABS_DIFF\tEXP\tREAL\n")
        f.write("WEIGHS\n")
        for exp, real in zip(np.ravel(torch.transpose(W_golden, 1, 0).detach().numpy()), W_real):
            absolute_diff = round(abs(abs(round(exp, 4)) - abs(round(real, 4))), 4)
            if absolute_diff > 0.0001:
                f.write(str(absolute_diff)+str(exp)+"\t"+str(real)+"\n")
        f.write("BIASES\n")
        for exp, real in zip(np.ravel(B_golden.detach().numpy()), B_real):
            absolute_diff = round(abs(abs(round(exp, 4)) - abs(round(real, 4))), 4)
            if absolute_diff > 0.0001:
                f.write(str(absolute_diff)+str(exp)+"\t"+str(real)+"\n")
        f.write("---------------------------\n")

