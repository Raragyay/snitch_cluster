import random

M = 1
N = 10
K = 5
f = open("inc/main.h", "w")

f.write("#define M " + str(M) +
        "\n#define N " + str(N) +
        "\n#define K " + str(K) + "\n")

f.write("#ifndef DATA_TYPE\n" +
        "#define DATA_TYPE float\n" +
        "#endif\n")

f.write("DATA_TYPE I[]={")
for i in range(0, M * K - 1):
    f.write(str(random.uniform(-1, 1)) + ",")
f.write(str(random.uniform(-1, 1)) + "};\n")

f.write("DATA_TYPE W[]={")
for i in range(0, K * N - 1):
    f.write(str(random.uniform(-1, 1)) + ",")
f.write(str(random.uniform(-1, 1)) + "};\n")

f.write("DATA_TYPE B[]={")
for i in range(0, M * N - 1):
    f.write(str(random.uniform(-1, 1)) + ",")
f.write(str(random.uniform(-1, 1)) + "};\n")

f.write("DATA_TYPE E[M*N];\n")

f.write("DATA_TYPE S[]={")
for i in range(0, M * N - 1):
    f.write(str(random.randint(0, 1)) + ".0,")
f.write(str(random.randint(0, 1)) + ".0};\n")

f.write("DATA_TYPE Y[M*N];\n")
f.write("float e=0.001;\n")
