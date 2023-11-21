import matplotlib.pyplot as plt
import numpy as np

plt.title("8 channels, fixed 4x4 kernel")
plt.xlabel("# of elements in input matrix")
plt.ylabel("# of cycles")

x = np.array([64, 256, 1024])

y_ref = np.array([1471.25, 5857.625, 23426.125])
y_new = np.array([1723.5, 6319.75, 24486.875])
y_ssr = np.array([1228.75, 4727, 19402.25])

plt.xlim(left=0, right=x[-1] + 20)
plt.ylim(bottom=0, top=max(y_ref.max(), y_new.max(), y_ssr.max()) + 500)

plt.xticks(x)

plt.plot(x, y_ref, "ro", label="Reference Impl")
plt.plot(x, y_new, "go", label="New Impl")
plt.plot(x, y_ssr, "bo", label="New Impl with SSR/FREP")

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y_ref, 1))(np.unique(x)), "r--")
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y_new, 1))(np.unique(x)), "g--")
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y_ssr, 1))(np.unique(x)), "b--")

plt.legend()

plt.savefig("fixed_kernel.png")