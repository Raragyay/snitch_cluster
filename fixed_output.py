import matplotlib.pyplot as plt
import numpy as np

plt.title("8 channels, fixed 4x4 output matrix")
plt.xlabel("# of elements in input matrix")
plt.ylabel("# of cycles")

x = np.array([64, 256, 1024, 1536])

y_ref = np.array([3326.375, 5857.625, 15538, 22058.375])
y_new = np.array([3238.75, 6319.75, 17606.375, 25581])
y_ssr = np.array([4121.125, 4727, 6468.125, 7492.125])

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

plt.savefig("fixed_output.png")