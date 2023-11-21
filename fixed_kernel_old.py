import matplotlib.pyplot as plt
import numpy as np

plt.title("8 channels, fixed 4x4 kernel")
plt.xlabel("# of elements in input matrix")
plt.ylabel("# of cycles")

x = np.array([64, 256, 1024])

y_ref = np.array([1471.25, 5857.625, 23426.125])
y_new = np.array([1723.5, 6319.75, 24486.875])
y_ssr = np.array([1228.75, 4727, 19402.25])

plt.xlim(left=0, right=x[-1] + 50)
plt.ylim(bottom=0, top=max(y_ref.max(), y_new.max(), y_ssr.max()) + 500)

xlims = plt.xlim()

x_ex = np.insert(x, 0, xlims[0])
x_ex = np.append(x_ex, xlims[1])

y_ref_line = np.polyfit(x, y_ref, 1)
y_ref_ex = np.insert(y_ref, 0, 0)
y_ref_ex = np.append(y_ref_ex, np.polyval(y_ref_line, xlims[1]))

plt.plot(x, y_ref, "ro", label="Reference Impl")
plt.plot(np.unique(x_ex), np.poly1d(np.polyfit(x_ex, y_ref_ex, 1))(np.unique(x_ex)), "r--")

y_new_line = np.polyfit(x, y_new, 1)
y_new_ex = np.insert(y_new, 0, 0)
y_new_ex = np.append(y_new_ex, np.polyval(y_new_line, xlims[1]))

plt.plot(x, y_new, "go", label="New Impl")
plt.plot(np.unique(x_ex), np.poly1d(np.polyfit(x_ex, y_new_ex, 1))(np.unique(x_ex)), "g--")

y_ssr_line = np.polyfit(x, y_ssr, 1)
y_ssr_ex = np.insert(y_ssr, 0, 0)
y_ssr_ex = np.append(y_ssr_ex, np.polyval(y_ssr_line, xlims[1]))

plt.plot(x, y_ssr, "bo", label="New Impl with SSR/FREP")
plt.plot(np.unique(x_ex), np.poly1d(np.polyfit(x_ex, y_ssr_ex, 1))(np.unique(x_ex)), "b--")

plt.legend()

plt.savefig("fixed_kernel.png")