import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

plt.title("32 channels, fixed 8x8 input matrix")
plt.xlabel("# of elements in kernel")
plt.ylabel("# of cycles")

x = np.array([4, 8, 16, 32, 64])

y_ref = np.array([6940.125, 5449.5, 4282, 3891.25, 3491.625])
y_new = np.array([12825.375, 8889, 6432.25, 5519.625, 4653.875])
y_ssr = np.array([17041.125, 8871, 4836.75, 2542, 1706.125])

plt.xlim(left=0, right=x[-1] + 1)
plt.ylim(bottom=0, top=max(y_ref.max(), y_new.max(), y_ssr.max()) + 300)

plt.xticks(x)

plt.plot(x, y_ref, "ro", label="Reference Impl")
plt.plot(x, y_new, "go", label="New Impl")
plt.plot(x, y_ssr, "bo", label="New Impl with SSR/FREP")

x_ref_curve = np.linspace(x.min(), x.max(), 300)

plt.plot(x_ref_curve, make_interp_spline(x, y_ref, k=2)(x_ref_curve), "r--")

plt.plot(x_ref_curve, make_interp_spline(x, y_new, k=2)(x_ref_curve), "g--")

plt.plot(x_ref_curve, make_interp_spline(x, y_ssr, k=2)(x_ref_curve), "b--")

plt.legend()

plt.savefig("fixed_input.png")