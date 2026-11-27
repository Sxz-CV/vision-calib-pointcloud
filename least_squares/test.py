import numpy as np

pts = np.array([[0,1], [1,3],[2,4],[3,4]], dtype=float)
x, y = pts[:, 0], pts[:, 1]

A = np.column_stack([np.ones_like(x), x])
b = y

sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print("a, b=", sol)
print("残差平方和 =", res)
print("秩 =", rank, "奇异值 =", sv)

r = b - A @ sol
print("Aᵀr =", A.T @ r)