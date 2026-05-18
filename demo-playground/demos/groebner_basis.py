import alkahest as ak
from alkahest import GroebnerBasis, solve, latex, eval_expr
import time

pool = ak.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# System: unit circle intersected with the line x + y = 1
# Solutions are (1, 0) and (0, 1)
f1 = x**2 + y**2 + pool.integer(-1)   # x^2 + y^2 - 1 = 0
f2 = x + y + pool.integer(-1)          # x   + y   - 1 = 0

print("Polynomial system:")
print("  f1 = x^2 + y^2 - 1")
print("  f2 = x + y - 1")

# ---

t0 = time.perf_counter()
gb = GroebnerBasis.compute([f1, f2], [x, y])
ms = (time.perf_counter() - t0) * 1e3

print("Grobner basis computed in " + str(round(ms, 3)) + " ms")
print("  generators: " + str(len(gb)))
print("  f1 in ideal: " + str(gb.contains(f1)))
print("  f2 in ideal: " + str(gb.contains(f2)))

# ---

sols = solve([f1, f2], [x, y])
print(str(len(sols)) + " solutions found:")
for i, s in enumerate(sols, 1):
    xv = round(eval_expr(s[x], {}), 8)
    yv = round(eval_expr(s[y], {}), 8)
    residual = abs(xv**2 + yv**2 - 1)
    tag = "ok" if residual < 1e-9 else "FAIL"
    print("  [" + str(i) + "] x=" + str(xv) + "  y=" + str(yv) + "  residual=" + str(round(residual, 2)) + " [" + tag + "]")
