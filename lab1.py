import numpy as np
from itertools import combinations
from scipy.optimize import  minimize

# Матрицы из условия (прямоугольные)
mid_A = np.array([[0.95, 1.00], [1.05, 1.00], [1.10, 1.00]])

rad_A_t = np.array([[1.00, 1.00], [1.00, 1.00], [1.00, 1.00]])   # томография
rad_A_r = np.array([[1.00, 0.00], [1.00, 0.00], [1.00, 0.00]])   # регрессия

mid_A_qv =np.array([[1.1,.9,1.1],[1.4,1,.8],[.8,1.4,1.2]])
rad_A_qv = np.ones([3,3])


PAIRS = list(combinations(range(3), 2))      # (0,1), (0,2), (1,2)


def solve(M, R, n_starts=50, seed=42):
    n = M.shape[1]

    def objective(x):
        x = np.asarray(x, dtype=float)
        R_abs_x = R @ np.abs(x)
        if np.any(R_abs_x <= 0):
            return 1e10
        return np.max(np.abs(M @ x) / R_abs_x)

    rng = np.random.default_rng(seed)
    best_delta, best_x = np.inf, None

    for _ in range(n_starts):
        x0 = rng.normal(size=n) * 2
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
        if res.fun < best_delta:
            best_delta, best_x = res.fun, res.x


    if res.fun < best_delta:
        best_delta, best_x = res.fun, res.x

    return best_delta, best_x / np.max(np.abs(best_x))


def build_singular(M, R, delta):
    m, n = M.shape

    def obj(e):
        A = M + e.reshape(m, n) * R
        if m == n:
            return np.linalg.det(A) ** 2
        # прямоугольная: сумма квадратов всех максимальных миноров
        return sum(np.linalg.det(A[list(rows)]) ** 2
                   for rows in combinations(range(m), n))

    res = minimize(obj, np.zeros(M.size), method='SLSQP',
                   bounds=[(-delta, delta)] * M.size,
                   options={'ftol': 1e-16, 'maxiter': 1000})
    if not res.success:
        raise RuntimeError(res.message)
    E = res.x.reshape(m, n)
    return M + E * R, E, res.fun


def check(A, M, R, delta, tol=1e-8):
    m, n = A.shape
    scale = np.max(np.abs(A))
    eps = 1e-9 * max(1.0, scale)
    if m == n:
        vals = np.array([np.linalg.det(A)])
    else:
        vals = np.array([np.linalg.det(A[list(p)])
                         for p in combinations(range(m), n)])
    singular = bool(np.all(np.abs(vals) < tol * scale ** n))
    inside = bool(np.all(np.abs(A - M) <= delta * R + eps))
    return singular, inside





def main():
    print("=" * 50)
    print("Матрица 3x2")
    print("=" * 50)

    for R, name in [(rad_A_t, "Томография"), (rad_A_r, "Регрессия")]:
        delta, x = solve(mid_A, R)
        A, E, fun = build_singular(mid_A, R, delta)
        singular, inside = check(A, mid_A, R, delta)
        print(f"\n{name}")
        print(f"  delta* = {delta:.9f}")
        print(f"  x      = {np.round(x, 6)}")
        print(f"  A' =\n{np.round(A, 9)}")
        print(f"  невязка = {fun:.2e}")
        print(f"  особенная: {singular},  внутри A: {inside}")

    print("\n" + "=" * 50)
    print("Матрица 3x3")
    print("=" * 50)
    delta3, x3 = solve(mid_A_qv, rad_A_qv)
    A3, E3, fun_3 = build_singular(mid_A_qv, rad_A_qv, delta3)
    singular3, inside3 = check(A3, mid_A_qv, rad_A_qv, delta3)
    print(f"  delta* = {delta3:.9f}")
    print(f"  x      = {np.round(x3, 6)}")
    print(f"  A' =\n{np.round(A3, 9)}")
    print(f"  особенная: {singular3},  внутри A: {inside3}")
    print(f"Определитель: {np.linalg.det(A3)}")
    print("=" * 50)

if __name__ == "__main__":
    main()