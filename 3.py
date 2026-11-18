"""
Лабораторная работа №3 по интервальному анализу. Вариант 9.

Исследование разрешимости линейной задачи о допусках (ЛЗД) для ИСЛАУ A_i x = b_i
с помощью распознающего функционала Tol и коррекция систем (b-, A-, Ab-коррекции).
Номера формул и теорем соответствуют отчёту.
"""
import itertools
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import intvalpy as ip
from scipy.optimize import linprog

DELTA = 0.10          # запас по Tol после минимальной коррекции
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Данные варианта 9: интервалы хранятся как [..., 2] = (нижний конец, верхний конец)
# ----------------------------------------------------------------------------
_A_rows = [
    [[0.65, 1.25], [0.70, 1.30]],
    [[0.75, 1.35], [0.75, 1.31]],
    [[0.80, 1.40], [0.67, 1.27]],
    [[-0.30, 0.30], [0.81, 1.35]],
]
_b_rows = [[9.75, 9.751], [9.85, 9.851], [9.90, 9.901], [4.80, 4.801]]

SYSTEMS = {
    1: (np.array(_A_rows[:2]), np.array(_b_rows[:2])),
    2: (np.array(_A_rows[:3]), np.array(_b_rows[:3])),
    3: (np.array(_A_rows[:4]), np.array(_b_rows[:4])),
}


# ----------------------------------------------------------------------------
# Базовые характеристики интервалов
# ----------------------------------------------------------------------------
def mid(X):
    return (X[..., 0] + X[..., 1]) / 2


def rad(X):
    return (X[..., 1] - X[..., 0]) / 2


def mag(X):
    return np.maximum(np.abs(X[..., 0]), np.abs(X[..., 1]))


def make(lo, hi):
    return np.stack([lo, hi], axis=-1)


def to_ip(X):
    return ip.Interval(X[..., 0], X[..., 1])


def chi(X):
    """Функционал Рачека (4); для нулевого интервала не определён (nan)."""
    lo, hi = X[..., 0], X[..., 1]
    out = np.full(lo.shape, np.nan)
    nz = ~((lo == 0) & (hi == 0))
    a = np.where(np.abs(lo) <= np.abs(hi), lo, hi)
    c = np.where(np.abs(lo) <= np.abs(hi), hi, lo)
    out[nz] = a[nz] / c[nz]
    return out


# ----------------------------------------------------------------------------
# Распознающий функционал Tol, формула (2)
#   Tol(x) = min_i { rad b_i - | mid b_i - sum_j a_ij x_j | },
#   где |.| — магнитуда интервала; для интервала sum_j a_ij x_j
#   mid = sum_j mid(a_ij) x_j,  rad = sum_j rad(a_ij) |x_j|.
# ----------------------------------------------------------------------------
def tol_rows(X, A, b):
    X = np.asarray(X, float)
    s_mid = X @ mid(A).T
    s_rad = np.abs(X) @ rad(A).T
    return rad(b) - (np.abs(mid(b) - s_mid) + s_rad)


def tol(X, A, b):
    return tol_rows(X, A, b).min(axis=-1)


def _tol_max_lp(A, b, weight):
    """Контрольная максимизация: Tol — вогнутая кусочно-линейная функция,
    её максимум находится точно задачей линейного программирования."""
    mA, rA, mb, rb = mid(A), rad(A), mid(b), rad(b)
    m, n = mA.shape
    w = np.ones(m) if weight is None else np.asarray(weight, float)
    c = np.zeros(2 * n + 1)
    c[-1] = -1.0                       # переменные: x (n), u = |x| (n), t
    rows, rhs = [], []
    for i in range(m):
        for s in (1.0, -1.0):          # t <= w_i (rad b_i - s(mid b_i - mA_i x) - rA_i u)
            r = np.zeros(2 * n + 1)
            r[:n] = -s * w[i] * mA[i]
            r[n:2 * n] = w[i] * rA[i]
            r[-1] = 1.0
            rows.append(r)
            rhs.append(w[i] * (rb[i] - s * mb[i]))
    for j in range(n):
        for s in (1.0, -1.0):          # u_j >= s x_j
            r = np.zeros(2 * n + 1)
            r[j], r[n + j] = s, -1.0
            rows.append(r)
            rhs.append(0.0)
    res = linprog(c, A_ub=rows, b_ub=rhs, bounds=[(None, None)] * (2 * n + 1),
                  method="highs")
    return res.x[:n], -res.fun


def maximize_tol(A, b, weight=None):
    """Глобальный максимум Tol: ralgb5 (intvalpy) + контроль методом ЛП."""
    x_r, f_r, *_ = ip.linear.Tol.maximize(to_ip(A), to_ip(b), weight=weight)
    x_r = np.asarray(x_r, float)
    x_l, f_l = _tol_max_lp(A, b, weight)
    assert abs(float(f_r) - f_l) < 1e-6, (f_r, f_l)
    x, f = (x_l, f_l) if f_l >= float(f_r) else (x_r, float(f_r))
    x = np.where(np.abs(x) < 1e-9, 0.0, x)
    return x, float(f)


def generators(x, A, b, tol_eps=1e-7):
    """Образующие: строки, на которых достигается минимум в точке arg max Tol."""
    r = tol_rows(x, A, b)
    return [int(i) + 1 for i in np.where(r - r.min() <= tol_eps)[0]]


def calT(A, b):
    """Характеристика совместности ЛЗД T(A, b) = max Tol0, формула (3), веса 1/|b_i|."""
    return maximize_tol(A, b, weight=1.0 / mag(b))[1]


def theorem_147(A, b):
    """Теорема 2: (i) 0 not in b_i, (ii) max_j chi(a_ij) < chi(b_i) для всех i
    => допусковое множество пусто."""
    cA, cb = chi(A), chi(b)
    cond1 = bool(np.all((b[:, 0] > 0) | (b[:, 1] < 0)))
    cond2 = bool(np.all(np.nanmax(cA, axis=1) < cb))
    return cond1 and cond2, np.nanmax(cA, axis=1), cb


# ----------------------------------------------------------------------------
# Коррекции
# ----------------------------------------------------------------------------
def widen_b(b, K):
    """Равномерное уширение правой части: b + K e, e = ([-1,1],...)^T (формула (6))."""
    return make(b[:, 0] - K, b[:, 1] + K)


def narrow_A(A, E):
    """A ⊖ E, E = ([-e_ij, e_ij]) — внутреннее вычитание (формула (7)):
    [a_ij - (-e_ij), a̅_ij - e_ij]; mid сохраняется, rad уменьшается на e_ij."""
    E = np.broadcast_to(np.asarray(E, float), A.shape[:-1])
    return make(A[..., 0] + E, A[..., 1] - E)


# ----------------------------------------------------------------------------
# Допусковое множество: точный многоугольник (в каждом ортанте —
# пересечение полуплоскостей  A_lo x >= b_lo,  A_hi x <= b_hi)
# ----------------------------------------------------------------------------
def _clip(poly, a, c):
    """Отсечение выпуклого многоугольника полуплоскостью a·x <= c."""
    out = []
    n = len(poly)
    for k in range(n):
        P, Q = poly[k], poly[(k + 1) % n]
        fp, fq = a @ P - c, a @ Q - c
        if fp <= 0:
            out.append(P)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append(P + t * (Q - P))
    return out


def xi_tol_polygons(A, b, center, half=200.0):
    mA, rA = mid(A), rad(A)
    polys = []
    for s in itertools.product((1.0, -1.0), repeat=2):
        s = np.array(s)
        cx, cy = center
        poly = [np.array(p) for p in [(cx - half, cy - half), (cx + half, cy - half),
                                      (cx + half, cy + half), (cx - half, cy + half)]]
        cons = [(-s[j] * np.eye(2)[j], 0.0) for j in range(2)]          # s_j x_j >= 0
        for i in range(mA.shape[0]):
            cons.append((-(mA[i] - rA[i] * s), -b[i, 0]))  # нижний конец (Ax)_i >= b_i
            cons.append((mA[i] + rA[i] * s, b[i, 1]))      # верхний конец (Ax)_i <= b̅_i
        for a, c in cons:
            poly = _clip(poly, a, c)
            if not poly:
                break
        if len(poly) >= 3:
            P = np.array(poly)
            area = 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) -
                             np.dot(P[:, 1], np.roll(P[:, 0], 1)))
            if area > 0:
                polys.append(P)
    return polys


def poly_area(polys):
    tot = 0.0
    for P in polys:
        tot += 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) -
                         np.dot(P[:, 1], np.roll(P[:, 0], 1)))
    return tot


# ----------------------------------------------------------------------------
# Анализ одной системы (шаги A.1 — A.3 в числах)
# ----------------------------------------------------------------------------
def analyze(A, b):
    x, T = maximize_tol(A, b)
    res = dict(argmax=x.tolist(), Tmax=T, gens=generators(x, A, b),
               rows=tol_rows(x, A, b).tolist(), calT=calT(A, b))
    if T >= 0:
        polys = xi_tol_polygons(A, b, x)
        res["area"] = poly_area(polys)
        V = np.vstack(polys)
        res["bbox"] = [V[:, 0].min(), V[:, 0].max(), V[:, 1].min(), V[:, 1].max()]
        res["_polys"] = polys
    else:
        res["area"] = 0.0
        res["_polys"] = []
    return res


def rel_width_change(A0, b0, A1, b1):
    """Изменение относительной ширины 1 - chi (4)-(5) по элементам данных."""
    dA = np.abs((1 - chi(A1)) - (1 - chi(A0)))
    db = np.abs((1 - chi(b1)) - (1 - chi(b0)))
    return float(np.nansum(dA)), float(np.nansum(db))


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
METHOD_NAMES = {"b": "$b$-коррекция", "A": "$A$-коррекция", "Ab": "$Ab$-коррекция"}
METHOD_COLORS = {"b": "#1f77b4", "A": "#2ca02c", "Ab": "#d62728"}


def plot_surface(ax, A, b, x1r, x2r, cmap=None, color=None, alpha=0.8, n=120):
    X1, X2 = np.meshgrid(np.linspace(*x1r, n), np.linspace(*x2r, n))
    Z = tol(np.stack([X1, X2], -1), A, b)
    if cmap is not None:
        ax.plot_surface(X1, X2, Z, cmap=cmap, alpha=alpha, linewidth=0, antialiased=True)
    else:
        ax.plot_surface(X1, X2, Z, color=color, alpha=alpha, linewidth=0)
    return Z


def window(points, polys, pad=0.6, min_half=1.0):
    pts = [np.atleast_2d(p) for p in points] + [P for P in polys]
    V = np.vstack(pts)
    lo, hi = V.min(0), V.max(0)
    c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2 * (1 + pad), min_half)
    return (c[0] - h[0], c[0] + h[0]), (c[1] - h[1], c[1] + h[1])


def main():
    results = {}
    for s, (A, b) in SYSTEMS.items():
        R = {}
        # ---------------- исходная система ----------------
        orig = analyze(A, b)
        tau = np.array(orig["argmax"])
        T = orig["Tmax"]
        th_ok, th_maxchiA, th_chib = theorem_147(A, b)
        orig["thm147"] = dict(holds=th_ok, max_chi_A=th_maxchiA.tolist(),
                              chi_b=th_chib.tolist())
        orig["Delta"] = float(np.min(rad(A) @ np.abs(tau)))           # формула (9)
        R["orig"] = orig

        # ---------------- B.1: b-коррекция ----------------
        K_b = abs(T) + DELTA
        b_b = widen_b(b, K_b)
        rb = analyze(A, b_b)
        rb.update(K=K_b, e=0.0, K_A=0.0, K_b=K_b)
        rb["dchiA"], rb["dchib"] = rel_width_change(A, b, A, b_b)
        rb["_A"], rb["_b"] = A, b_b
        R["b"] = rb

        # ---------------- B.2: A-коррекция ----------------
        e_need = abs(T) / np.abs(tau).sum()          # из (8b) при e_ij = e, K = |T|
        e_max = float(rad(A).min())                  # из (8a)
        A_u = narrow_A(A, e_max)                     # максимально допустимое равномерное сужение
        r_u = analyze(A_u, b)
        A_pt = narrow_A(A, rad(A))                   # предельный случай e_ij = rad a_ij
        r_pt = analyze(A_pt, b)
        rA = r_pt
        rA.update(e_need=e_need, e_max=e_max, uniform=dict(
            argmax=r_u["argmax"], Tmax=r_u["Tmax"], gens=r_u["gens"]),
            K_A=r_pt["Tmax"] - T, K_b=0.0, feasible=r_pt["Tmax"] >= 0)
        if rA["feasible"]:
            rA["dchiA"], rA["dchib"] = rel_width_change(A, b, A_pt, b)
            # наименьшее сужение вида e_ij = min(e, rad a_ij), дающее max Tol >= 0
            lo_e, hi_e = e_max, float(rad(A).max())
            for _ in range(60):
                e_mid = (lo_e + hi_e) / 2
                if maximize_tol(narrow_A(A, np.minimum(e_mid, rad(A))), b)[1] >= 0:
                    hi_e = e_mid
                else:
                    lo_e = e_mid
            rA["e_clip_min"] = hi_e
            rA["rad_left"] = float((rad(A) - np.minimum(hi_e, rad(A))).max())
        rA["_A"], rA["_b"] = A_pt, b
        R["A"] = rA

        # ---------------- B.3: Ab-коррекция ----------------
        T_A = r_u["Tmax"]                             # этап 1: сужение A (e = e_max)
        K_ab = abs(T_A) + DELTA                       # этап 2: уширение b
        b_ab = widen_b(b, K_ab)
        rab = analyze(A_u, b_ab)
        rab.update(e=e_max, K=K_ab, T_after_A=T_A, K_A=T_A - T, K_b=K_ab)
        rab["dchiA"], rab["dchib"] = rel_width_change(A, b, A_u, b_ab)
        rab["_A"], rab["_b"] = A_u, b_ab
        R["Ab"] = rab
        results[s] = R

        # ---------------- графики ----------------
        make_plots(s, A, b, R)

    make_original_plot(results)
    make_compare_plot(results)
    dump(results)
    report(results)


def make_original_plot(results):
    fig = plt.figure(figsize=(15, 4.6))
    for k, (s, (A, b)) in enumerate(SYSTEMS.items()):
        tau, T = np.array(results[s]["orig"]["argmax"]), results[s]["orig"]["Tmax"]
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        x1r, x2r = (tau[0] - 4, tau[0] + 4), (tau[1] - 4, tau[1] + 4)
        plot_surface(ax, A, b, x1r, x2r, cmap="viridis")
        ax.scatter(*tau, T, color="red", s=40, depthshade=False, zorder=10)
        ax.set_title(f"$A_{s}x=b_{s}$: max Tol = {T:.4f}\nв точке ({tau[0]:.3f}, {tau[1]:.3f})",
                     fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("Tol")
        ax.view_init(25, -60)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/tol_original.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_plots(s, A, b, R):
    """Шаги A.2 и A.3 — только для коррекций, после которых Xi_tol непусто."""
    tau, T = np.array(R["orig"]["argmax"]), R["orig"]["Tmax"]
    methods = [m for m in ("b", "A", "Ab") if R[m]["Tmax"] >= 0]
    nm = len(methods)
    width = 16 if nm == 3 else 11.5

    # --- шаг A.2: поверхности Tol до и после коррекции ---
    fig = plt.figure(figsize=(width, 5))
    for k, m in enumerate(methods):
        r = R[m]
        xm = np.array(r["argmax"])
        pts = [tau, xm]
        x1r, x2r = window(pts, r["_polys"], pad=0.8, min_half=2.5)
        ax = fig.add_subplot(1, nm, k + 1, projection="3d")
        plot_surface(ax, A, b, x1r, x2r, color="0.6", alpha=0.35)
        plot_surface(ax, r["_A"], r["_b"], x1r, x2r, cmap="viridis", alpha=0.85)
        ax.scatter(*tau, T, color="black", s=35, depthshade=False)
        ax.scatter(*xm, r["Tmax"], color="red", s=45, depthshade=False)
        ttl = f"{METHOD_NAMES[m]}\nmax Tol = {r['Tmax']:.4f} в ({xm[0]:.4f}, {xm[1]:.4f})"
        if m == "A":
            ttl = "$A$-коррекция (предел: $A \\ominus E$, $e_{ij}=\\mathrm{rad}\\,a_{ij}$)\n" \
                  f"max Tol = {r['Tmax']:.4f} в ({xm[0]:.4f}, {xm[1]:.4f})"
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("Tol")
        ax.view_init(22, -60)
    fig.suptitle(f"Система $A_{s}x=b_{s}$: серая поверхность — до коррекции, "
                 f"цветная — после; чёрная точка — исходный максимум, красная — новый",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/tol_surf_{s}.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # --- шаг A.3: допусковые множества на плоскости ---
    fig, axes = plt.subplots(1, nm, figsize=(width, 5))
    for ax, m in zip(np.atleast_1d(axes), methods):
        r = R[m]
        xm = np.array(r["argmax"])
        x1r, x2r = window([xm], r["_polys"], pad=0.5, min_half=1e-3)
        X1, X2 = np.meshgrid(np.linspace(*x1r, 400), np.linspace(*x2r, 400))
        Z = tol(np.stack([X1, X2], -1), r["_A"], r["_b"])
        cs = ax.contour(X1, X2, Z, levels=12, cmap="viridis", linewidths=0.7)
        ax.clabel(cs, fontsize=7, fmt="%.3g")
        for P in r["_polys"]:
            ax.add_patch(MplPolygon(P, closed=True, facecolor=METHOD_COLORS[m],
                                    alpha=0.35, edgecolor=METHOD_COLORS[m], lw=1.5))
        ax.plot(*xm, "rx", ms=10, mew=2, label=f"arg max Tol = ({xm[0]:.4f}, {xm[1]:.4f})")
        title = METHOD_NAMES[m]
        if m == "A":
            title += " (предельный случай)"
        ax.set_title(title + f"\nmax Tol = {r['Tmax']:.5f}", fontsize=10)
        ax.set_xlim(x1r); ax.set_ylim(x2r)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Допусковое множество $\\Xi_{{tol}}$ системы $A_{s}x=b_{s}$ после коррекций",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/xi_tol_{s}.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_compare_plot(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (s, R) in zip(axes, results.items()):
        tau = np.array(R["orig"]["argmax"])
        polys_all = []
        for m in ("b", "Ab", "A"):
            r = R[m]
            for k, P in enumerate(r["_polys"]):
                ax.add_patch(MplPolygon(P, closed=True, facecolor=METHOD_COLORS[m],
                                        alpha=0.3, edgecolor=METHOD_COLORS[m], lw=1.5,
                                        label=METHOD_NAMES[m] if k == 0 else None))
                polys_all.append(P)
            if r["Tmax"] >= 0:
                xm = r["argmax"]
                ax.plot(*xm, "x", color=METHOD_COLORS[m], ms=9, mew=2)
        ax.plot(*tau, "ko", ms=5, label="arg max Tol исходной системы")
        x1r, x2r = window([tau], polys_all, pad=0.25, min_half=0.5)
        ax.set_xlim(x1r); ax.set_ylim(x2r)
        ax.set_title(f"$A_{s}x=b_{s}$", fontsize=11)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Сравнение допусковых множеств после коррекций (запас max Tol = "
                 f"{DELTA} для $b$- и $Ab$-коррекций)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/compare.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def dump(results):
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(clean(results), f, ensure_ascii=False, indent=2)


def report(results):
    for s, R in results.items():
        o = R["orig"]
        print(f"===== Система A{s} x = b{s} =====")
        print(f"  arg max Tol = {np.round(o['argmax'], 6)}, max Tol = {o['Tmax']:.6f}, "
              f"образующие: {o['gens']}, T(A,b) = {o['calT']:.6f}")
        print(f"  Теорема 2 выполнена: {o['thm147']['holds']}, Delta (9) = {o['Delta']:.6f}")
        for m in ("b", "A", "Ab"):
            r = R[m]
            extra = ""
            if m == "A":
                extra = (f" e_need={r['e_need']:.6f}, e_max={r['e_max']:.2f}, "
                         f"Tol(e_max)={r['uniform']['Tmax']:.6f}")
                if "rad_left" in r:
                    extra += f", остаточный rad a_ij = {r['rad_left']:.2e}"
            print(f"  {m:>2}: argmax={np.round(r['argmax'], 6)}, max Tol={r['Tmax']:.6f}, "
                  f"образующие={r['gens']}, area={r['area']:.6g}, T={r['calT']:.6f}, "
                  f"K_A={r['K_A']:.6f}, K_b={r['K_b']:.6f}{extra}")


if __name__ == "__main__":
    main()