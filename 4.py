"""
Лабораторная работа №4 по интервальному анализу.
Интервальные мода и медианы, коэффициент Жаккара,
оценивание параметров моделей a + X = Y и t * X = Y.

Запуск:  python 4.py  (или Run в PyCharm)

Структура папки Lab4:
    Lab4/
        4.py
        -0.205_lvl_side_a_fast_data.bin
        0.225_lvl_side_a_fast_data.bin
        figures/        <- сюда сохраняются графики
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # папка Lab4
DATA_DIR = BASE_DIR                                     # .bin лежат рядом с 4.py
FILE_X = "-0.205_lvl_side_a_fast_data.bin"
FILE_Y = "0.225_lvl_side_a_fast_data.bin"
OUT_DIR = os.path.join(BASE_DIR, "figures")

CHANNEL = 0              # канал АЦП (0..7)
N = 14
RAD = 1.0 / 2 ** N       # rad x = rad y = 1/2^N
EPS = 2.0 ** -20         # точность (шаг сетки) по параметру, ≈ 9.5e-7
A_RANGE = (-1.0, 1.0)    # диапазон перебора a
T_RANGE = (-2.0, 2.0)    # диапазон перебора t

HEADER_SIZE = 256        # заголовок файла, байт
FRAME_SIZE = 16400       # кадр, байт
FRAME_HEADER = 16        # заголовок кадра, байт
N_POINTS = 1024          # точек в кадре
N_CHANNELS = 8           # значений в точке


# ------------------------------------------------------------------
# A. Чтение данных и формирование интервальных выборок
# ------------------------------------------------------------------
def read_bin(path):
    """Чтение файла формата Save to BIN: возвращает (side, mode, frames, codes),
    codes имеет форму (frames, 1024, 8)."""
    raw = np.fromfile(path, dtype=np.uint8)
    side, mode = int(raw[0]), int(raw[1])
    frame_count = int(raw[2:4].view("<u2")[0])
    body = raw[HEADER_SIZE:HEADER_SIZE + frame_count * FRAME_SIZE]
    frames = body.reshape(frame_count, FRAME_SIZE)
    data = frames[:, FRAME_HEADER:].copy().view("<u2")
    codes = data.reshape(frame_count, N_POINTS, N_CHANNELS).astype(np.int64)
    return side, mode, frame_count, codes


def to_volts(code):
    return code / 16384.0 - 0.5


def make_sample(path, channel=CHANNEL):
    """Интервальная выборка x_i = [v_i - rad, v_i + rad]."""
    side, mode, fc, codes = read_bin(path)
    v = to_volts(codes[:, :, channel].ravel())
    return v - RAD, v + RAD, v, (side, mode, fc)


# ------------------------------------------------------------------
# Интервальные статистики
# ------------------------------------------------------------------
def elementary_subintervals(lo, hi):
    """Разбиение оси концами интервалов на элементарные подынтервалы
    [y_j, y_{j+1}] ненулевой длины и частоты их покрытия mu_j."""
    y = np.unique(np.concatenate([lo, hi]))
    left, right = y[:-1], y[1:]
    slo, shi = np.sort(lo), np.sort(hi)
    # x_i содержит [y_j, y_{j+1}]  <=>  lo_i <= y_j  и  hi_i >= y_{j+1} (hi_i > y_j)
    mu = (np.searchsorted(slo, left, side="right")
          - np.searchsorted(shi, left, side="right"))
    return left, right, mu


def merge(left, right):
    """Объединение подряд идущих отрезков в компоненты мультиинтервала."""
    comps = [[left[0], right[0]]]
    for l, r in zip(left[1:], right[1:]):
        if l <= comps[-1][1]:
            comps[-1][1] = max(comps[-1][1], r)
        else:
            comps.append([l, r])
    return comps


def interval_mode(lo, hi):
    """Мода интервальной выборки (Баженов, табл. 3.1):
    объединение элементарных подынтервалов с максимальной частотой."""
    if lo.max() <= hi.min():                       # выборка совместна
        return [[lo.max(), hi.min()]], len(lo)
    left, right, mu = elementary_subintervals(lo, hi)
    m = mu.max()
    k = mu == m
    return merge(left[k], right[k]), int(m)


def median_kreinovich(lo, hi):
    """medK = [med(lo_i), med(hi_i)]."""
    return np.median(lo), np.median(hi)


def median_prolubnikov(lo, hi):
    """Me_f (Пролубников): центральный элемент вариационного ряда
    элементарных подынтервалов r_j, взятых с повторениями mu_j."""
    left, right, mu = elementary_subintervals(lo, hi)
    keep = mu > 0
    left, right, mu = left[keep], right[keep], mu[keep]
    S = int(mu.sum())
    cum = np.cumsum(mu)
    if S % 2 == 1:
        pos = [(S + 1) // 2]
    else:
        pos = [S // 2, S // 2 + 1]
    idx = [int(np.searchsorted(cum, p, side="left")) for p in pos]
    i, j = min(idx), max(idx)
    info = dict(S=S, idx=(i, j), mu=(int(mu[i]), int(mu[j])),
                left_sum=int(cum[i - 1]) if i > 0 else 0,
                right_sum=int(S - cum[j]))
    return (left[i], right[j]), info


# ------------------------------------------------------------------
# Коэффициент Жаккара и функционал F(s)
# ------------------------------------------------------------------
def jaccard(lo, hi):
    """Ji(X) = wid(∧x_i) / wid(∨x_i) (∧, ∨ — в арифметике Каухера)."""
    return (hi.min() - lo.max()) / (hi.max() - lo.min())


def extremes(lo, hi):
    """(min lo, max lo, min hi, max hi) — всё, от чего зависит Ji."""
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    return lo.min(), lo.max(), hi.min(), hi.max()


def transform_extremes(ext, s, model):
    """Крайние значения концов выборки T_s(X) на сетке значений s."""
    lmin, lmax, hmin, hmax = ext
    if model == "a":                               # a + X
        return lmin + s, lmax + s, hmin + s, hmax + s
    # t * X: при t < 0 концы меняются местами
    pos = s >= 0
    new_lmin = np.where(pos, s * lmin, s * hmax)
    new_lmax = np.where(pos, s * lmax, s * hmin)
    new_hmin = np.where(pos, s * hmin, s * lmax)
    new_hmax = np.where(pos, s * hmax, s * lmin)
    return new_lmin, new_lmax, new_hmin, new_hmax


def functional(s, ext_x, ext_y, model):
    """F(s) = Ji(T_s(X) ∪ Y) для всего массива значений s."""
    lmin, lmax, hmin, hmax = transform_extremes(ext_x, s, model)
    ylmin, ylmax, yhmin, yhmax = ext_y
    num = np.minimum(hmin, yhmin) - np.maximum(lmax, ylmax)
    den = np.maximum(hmax, yhmax) - np.minimum(lmin, ylmin)
    return num / den


def apply_model(lo, hi, s, model):
    if model == "a":
        return lo + s, hi + s
    a, b = s * lo, s * hi
    return np.minimum(a, b), np.maximum(a, b)


def estimate(ext_x, ext_y, model, grid):
    F = functional(grid, ext_x, ext_y, model)
    Fmax = F.max()
    arg = np.flatnonzero(F >= Fmax - 1e-12)
    # множество argmax может быть отрезком или несвязным: разбиваем на компоненты
    breaks = np.flatnonzero(np.diff(arg) > 1)
    starts = np.concatenate([[arg[0]], arg[breaks + 1]])
    ends = np.concatenate([arg[breaks], [arg[-1]]])
    plateau = [(grid[i], grid[j]) for i, j in zip(starts, ends)]
    if len(plateau) == 1:
        s_max = 0.5 * (plateau[0][0] + plateau[0][1])  # середина отрезка argmax
    else:
        s_max = np.nan                                 # argmax несвязно
    ok = np.flatnonzero(F >= 0)
    if ok.size:
        connected = np.all(np.diff(ok) == 1)
        s_int = (grid[ok[0]], grid[ok[-1]])
    else:
        connected, s_int = True, None
    return dict(F=F, Fmax=Fmax, s_max=s_max, plateau=plateau,
                s_int=s_int, connected=connected)


# ------------------------------------------------------------------
# Вывод
# ------------------------------------------------------------------
def fmt_int(iv, k=6):
    return f"[{iv[0]:.{k}f}, {iv[1]:.{k}f}]"


def plot_data(vx, vy, stats_x, stats_y, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for row, (v, st, name, color) in enumerate(
            [(vx, stats_x, "X", "tab:blue"), (vy, stats_y, "Y", "tab:orange")]):
        ax = axes[row, 0]
        ax.plot(v, ".", ms=0.6, color=color)
        ax.set_title(f"Выборка {name}: середины интервалов (канал {CHANNEL})")
        ax.set_xlabel("номер отсчёта")
        ax.set_ylabel("V, В")
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        med = np.median(v)
        lo_w, hi_w = med - 0.03, med + 0.03
        edges = (np.round(np.arange(lo_w, hi_w, RAD) / RAD) - 0.5) * RAD
        ax.hist(v, bins=edges, color=color, alpha=0.6)
        m_lo, m_hi = st["mode"][0][0], st["mode"][-1][1]
        ax.axvspan(m_lo, m_hi, color="red", alpha=0.9, label="mode")
        ax.axvline(st["medK"][0], color="green", ls="--", lw=1, label="medK")
        ax.axvline(st["medK"][1], color="green", ls="--", lw=1)
        ax.axvline(0.5 * sum(st["medP"]), color="purple", ls=":", lw=1.5,
                   label="medP")
        ax.set_xlim(lo_w, hi_w)
        ax.set_title(f"Гистограмма середин {name} (шаг — один код АЦП)")
        ax.set_xlabel("V, В")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model(grid, results, model, path, windows):
    names = {"B.1": "B.1: исходные выборки", "B.2": "B.2: интервальные моды",
             "B.3": "B.3: медианы Крейновича", "B.4": "B.4: медианы Пролубникова"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key in zip(axes.ravel(), ["B.1", "B.2", "B.3", "B.4"]):
        r = results[key]
        w0, w1 = windows[key]
        m = (grid >= w0) & (grid <= w1)
        step = max(1, m.sum() // 20000)
        ax.plot(grid[m][::step], r["F"][m][::step], lw=1.2)
        if np.isfinite(r["s_max"]) and r["plateau"][0][0] == r["plateau"][0][1]:
            ax.plot(r["s_max"], r["Fmax"], "ro", ms=6,
                    label=f"${model}_{{max}}$ = {r['s_max']:.6f}\n"
                          f"$F_{{max}}$ = {r['Fmax']:.6f}")
            ax.axvline(r["s_max"], color="r", ls=":", lw=1)
        else:                                       # максимум на множестве
            for i, (p0, p1) in enumerate(r["plateau"]):
                ax.plot([p0, p1], [r["Fmax"], r["Fmax"]], "r-", lw=4,
                        label=(f"argmax F, $F_{{max}}$ = {r['Fmax']:.6f}"
                               if i == 0 else None))
        if key != "B.1":
            ax.axhline(0, color="k", lw=0.8)
        else:
            ax.text(0.30, 0.55, "F(s) < 0 на всём диапазоне", transform=ax.transAxes,
                    va="top")
        if r["s_int"] is not None:
            ax.axvspan(*r["s_int"], color="green", alpha=0.15, label="F ≥ 0")
        ax.set_title(names[key])
        ax.set_xlabel(model)
        ax.set_ylabel(f"F({model})")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(grid, results, model, path, window):
    fig, ax = plt.subplots(figsize=(10, 5))
    m = (grid >= window[0]) & (grid <= window[1])
    labels = {"B.2": "mode", "B.3": "medK", "B.4": "medP"}
    for key, ls in zip(["B.2", "B.3", "B.4"], ["-", "--", "-."]):
        r = results[key]
        line, = ax.plot(grid[m], r["F"][m], ls, lw=1.5, label=f"{key} ({labels[key]})")
        ax.plot(r["s_max"], r["Fmax"], "o", color=line.get_color())
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(model)
    ax.set_ylabel(f"F({model})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xlo, xhi, vx, hx = make_sample(os.path.join(DATA_DIR, FILE_X))
    ylo, yhi, vy, hy = make_sample(os.path.join(DATA_DIR, FILE_Y))
    print(f"rad = 1/2^{N} = {RAD:.6e} В")
    print(f"X: side={hx[0]}, mode={hx[1]}, кадров={hx[2]}, n={len(vx)}")
    print(f"Y: side={hy[0]}, mode={hy[1]}, кадров={hy[2]}, n={len(vy)}")

    stats = {}
    for name, lo, hi, v in [("X", xlo, xhi, vx), ("Y", ylo, yhi, vy)]:
        mode, mu = interval_mode(lo, hi)
        mk = median_kreinovich(lo, hi)
        mp, info = median_prolubnikov(lo, hi)
        stats[name] = dict(mode=mode, mu=mu, medK=mk, medP=mp, infoP=info)
        print(f"\n{name}: mean = {v.mean():.6f}, med = {np.median(v):.6f}, "
              f"min = {v.min():.6f}, max = {v.max():.6f}")
        print(f"  Ji({name}) = {jaccard(lo, hi):.6f}")
        print(f"  mode = {' ∪ '.join(fmt_int(c) for c in mode)},  частота μ = {mu}")
        print(f"  medK = {fmt_int(mk)}")
        print(f"  medP = {fmt_int(mp)},  {info}")

    plot_data(vx, vy, stats["X"], stats["Y"], os.path.join(OUT_DIR, "data.png"))

    def ext_of(iv):
        if isinstance(iv[0], (list, tuple)):      # мультиинтервал -> оболочка
            iv = (iv[0][0], iv[-1][1])
        return extremes(iv[0], iv[1])

    variants = {
        "B.1": (extremes(xlo, xhi), extremes(ylo, yhi)),
        "B.2": (ext_of(stats["X"]["mode"]), ext_of(stats["Y"]["mode"])),
        "B.3": (ext_of(stats["X"]["medK"]), ext_of(stats["Y"]["medK"])),
        "B.4": (ext_of(stats["X"]["medP"]), ext_of(stats["Y"]["medP"])),
    }

    all_results = {}
    for model, rng in [("a", A_RANGE), ("t", T_RANGE)]:
        n_pts = int(round((rng[1] - rng[0]) / EPS)) + 1
        grid = rng[0] + EPS * np.arange(n_pts)   # точные узлы, кратные 2^-20
        res = {k: estimate(ex, ey, model, grid) for k, (ex, ey) in variants.items()}
        all_results[model] = (grid, res)

        print(f"\nМодель {'a + X = Y' if model == 'a' else 't * X = Y'}, ε = {EPS}")
        print(f"{'':5}{model + '_max':>12}{'F_max':>12}   argmax           F >= 0")
        for k, r in res.items():
            si = fmt_int(r["s_int"]) if r["s_int"] is not None else "пусто"
            pl = " ∪ ".join(fmt_int(p) for p in r["plateau"])
            print(f"{k:5}{r['s_max']:12.6f}{r['Fmax']:12.6f}   {pl}  {si}"
                  f"{'' if r['connected'] else ' (несвязно)'}")

        # проверка: прямой расчёт Ji по всем интервалам в точке максимума
        for k, (lo_, hi_, ylo_, yhi_) in {
                "B.1": (xlo, xhi, ylo, yhi)}.items():
            s0 = res[k]["plateau"][0][0]
            tlo, thi = apply_model(lo_, hi_, s0, model)
            direct = jaccard(np.concatenate([tlo, ylo_]), np.concatenate([thi, yhi_]))
            print(f"  проверка {k}: Ji(T(X) ∪ Y) при s = {s0:.6f} равно {direct:.6f}")

        # окна для графиков
        windows = {"B.1": rng}
        for k in ["B.2", "B.3", "B.4"]:
            c = res[k]["s_max"]
            half = 5e-4 if model == "a" else 4e-3
            windows[k] = (c - half, c + half)
        plot_model(grid, res, model, os.path.join(OUT_DIR, f"F_{model}.png"), windows)
        centers = [res[k]["s_max"] for k in ["B.2", "B.3", "B.4"]]
        half = 3e-4 if model == "a" else 2.5e-3
        plot_overlay(grid, res, model, os.path.join(OUT_DIR, f"F_{model}_overlay.png"),
                     (min(centers) - half, max(centers) + half))

    # влияние выбросов на B.1: удаление точек дальше 0.05 В от медианы
    print("\nB.1 без грубых выбросов (|v - med| > 0.05 В):")
    kx = np.abs(vx - np.median(vx)) <= 0.05
    ky = np.abs(vy - np.median(vy)) <= 0.05
    print(f"  удалено: X — {np.sum(~kx)}, Y — {np.sum(~ky)}")
    ex, ey = extremes(xlo[kx], xhi[kx]), extremes(ylo[ky], yhi[ky])
    for model in ["a", "t"]:
        grid = all_results[model][0]
        r = estimate(ex, ey, model, grid)
        pl = " ∪ ".join(fmt_int(p) for p in r["plateau"])
        print(f"  {model}: F_max = {r['Fmax']:.6f}, argmax = {pl}")
    print(f"  Ji(X без выбросов) = {jaccard(xlo[kx], xhi[kx]):.6f}, "
          f"Ji(Y без выбросов) = {jaccard(ylo[ky], yhi[ky]):.6f}")


if __name__ == "__main__":
    main()