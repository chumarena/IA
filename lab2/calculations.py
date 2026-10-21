"""Формулы и вычисления: функции, интервальные оценки, расстояния, константы Липшица.
Вариант 8: f_2(x) = x^4 - 5x^2 + 4,  f_3(x, y) = x^2 + 2xy + 2y^2 - 3x + y."""
import numpy as np
import intvalpy as ip

# ======================================================================
# Общие вспомогательные функции
# ======================================================================
P = lambda c: ip.Interval(c, c)            # точка как вырожденный интервал


def cut(x, lo=-1, hi=1):
    """Отсечение числа к отрезку [lo, hi] (для теоремы Баумана)."""
    return min(max(x, lo), hi)


def hausdorff(F, ran):
    """Расстояние Хаусдорфа: max(|a1 - a2|, |b1 - b2|); то же, что ip.dist(F, ran)."""
    return max(abs(F.a - ran.a), abs(F.b - ran.b))


def hull(A, B):
    """Интервальная оболочка □(A ∪ B)."""
    return ip.Interval(min(A.a, B.a), max(A.b, B.b))


def horner(coeffs, x):
    """Многочлен по схеме Горнера; coeffs — от старшего к свободному члену."""
    r = coeffs[0]
    for a in coeffs[1:]:
        r = r * x + a
    return r


# ======================================================================
# 1. ФУНКЦИИ ОДНОЙ ПЕРЕМЕННОЙ
# ======================================================================
def f(x):      return x**3 - 3*x**2 + 2
def f_dif(x):  return 3*x**2 - 6*x
X = ip.Interval(0, 3)

# f_2' = 4x^3 - 10x = 2x(2x^2 - 5) = 0 при x = 0, ±sqrt(2.5);  f_2'' = 0 при x = ±sqrt(5/6).
# X_2 = [-1, 2] содержит локальный максимум x = 0, локальный минимум x = sqrt(2.5)
# и обе точки перегиба x = ±0.9129; интервал несимметричен и содержит ноль внутри.
def f_2(x):     return x**4 - 5*x**2 + 4
def f_2_dif(x): return 4*x**3 - 10*x
X_2 = ip.Interval(-1, 2)

# версии для numpy (графики, сетки)
f_np       = lambda x: x**3 - 3*x**2 + 2
f_dif_np   = lambda x: 3*x**2 - 6*x
f_2_np     = lambda x: x**4 - 5*x**2 + 4
f_2_dif_np = lambda x: 4*x**3 - 10*x

# центры для B.3 и B.4: 11 точек, включая концы и середину интервала
centers   = np.linspace(X.a, X.b, 11)
centers_2 = np.linspace(X_2.a, X_2.b, 11)

# ---------- A: точная область значений ----------
# f:   f'(x) = 3x(x - 2) = 0 при x = 0, 2  ->  ran = [f(2), f(0)] = [-2, 2]
# f_2: f_2(-1) = 0, f_2(0) = 4, f_2(sqrt 2.5) = -2.25, f_2(2) = 0  ->  ran = [-2.25, 4]
def exact_range(func_np, X, n=10**6):
    """ran(f, X) перебором по сетке из n точек."""
    y = func_np(np.linspace(X.a, X.b, n))
    return ip.Interval(y.min(), y.max())

ran_f   = exact_range(f_np, X)        # [-2, 2]
ran_f_2 = ip.Interval(-2.25, 4)       # аналитически; сетка даёт то же
ran_f_2_grid = exact_range(f_2_np, X_2)

# ---------- B.1: естественное расширение ----------
int_rash   = f(X)                     # [-25, 29]
int_rash_2 = f_2(X_2)                 # [0,16] - [0,20] + 4 = [-16, 20]

# ---------- B.2: схема Горнера / другое преобразование ----------
int_rash_h = horner([1, -3, 0, 2], X)                             # [-25, 2]
# Для f_2 на X_2 = [-1, 2] Горнер хуже исходной записи, т.к. X_2 содержит ноль:
# X·X = [-2, 4], а X^2 = [0, 4].
int_rash_2_h_horner = horner([1, 0, -5, 0, 4], X_2)               # для сравнения
# разложение на множители: (x^2 - 1)(x^2 - 4)
int_rash_2_factor   = (X_2**2 - 1) * (X_2**2 - 4)                  # для сравнения
# выделение полного квадрата: x^4 - 5x^2 + 4 = (x^2 - 5/2)^2 - 9/4
int_rash_2_h        = (X_2**2 - 2.5)**2 - 2.25                     # [-2.25, 4]

# ---------- B.3: дифференциальная центрированная форма ----------
def f_mv(f, f_dif, X, c):
    """F_mv(X, c) = f(c) + f'(X)(X - c)."""
    return f(P(c)) + f_dif(X) * (X - P(c))

f_c   = {c: f_mv(f, f_dif, X, c)       for c in centers}
f_2_c = {c: f_mv(f_2, f_2_dif, X_2, c) for c in centers_2}

# ---------- B.4: наклонная центрированная форма ----------
# Наклон нельзя получить подстановкой X в (f(x) - f(c))/(x - c):
# X - c содержит 0. Поэтому дробь сокращается аналитически.
def poly_slope_coeffs(coeffs, c):
    """Коэффициенты частного (p(x) - p(c)) / (x - c): схема Горнера в точке c."""
    b = [coeffs[0]]
    for a in coeffs[1:-1]:
        b.append(b[-1] * c + a)
    return b

def poly_slope(coeffs, X, c):
    """Интервальный наклон многочлена при x ∈ X: частное по схеме Горнера на X."""
    return horner(poly_slope_coeffs(coeffs, c), X)

def f_slope(X, c):
    return poly_slope([1, -3, 0, 2], X, c)

def f_2_slope_horner(X, c):
    """Наклон f_2 общим способом (частное по Горнеру) — для сравнения."""
    return poly_slope([1, 0, -5, 0, 4], X, c)

def f_2_slope(X, c):
    # x⁴ - c⁴ - 5(x² - c²) = (x - c)(x + c)(x² + c² - 5)  =>  f_2∠(x, c) = (x + c)(x² + c² - 5).
    # Квадрат X² вычисляется точно, поэтому оценка уже, чем частное по Горнеру.
    return (X + c) * (X**2 + c**2 - 5)

def f_sl(f, slope, X, c):
    """F_sl(X, c) = f(c) + f∠(X, c)(X - c)."""
    return f(P(c)) + slope(X, c) * (X - P(c))

f_c_n   = {c: f_sl(f, f_slope, X, c)       for c in centers}
f_2_c_n = {c: f_sl(f_2, f_2_slope, X_2, c) for c in centers_2}
f_2_c_n_horner = {c: f_sl(f_2, f_2_slope_horner, X_2, c) for c in centers_2}   # для сравнения

# ---------- B.5: бицентрированная форма (теорема Баумана) ----------
def bauman_centers(f_dif, X):
    D = f_dif(X)
    p = cut(D.mid / D.rad) if D.rad > 0 else np.sign(D.mid)
    return p, X.mid - p * X.rad, X.mid + p * X.rad

def f_bic(f, f_dif, X):
    """Нижняя граница — из F_mv с центром c_*, верхняя — с центром c^*."""
    _, c_lo, c_up = bauman_centers(f_dif, X)
    return ip.Interval(f_mv(f, f_dif, X, c_lo).a, f_mv(f, f_dif, X, c_up).b)

bic   = f_bic(f, f_dif, X)
bic_2 = f_bic(f_2, f_2_dif, X_2)

# ---------- C: расстояния Хаусдорфа до ran ----------
forms_1 = {'B.1 естественное': int_rash, 'B.2 Горнер': int_rash_h, 'B.5 бицентр.': bic}
forms_2 = {'B.1 естественное': int_rash_2, 'B.2 полный квадрат': int_rash_2_h,
           'B.2 Горнер': int_rash_2_h_horner, 'B.2 множители': int_rash_2_factor,
           'B.5 бицентр.': bic_2}

dist_1 = {k: hausdorff(F, ran_f)   for k, F in forms_1.items()}
dist_2 = {k: hausdorff(F, ran_f_2) for k, F in forms_2.items()}
dist_mv_1 = {c: hausdorff(F, ran_f)   for c, F in f_c.items()}      # B.3 для каждого c
dist_mv_2 = {c: hausdorff(F, ran_f_2) for c, F in f_2_c.items()}
dist_sl_1 = {c: hausdorff(F, ran_f)   for c, F in f_c_n.items()}    # B.4 для каждого c
dist_sl_2 = {c: hausdorff(F, ran_f_2) for c, F in f_2_c_n.items()}
dist_sl_2_horner = {c: hausdorff(F, ran_f_2) for c, F in f_2_c_n_horner.items()}

# ---------- C.1: константы Липшица ----------
# L — константа Липшица функции f: по теореме Лагранжа L = max |f'(x)| на X.
def lipschitz(f_dif_np, X, n=10**6):
    return np.abs(f_dif_np(np.linspace(X.a, X.b, n))).max()

L_1 = lipschitz(f_dif_np, X)       # 9:  |3x² - 6x| максимален при x = 3
L_2 = lipschitz(f_2_dif_np, X_2)   # 12: |4x³ - 10x| максимален при x = 2

# L_F — константа Липшица естественного расширения: wid F(x) <= L_F·wid x.
#   f_1: max(3x² + 6x) на [0, 3] = 45
#   f_2: max(4|x|³ + 10|x|) на [-1, 2] = 52 (при x = 2)
def lipschitz_extension(F, X, h=1e-4, n=3000):
    """Численная оценка L_F = sup wid F(x) / wid x по маленьким подынтервалам x ⊆ X."""
    return max(F(ip.Interval(t, t + h)).wid / h
               for t in np.linspace(X.a, X.b - h, n))

L_F_1 = lipschitz_extension(f, X)       # ≈ 45
L_F_2 = lipschitz_extension(f_2, X_2)   # ≈ 52

# ---------- C.2: теоретические оценки ----------
theor_1,   theor_2   = L_1 * X.rad,   L_2 * X_2.rad       # по L функции f
theor_F_1, theor_F_2 = L_F_1 * X.rad, L_F_2 * X_2.rad     # по L_F расширения


# ======================================================================
# 2. ФУНКЦИЯ ДВУХ ПЕРЕМЕННЫХ
# ======================================================================
def f_3(x, y):
    return x**2 + 2*x*y + 2*y**2 - 3*x + y

X_3 = ip.Interval([-1, -2], [2, 1])               # x ∈ [-1, 2], y ∈ [-2, 1]
X1, X2 = X_3[0], X_3[1]
grad_3 = lambda x, y: (2*x + 2*y - 3, 2*x + 4*y + 1)   # (∂f/∂x, ∂f/∂y)

# ---------- E.2: точная область значений ----------
# grad = 0 в (3.5, -2) ∉ X; гессиан [[2, 2], [2, 4]] > 0 -> f_3 выпукла.
# Максимум — в вершине бруса, минимум — на границе: на ребре x = 2
# f_3(2, y) = 2y² + 5y - 2, минимум при y = -5/4: f_3(2, -1.25) = -5.125.
def edge_minima():
    """Минимумы f_3 на четырёх рёбрах бруса (квадратичные функции одной переменной)."""
    res = {}
    for xv in (X1.a, X1.b):     # f_3(xv, y) = 2y² + (2xv + 1)y + xv² - 3xv
        y0 = cut(-(2*xv + 1) / 4, X2.a, X2.b)
        res[(xv, y0)] = f_3(xv, y0)
    for yv in (X2.a, X2.b):     # f_3(x, yv) = x² + (2yv - 3)x + 2yv² + yv
        x0 = cut(-(2*yv - 3) / 2, X1.a, X1.b)
        res[(x0, yv)] = f_3(x0, yv)
    return res

vertices_3 = {(x, y): f_3(x, y) for x in (X1.a, X1.b) for y in (X2.a, X2.b)}
edges_3 = edge_minima()
ran_f3 = ip.Interval(min(edges_3.values()), max(vertices_3.values()))   # [-5.125, 14]

# ---------- E.3: естественное расширение ----------
F_nat = f_3(X1, X2)                                               # [-16, 20]

# ---------- E.4: эквивалентные выражения (выделение полных квадратов) ----------
# f_3 = ½(x - 7/2)² + 2(y + x/2 + 1/4)² - 25/4      (квадрат сначала по y, затем по x)
#     = (x + y - 3/2)² + (y + 2)² - 25/4             (квадрат сначала по x)
#     = (x + y)² + y² - 3x + y
F_eq  = 0.5*(X1 - 3.5)**2 + 2*(X2 + 0.5*X1 + 0.25)**2 - 6.25
F_eq2 = (X1 + X2 - 1.5)**2 + (X2 + 2)**2 - 6.25
F_eq3 = (X1 + X2)**2 + X2**2 - 3*X1 + X2

# ---------- E.5: дифференциальная центрированная форма ----------
D1, D2 = grad_3(X1, X2)                                           # [-9, 3], [-9, 9]
F_mv = lambda c1, c2: f_3(P(c1), P(c2)) + D1*(X1 - P(c1)) + D2*(X2 - P(c2))
centers_3 = {'mid X': (X1.mid, X2.mid), '(0, 0)': (0, 0), 'вершина (-1, -2)': (X1.a, X2.a)}

# ---------- E.6: бицентрированная форма (теорема Баумана) ----------
p = [cut(D.mid / D.rad) for D in (D1, D2)]
c_lo = [Xi.mid - pi * Xi.rad for Xi, pi in zip((X1, X2), p)]      # центр для нижней границы
c_up = [Xi.mid + pi * Xi.rad for Xi, pi in zip((X1, X2), p)]      # центр для верхней границы
F_bic = ip.Interval(F_mv(*c_lo).a, F_mv(*c_up).b)

# ---------- E.7: дробление пополам по x и по y ----------
halves = lambda Xi: (ip.Interval(Xi.a, Xi.mid), ip.Interval(Xi.mid, Xi.b))
split_x = [f_3(h, X2) for h in halves(X1)]
split_y = [f_3(X1, h) for h in halves(X2)]
F_split_x = hull(*split_x)
F_split_y = hull(*split_y)

# ---------- E.8: покоординатная липшицевская оценка ----------
# Производные линейны и каждая переменная входит в них один раз,
# поэтому D1, D2 — точные области значений производных
L1, L2 = ip.mag(D1), ip.mag(D2)                                   # 9, 9
L_bound = L1 * X1.rad + L2 * X2.rad                               # 27

# ---------- E.9: все оценки для сравнения с ran(f_3, X) ----------
estimates_3 = {'E.3 естественное': F_nat,
               'E.4 ½(x-7/2)²+2(y+x/2+1/4)²-25/4': F_eq,
               'E.4 (x+y-3/2)²+(y+2)²-25/4': F_eq2,
               'E.4 (x+y)²+y²-3x+y': F_eq3,
               **{f'E.5 F_mv, c = {n}': F_mv(*c) for n, c in centers_3.items()},
               'E.6 бицентрированная': F_bic,
               'E.7 дробление по x': F_split_x,
               'E.7 дробление по y': F_split_y}
dist_3 = {name: hausdorff(F, ran_f3) for name, F in estimates_3.items()}
