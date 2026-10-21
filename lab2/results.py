"""Вывод результатов: таблицы оценок и расстояний, проверки липшицевских оценок."""
import calculations as c


def report_1d(name, ran, forms, dist, f_mv, dist_mv, f_sl, dist_sl):
    """Таблица для функции одной переменной: B.1, B.2, B.5 и B.3, B.4 по центрам."""
    print(f'\n===== {name}: ran = {ran} =====')
    for k, F in forms.items():
        print(f'{k:20s} F = {str(F):28s} dist = {dist[k]:.4f}')
    print(f'{"c":>6s}   {"F_mv":28s}  {"dist":>7s}  {"F_sl":28s}  {"dist":>7s}')
    for cc in f_mv:
        print(f'{cc:6.2f}   {str(f_mv[cc]):28s}  {dist_mv[cc]:7.3f}  '
              f'{str(f_sl[cc]):28s}  {dist_sl[cc]:7.3f}')


def report_lipschitz(name, F, ran, L, th, LF, thF):
    """C.3: сравнение rad F(X) с L·rad X и L_F·rad X."""
    print(f'\n{name}: rad F(X) = {F.rad:.4f}, rad ran = {ran.rad:.4f}')
    print(f'  L   = {L:.4f},  L·rad X   = {th:.4f}  ->',
          'выполняется' if F.rad <= th + 1e-12 else f'нарушается (в {F.rad / th:.2f} раза)')
    print(f'  L_F = {LF:.4f}, L_F·rad X = {thF:.4f}  ->',
          'выполняется' if F.rad <= thF else 'нарушается')


if __name__ == '__main__':
    # ----- 1. Функции одной переменной: A, B, C -----
    report_1d('f_1', c.ran_f, c.forms_1, c.dist_1,
              c.f_c, c.dist_mv_1, c.f_c_n, c.dist_sl_1)
    report_1d('f_2', c.ran_f_2, c.forms_2, c.dist_2,
              c.f_2_c, c.dist_mv_2, c.f_2_c_n, c.dist_sl_2)
    print(f'\nf_2: ran по сетке = {c.ran_f_2_grid}, f_2\'(X) = {c.f_2_dif(c.X_2)}')
    print('f_2: Бауман p, c_*, c^* =', c.bauman_centers(c.f_2_dif, c.X_2))
    print('f_2: наклонная форма с частным по Горнеру (для сравнения):')
    for cc, F in c.f_2_c_n_horner.items():
        print(f'   c = {cc:5.2f}  F_sl = {str(F):26s} dist = {c.dist_sl_2_horner[cc]:.3f}')

    # ----- C.1 – C.3 -----
    report_lipschitz('f_1', c.int_rash, c.ran_f, c.L_1, c.theor_1, c.L_F_1, c.theor_F_1)
    report_lipschitz('f_2', c.int_rash_2, c.ran_f_2, c.L_2, c.theor_2, c.L_F_2, c.theor_F_2)

    # ----- 2. Функция двух переменных: E.2 – E.9 -----
    print(f'\n===== f_3: ran = {c.ran_f3} =====')
    print(f'E.2  значения в вершинах: {c.vertices_3}')
    print(f'E.2  минимумы на рёбрах:  {c.edges_3}')
    print(f'E.5  градиент на X: ∂f/∂x ∈ {c.D1}, ∂f/∂y ∈ {c.D2}')
    print(f'E.6  p = {c.p}, центры Баумана: c_* = {c.c_lo}, c^* = {c.c_up}')
    print(f'E.7  по x: {c.split_x}, по y: {c.split_y}')
    print(f'E.8  L1 = {c.L1}, L2 = {c.L2}, L1·rad X1 + L2·rad X2 = {c.L_bound}; '
          f'rad F_nat = {c.F_nat.rad} ->',
          'выполняется' if c.F_nat.rad <= c.L_bound else 'нарушается')
    print(f'E.9  {"оценка":36s} {"F":22s} {"rad F":>7s} {"dist":>7s}')
    for name, F in c.estimates_3.items():
        print(f'     {name:36s} {str(F):22s} {F.rad:7.3f} {c.dist_3[name]:7.3f}')
