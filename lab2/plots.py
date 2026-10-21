"""Графики: f_1 и f_2 (пункт A), поверхность и линии уровня f_3 (пункт E.1)."""
import numpy as np
import matplotlib.pyplot as plt

from calculations import f_np, f_2_np, f_3, X, X_2, X_3, ran_f3


def plot(func_np, X, title, save=None):
    """График функции одной переменной на интервале X."""
    x = np.linspace(X.a, X.b, 1000)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, func_np(x))
    plt.title(title); plt.xlabel('x'); plt.ylabel(title); plt.grid(True)
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    else:
        plt.show()


def plot_surface_and_levels(func, box, n=200, levels=15, save=None):
    """Поверхность z = func(x, y) и линии уровня на брусе box."""
    x = np.linspace(box[0].a, box[0].b, n)
    y = np.linspace(box[1].a, box[1].b, n)
    Xg, Yg = np.meshgrid(x, y)
    Z = func(Xg, Yg)

    fig = plt.figure(figsize=(13, 5.5))

    # поверхность
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Xg, Yg, Z, cmap='Blues', edgecolor='none')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
    ax1.set_title('Поверхность z = f(x, y)')

    # линии уровня
    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(Xg, Yg, Z, levels=levels, cmap='Blues')
    cs = ax2.contour(Xg, Yg, Z, levels=levels, colors='k', linewidths=0.6)
    ax2.clabel(cs, fontsize=8, fmt='%.1f')
    ax2.plot(2, -1.25, 'ro', ms=6, clip_on=False, zorder=5, label='min (2, -1.25)')   # точка минимума на границе
    ax2.plot(-1, -2, 'r^', ms=7, clip_on=False, zorder=5, label='max (-1, -2)')       # точка максимума (вершина)
    ax2.legend(loc='upper right', fontsize=8)
    fig.colorbar(cf, ax=ax2, label='z')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_title('Линии уровня f(x, y)')
    ax2.set_aspect('equal')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    import sys
    save = '--save' in sys.argv          # python plots.py --save — сохранить PNG для отчёта
    plot(f_np, X, 'f_1(x)', 'f1_plot.png' if save else None)                    # A
    plot(f_2_np, X_2, 'f_2(x)', 'f2_plot.png' if save else None)                # A
    plot_surface_and_levels(f_3, X_3, save='f3_plot.png' if save else None)     # E.1
