import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", app_title="S5-APP5")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # This is a markdown file

    And since marimo is based, you can write $\LaTeX$ just like than and even emojis (😜) without any issues. Longer equations look quite good:

    $$
    \vec{F} = \frac{Gm_1m_2}{\vec{R}^2}
    $$
    """)
    return


@app.cell
def _():
    x = np.arange(0, 4 * np.pi, 0.01)
    s_1 = mo.ui.slider(1, 100)
    s_2 = mo.ui.slider(1, 100)
    mo.vstack([s_1, s_2])
    return s_1, s_2, x


@app.cell
def _(s_1, s_2, x):
    y_1 = np.sin(x * (s_2.value / 100)) * s_1.value
    y_2 = np.cos(x * (s_1.value / 100)) * s_2.value
    return y_1, y_2


@app.cell
def _(x, y_1, y_2):
    plt.plot(x, y_1, color="red")
    plt.plot(x, y_2, color="blue")
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
