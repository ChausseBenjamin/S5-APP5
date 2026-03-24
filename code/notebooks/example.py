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
    y = np.sin(x)
    return x, y


@app.cell
def _(x, y):
    plt.plot(x, y, color="red")
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
