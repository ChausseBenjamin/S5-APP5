# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy>=2.4.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mandat - 01.
    ## Probabilitées de la cible
    """)
    return


@app.cell(hide_code=True)
def fonctions(np):
    def random_value(variance, N):
        return np.random.normal(0, variance, N)


    def is_in_circle(x, y, radius):
        return (x**2 + y**2) <= radius**2

    return is_in_circle, random_value


@app.cell
def program(is_in_circle, np, random_value):

    N = 20000000

    X = random_value(0.1, N)
    Y = random_value(0.05, N)
    results = is_in_circle(X, Y, 0.1)
    success = np.sum(results)
    probability = success / N

    print("z<1: ", probability)

    X = random_value(0.1, N)
    Y = random_value(0.4, N)
    results = is_in_circle(X, Y, 0.1)
    success = np.sum(results)
    probability = success / N

    print("z>10: ", probability)
    return


if __name__ == "__main__":
    app.run()
