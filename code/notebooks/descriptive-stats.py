import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib as mpl


@app.cell
def _():
    dataset = np.loadtxt("input/TempsDeJeu.txt")
    return (dataset,)


@app.cell
def _(dataset):
    def hours(m):
        return m / 60


    stats_dict = {
        "Moyenne": hours(dataset.mean()),
        "Mediane": hours(np.median(dataset)),
        "Mode": hours(stats.mode(dataset).mode.item()),
        "Ecart type": hours(np.std(dataset, ddof=1)),
        "Variance": hours(np.var(dataset, ddof=1)),
        "Min": hours(dataset.min()),
        "Max": hours(dataset.max()),
        "Etendue": hours(np.ptp(dataset)),
    }

    df = pl.DataFrame(
        {
            "Metrique": list(stats_dict.keys()),
            "Valeur": [round(v, 2) for v in stats_dict.values()],
        }
    )
    return (df,)


@app.cell(hide_code=True)
def _(df):
    mo.md(rf"""
    # I) Statistiques descriptives des temps de jeu

    {mo.ui.table(df, selection=None)}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # II) Construction de l'histogramme des données
    """)
    return


@app.cell
def _(dataset):
    k = int(np.ceil(1 + np.log2(len(dataset))))
    min_val, max_val = dataset.min(), dataset.max()
    amplitude = (max_val - min_val) / k

    classes = []
    limites = []
    centres = []
    frequences = []

    for i in range(k):
        lower = min_val + i * amplitude
        upper = min_val + (i + 1) * amplitude
        centres.append((lower + upper) / 2)
        limites.append(f"[{lower:.0f}, {upper:.0f})")

        if i == k - 1:
            limites[-1] = f"[{lower:.0f}, {upper:.0f}]"

        count = np.sum(
            (dataset >= lower) & (dataset < upper)
            if i < k - 1
            else (dataset >= lower) & (dataset <= upper)
        )
        frequences.append(count)

    freq_relatives = [f / len(dataset) for f in frequences]
    freq_cumulees = np.cumsum(frequences).tolist()

    classes_col = [f"Classe {i + 1}" for i in range(k)]

    freq_df = pl.DataFrame(
        {
            "Classes": classes_col,
            "Limites": limites,
            "Centres": [round(c, 1) for c in centres],
            "Freq. relatives": [round(f, 4) for f in freq_relatives],
            "Freq. cumulees": freq_cumulees,
        }
    )
    return freq_df, k


@app.cell(hide_code=True)
def _(freq_df, k):
    mo.md(rf"""
    ## Nombre de classes
    La loi de Sturges est utilisée pour déterminer le nombre de classes: 

    $$
    k = \lceil 1 + \log_2(n) \rceil = {k}
    $$

    Où: 

    | | |
    |-|-|
    | k | nombre de classes |
    | n | taille de l'échantillon |

    ## Population du tableau de valeurs

    {mo.ui.table(freq_df, selection=None)}
    """)
    return


@app.cell
def _(dataset, k):
    plt.figure(figsize=(10, 6))
    plt.hist(dataset, bins=k, edgecolor="black", alpha=0.7)
    plt.xlabel("Temps de jeu (minutes)")
    plt.ylabel("Frequence")
    plt.title(f"Histogramme des temps de jeu ({k} classes)")
    plt.axvline(
        dataset.mean(),
        color="red",
        linestyle="--",
        label=f"Moyenne: {dataset.mean():.1f}",
    )
    plt.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Les données suivent-elles une distribution normale?

    - TODO: observation de l'histogramme
    - TODO: Test quantitatif de Khi-deux (aussi appellé test de Pearson)
    """)
    return


if __name__ == "__main__":
    app.run()
