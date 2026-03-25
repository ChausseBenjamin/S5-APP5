import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="full",
    app_title="Mandat 2: Statistiques descriptives et inférence statistique",
    auto_download=["html"],
)

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
    # III) Les données suivent-elles une distribution normale?

    - TODO: observation de l'histogramme
    """)
    return


@app.cell
def _(dataset):
    _mean = dataset.mean()
    _std = dataset.std(ddof=1)
    _n = len(dataset)

    _k_chi2 = int(np.ceil(1 + np.log2(_n)))
    _min_val, _max_val = dataset.min(), dataset.max()
    _amplitude = (_max_val - _min_val) / _k_chi2

    _observed_freq = []
    _expected_freq = []
    _bins_edges = []

    for _i in range(_k_chi2):
        _lower = _min_val + _i * _amplitude
        _upper = _min_val + (_i + 1) * _amplitude
        _bins_edges.append((_lower, _upper))

        _count_obs = np.sum(
            (dataset >= _lower) & (dataset < _upper)
            if _i < _k_chi2 - 1
            else (dataset >= _lower) & (dataset <= _upper)
        )
        _observed_freq.append(_count_obs)

        _prob_lower = stats.norm.cdf(_lower, loc=_mean, scale=_std)
        _prob_upper = stats.norm.cdf(_upper, loc=_mean, scale=_std)
        _prob_class = _prob_upper - _prob_lower
        _expected_freq.append(_n * _prob_class)

    _observed_freq = np.array(_observed_freq)
    _expected_freq = np.array(_expected_freq)

    chi2_stat = np.sum((_observed_freq - _expected_freq) ** 2 / _expected_freq)

    ddof_chi2 = _k_chi2 - 1 - 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=ddof_chi2)

    alpha = 0.05
    chi2_critical = stats.chi2.ppf(1 - alpha, df=ddof_chi2)

    chi2_result_df = pl.DataFrame(
        {
            "Classe": [
                f"[{_lower:.0f}, {_upper:.0f}]" for _lower, _upper in _bins_edges
            ],
            "Freq. observee": _observed_freq.tolist(),
            "Freq. attendue": [round(_f, 2) for _f in _expected_freq],
        }
    )
    return alpha, chi2_critical, chi2_result_df, chi2_stat, ddof_chi2, p_value


@app.cell(hide_code=True)
def _(alpha, chi2_critical, chi2_result_df, chi2_stat, ddof_chi2, p_value):
    reject_h0 = p_value < alpha
    conclusion = "rejeter" if reject_h0 else "ne pas rejeter"

    mo.md(rf"""
    ## Test de Khi-deux (test de Pearson)

    **Hypothèses:**
    - H₀: Les données suivent une distribution normale
    - H₁: Les données ne suivent pas une distribution normale

    **Résultats:**

    {mo.ui.table(chi2_result_df, selection=None)}

    **Statistiques du test:**

    | Métrique | Valeur |
    |----------|--------|
    | Statistique χ² | {chi2_stat:.4f} |
    | Degrés de liberté | {ddof_chi2} |
    | Valeur p | {p_value:.4f} |
    | Valeur critique (α={alpha}) | {chi2_critical:.4f} |

    **Conclusion:**

    Au seuil de signification α = {alpha}, on {"**rejette**" if reject_h0 else "**ne rejette pas**"} l'hypothèse nulle.

    {"Les données **ne suivent pas** une distribution normale." if reject_h0 else "Les données **suivent** une distribution normale."}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # IV) Intervalle de confiance pour la moyenne

    TODO: Calculez l'intervalle de confiance pour la moyenne des temps de jeu avec un niveau de
    confiance de 95%. Vous ferez alors usage des tables de la distribution normale centrée réduite.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # V) Test d'hypothèse sur la moyenne

    TODO: Effectuez un test d'hypothèse approprié sur la moyenne afin d'évaluer si les données
    fournissent suffisamment de preuves pour rejeter ou ne pas rejeter l'hypothèse de votre
    patron énoncée précédemment. Considérez un niveau de confiance de 95%. Quelle est ici
    l'erreur de première espèce?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VI) Erreur de deuxième espèce

    TODO: Supposons que la moyenne échantillonnale est un très bon estimé de la moyenne de la
    population et qu'elle peut être considérée comme étant cette dernière, quelle est l'erreur de
    deuxième espèce commise au point précédent si on ne rejette pas l'hypothèse nulle?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VII) Test d'hypothèse bilatéral sur la variance

    TODO: En supposant que l'écart-type des temps de jeu est de 50, effectuez un test d'hypothèse
    bilatéral sur la variance avec un seuil de signification de 5% afin d'évaluer si les données
    fournissent suffisamment de preuves pour rejeter ou ne pas rejeter l'hypothèse nulle.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VI) Trouver l'erreur de deuxieme espece

    TODO: Supposons que la moyenne échantillonnale est un très bon estimé de la moyenne de la
    population et qu’elle peut être considérée comme étant cette dernière, quelle est l’erreur de
    deuxième espèce commise au point précédent si on ne rejette pas l’hypothèse nulle?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VII) Test d'hypothese bilateral sur la variance

    TODO: En supposant que l’écart-type des temps de jeu est de 50, effectuez un test d'hypothèse
    bilatéral sur la variance avec un seuil de signification de 5% afin d’évaluer si les données
    fournissent suffisamment de preuves pour rejeter ou ne pas rejeter l'hypothèse nulle.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
