import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="full",
    app_title="Mandat 2: Statistiques descriptives et inférence statistique",
)

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib as mpl


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mandat - 02: Statistiques descriptives et inférence statistique
    """)
    return


@app.cell
def _():
    dataset = np.loadtxt("input/TempsDeJeu.txt")
    return (dataset,)


@app.cell
def _(dataset):
    def hours(m):
        return m / 60

    stats_dict = {
        "Moyenne": dataset.mean(),
        "Mediane": np.median(dataset),
        "Mode": stats.mode(dataset).mode.item(),
        "Ecart type": np.std(dataset, ddof=1),
        "Variance": np.var(dataset, ddof=1),
        "Min": dataset.min(),
        "Max": dataset.max(),
        "Etendue": np.ptp(dataset),
    }

    df = pl.DataFrame(
        {
            "Metrique": list(stats_dict.keys()),
            "Valeur": [round(v, 2) for v in stats_dict.values()],
        }
    )
    return df, hours


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

    En observant l'histogramme ci-dessus, on peut voir que les données semblent suivre une distribution
    approximativement normale (forme de cloche). Pour le confirmer quantitativement, nous utilisons le test de Khi-deux.
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

    # Calculate initial observed and expected frequencies
    _observed_freq_raw = []
    _expected_freq_raw = []
    _bins_edges_raw = []

    for _i in range(_k_chi2):
        _lower = _min_val + _i * _amplitude
        _upper = _min_val + (_i + 1) * _amplitude
        _bins_edges_raw.append((_lower, _upper))

        _count_obs = np.sum(
            (dataset >= _lower) & (dataset < _upper)
            if _i < _k_chi2 - 1
            else (dataset >= _lower) & (dataset <= _upper)
        )
        _observed_freq_raw.append(_count_obs)

        _prob_lower = stats.norm.cdf(_lower, loc=_mean, scale=_std)
        _prob_upper = stats.norm.cdf(_upper, loc=_mean, scale=_std)
        _prob_class = _prob_upper - _prob_lower
        _expected_freq_raw.append(_n * _prob_class)

    # ADJUSTMENT: Merge classes with expected frequency < 5
    _observed_freq = []
    _expected_freq = []
    _bins_edges = []

    _i = 0
    while _i < len(_expected_freq_raw):
        # Start a new class
        _obs_merged = _observed_freq_raw[_i]
        _exp_merged = _expected_freq_raw[_i]
        _lower_merged = _bins_edges_raw[_i][0]
        _upper_merged = _bins_edges_raw[_i][1]

        # Merge with subsequent classes while expected frequency < 5
        while _exp_merged < 5 and _i + 1 < len(_expected_freq_raw):
            _i += 1
            _obs_merged += _observed_freq_raw[_i]
            _exp_merged += _expected_freq_raw[_i]
            _upper_merged = _bins_edges_raw[_i][1]

        _observed_freq.append(_obs_merged)
        _expected_freq.append(_exp_merged)
        _bins_edges.append((_lower_merged, _upper_merged))
        _i += 1

    # Convert to arrays
    _observed_freq = np.array(_observed_freq)
    _expected_freq = np.array(_expected_freq)

    chi2_stat = np.sum((_observed_freq - _expected_freq) ** 2 / _expected_freq)

    # Degrees of freedom: (number of classes after merging) - 1 - 2
    ddof_chi2 = len(_observed_freq) - 1 - 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=ddof_chi2)

    alpha = 0.05
    chi2_critical = stats.chi2.ppf(1 - alpha, df=ddof_chi2)

    chi2_result_df = pl.DataFrame(
        {
            "Classe": [
                f"[{_lower:.0f}, {_upper:.0f}]"
                for _lower, _upper in _bins_edges
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

    Calculez l'intervalle de confiance pour la moyenne des temps de jeu avec un niveau de
    confiance de 95%. Vous ferez alors usage des tables de la distribution normale centrée réduite.
    """)
    return


@app.cell
def _(dataset, hours):
    confidence_level = 0.95
    alpha_ci = 1 - confidence_level

    sample_mean = dataset.mean()
    sample_std = dataset.std(ddof=1)
    n_samples = len(dataset)

    z_critical = stats.norm.ppf(1 - alpha_ci / 2)

    standard_error = sample_std / np.sqrt(n_samples)
    margin_of_error = z_critical * standard_error

    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error

    ci_df = pl.DataFrame(
        {
            "Paramètre": [
                "Moyenne de l'échantillon (min)",
                "Moyenne de l'échantillon (h)",
                "Écart-type de l'échantillon (min)",
                "Taille de l'échantillon",
                "Niveau de confiance",
                "Valeur critique z",
                "Erreur standard (min)",
                "Marge d'erreur (min)",
                "Limite inférieure (min)",
                "Limite supérieure (min)",
                "Limite inférieure (h)",
                "Limite supérieure (h)",
            ],
            "Valeur": [
                f"{sample_mean:.2f}",
                f"{sample_mean:.2f}",
                f"{sample_std:.2f}",
                f"{n_samples}",
                f"{confidence_level * 100:.0f}%",
                f"{z_critical:.4f}",
                f"{standard_error:.2f}",
                f"{margin_of_error:.2f}",
                f"{ci_lower:.2f}",
                f"{ci_upper:.2f}",
                f"{ci_lower:.2f}",
                f"{ci_upper:.2f}",
            ],
        }
    )
    return alpha_ci, ci_df, ci_lower, ci_upper, confidence_level


@app.cell(hide_code=True)
def _(alpha_ci, ci_df, ci_lower, ci_upper, confidence_level, hours):
    mo.md(rf"""
    ## Calcul de l'intervalle de confiance

    Pour un niveau de confiance de {confidence_level * 100:.0f}%, l'intervalle de confiance pour la moyenne est calculé avec la formule:

    $$
    \bar{{x}} \pm z_{{\alpha/2}} \times \frac{{s}}{{\sqrt{{n}}}}
    $$

    Où:

    | Symbole | Description |
    |---------|-------------|
    | $\bar{{x}}$ | Moyenne de l'échantillon |
    | $z_{{\alpha/2}}$ | Valeur critique de la distribution normale (pour α={alpha_ci}) |
    | $s$ | Écart-type de l'échantillon |
    | $n$ | Taille de l'échantillon |

    **Résultats:**

    {mo.ui.table(ci_df, selection=None)}

    **Conclusion:**

    Avec un niveau de confiance de {confidence_level * 100:.0f}%, l'intervalle de confiance pour la moyenne des temps de jeu est:

    $$
    [{ci_lower:.2f}, {ci_upper:.2f}] \text{{ minutes}} = [{hours(ci_lower):.2f}, {hours(ci_upper):.2f}] \text{{ heures}}
    $$

    Cela signifie que nous sommes {confidence_level * 100:.0f}% confiants que la vraie moyenne de la population se situe dans cet intervalle.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # V) Test d'hypothèse sur la moyenne

    Effectuez un test d'hypothèse approprié sur la moyenne afin d'évaluer si les données
    fournissent suffisamment de preuves pour rejeter ou ne pas rejeter l'hypothèse de votre
    patron énoncée précédemment. Considérez un niveau de confiance de 95%. Quelle est ici
    l'erreur de première espèce?
    """)
    return


@app.cell
def _(dataset, hours):
    # Hypothesis test on the mean
    # H0: μ ≥ 300 (boss claims average playtime is at least 300 minutes)
    # H1: μ < 300 (one-tailed test, left-tailed)

    mu_0 = 300  # hypothesized population mean (5 hours = 300 minutes)
    alpha_v = 0.05  # significance level (95% confidence)

    # Sample statistics
    x_bar = dataset.mean()
    s = dataset.std(ddof=1)
    n = len(dataset)

    # Test statistic (z-test since n is large)
    z_stat = (x_bar - mu_0) / (s / np.sqrt(n))

    # Critical value for one-tailed test (left-tailed)
    z_critical_v = stats.norm.ppf(alpha_v)

    # P-value for one-tailed test
    p_value_v = stats.norm.cdf(z_stat)

    # Decision
    reject_h0_v = z_stat < z_critical_v

    # Type I error (alpha) - probability of rejecting H0 when H0 is true
    type_i_error = alpha_v

    hyp_test_df = pl.DataFrame(
        {
            "Paramètre": [
                "Hypothèse nulle H₀",
                "Hypothèse alternative H₁",
                "Moyenne hypothétisée μ₀ (min)",
                "Moyenne hypothétisée μ₀ (h)",
                "Moyenne échantillonnale x̄ (min)",
                "Moyenne échantillonnale x̄ (h)",
                "Écart-type s (min)",
                "Taille de l'échantillon n",
                "Statistique de test z",
                "Valeur critique zα",
                "Valeur p",
                "Niveau de signification α",
                "Erreur de type I (α)",
            ],
            "Valeur": [
                "μ ≥ 300",
                "μ < 300",
                f"{mu_0}",
                f"{mu_0:.2f}",
                f"{x_bar:.2f}",
                f"{x_bar:.2f}",
                f"{s:.2f}",
                f"{n}",
                f"{z_stat:.4f}",
                f"{z_critical_v:.4f}",
                f"{p_value_v:.4f}",
                f"{alpha_v}",
                f"{type_i_error}",
            ],
        }
    )
    return (
        alpha_v,
        hyp_test_df,
        mu_0,
        n,
        reject_h0_v,
        s,
        type_i_error,
        x_bar,
        z_critical_v,
        z_stat,
    )


@app.cell(hide_code=True)
def _(
    alpha_v,
    hours,
    hyp_test_df,
    mu_0,
    reject_h0_v,
    type_i_error,
    z_critical_v,
    z_stat,
):
    mo.md(rf"""
    ## Test d'hypothèse unilatéral à gauche

    **Formulation des hypothèses:**
    - H₀: μ ≥ {mu_0} minutes ({hours(mu_0):.2f} heures) - L'hypothèse du patron
    - H₁: μ < {mu_0} minutes - Le temps de jeu moyen est inférieur à 5 heures

    **Méthode:** Test Z (n ≥ 30, distribution normale)

    La statistique de test est:
    $$
    z = \frac{{\bar{{x}} - \mu_0}}{{s / \sqrt{{n}}}} = {z_stat:.4f}
    $$

    **Résultats:**

    {mo.ui.table(hyp_test_df, selection=None)}

    **Règle de décision:**

    On rejette H₀ si z < z_α = {z_critical_v:.4f}

    Puisque z = {z_stat:.4f} {"<" if z_stat < z_critical_v else "≥"} {z_critical_v:.4f}, on {"**rejette**" if reject_h0_v else "**ne rejette pas**"} H₀.

    **Conclusion:**

    Au niveau de signification α = {alpha_v}, {"les données fournissent suffisamment de preuves pour rejeter l'hypothèse du patron. Le temps de jeu moyen est significativement inférieur à 5 heures par semaine." if reject_h0_v else "les données ne fournissent pas suffisamment de preuves pour rejeter l'hypothèse du patron. On ne peut pas conclure que le temps de jeu moyen est inférieur à 5 heures par semaine."}

    **Erreur de première espèce (α):**

    L'erreur de type I est la probabilité de rejeter H₀ alors qu'elle est vraie. Ici, α = {type_i_error} = {type_i_error * 100}%.

    Cela signifie qu'il y a {type_i_error * 100}% de chance de conclure à tort que le temps de jeu moyen est inférieur à 5 heures, alors qu'en réalité il est d'au moins 5 heures.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VI) Erreur de deuxième espèce

    Supposons que la moyenne échantillonnale est un très bon estimé de la moyenne de la
    population et qu'elle peut être considérée comme étant cette dernière, quelle est l'erreur de
    deuxième espèce commise au point précédent si on ne rejette pas l'hypothèse nulle?
    """)
    return


@app.cell
def _(alpha_v, mu_0, n, s, x_bar):
    # Type II error (β) calculation
    # β = P(not rejecting H0 | H1 is true)
    # Assuming the true population mean is equal to the sample mean (x_bar)

    # True population mean (assumed to be the sample mean)
    mu_true = x_bar

    # Standard error
    se = s / np.sqrt(n)

    # Critical value for the test (x_c such that we reject H0 if x̄ < x_c)
    # From z_α = (x_c - μ₀) / se, we get x_c = μ₀ + z_α * se
    z_alpha = stats.norm.ppf(alpha_v)
    x_critical = mu_0 + z_alpha * se

    # Type II error: P(x̄ ≥ x_c | μ = μ_true)
    # β = P(Z ≥ (x_c - μ_true) / se)
    z_beta = (x_critical - mu_true) / se
    beta = 1 - stats.norm.cdf(z_beta)

    # Power of the test
    power = 1 - beta

    type_ii_df = pl.DataFrame(
        {
            "Paramètre": [
                "Vraie moyenne de la population μ (min)",
                "Moyenne hypothétisée μ₀ (min)",
                "Erreur standard",
                "Valeur critique x_c (min)",
                "z_β",
                "Erreur de type II (β)",
                "Puissance du test (1 - β)",
            ],
            "Valeur": [
                f"{mu_true:.2f}",
                f"{mu_0}",
                f"{se:.4f}",
                f"{x_critical:.2f}",
                f"{z_beta:.4f}",
                f"{beta:.4f}",
                f"{power:.4f}",
            ],
        }
    )
    return beta, mu_true, power, se, type_ii_df, x_critical, z_beta


@app.cell(hide_code=True)
def _(
    alpha_v,
    beta,
    hours,
    mu_0,
    mu_true,
    power,
    se,
    type_ii_df,
    x_critical,
    z_beta,
):
    mo.md(rf"""
    ## Calcul de l'erreur de deuxième espèce (β)

    L'erreur de type II (β) est la probabilité de ne pas rejeter H₀ alors que H₁ est vraie.

    **Hypothèses:**
    - On suppose que la vraie moyenne de la population est μ = x̄ = {mu_true:.2f} minutes ({hours(mu_true):.2f} heures)
    - H₀: μ ≥ {mu_0} minutes
    - H₁: μ < {mu_0} minutes

    **Calcul:**

    1. La valeur critique x_c est déterminée par:
    $$
    x_c = \mu_0 + z_\alpha \times SE = {mu_0} + {stats.norm.ppf(alpha_v):.4f} \times {se:.4f} = {x_critical:.2f}
    $$

    2. L'erreur de type II est:
    $$
    \beta = P(\bar{{X}} \geq x_c | \mu = \mu_{{vrai}}) = P\left(Z \geq \frac{{x_c - \mu_{{vrai}}}}{{SE}}\right) = P(Z \geq {z_beta:.4f}) = {beta:.4f}
    $$

    **Résultats:**

    {mo.ui.table(type_ii_df, selection=None)}

    **Interprétation:**

    - L'erreur de type II (β) = {beta:.4f} = {beta * 100:.2f}%
    - Cela signifie qu'il y a {beta * 100:.2f}% de chance de ne pas rejeter l'hypothèse du patron (μ ≥ 300 min) alors qu'en réalité la vraie moyenne est de {mu_true:.2f} minutes.
    - La puissance du test (1 - β) = {power:.4f} = {power * 100:.2f}%
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VII) Test d'hypothèse bilatéral sur la variance

    En supposant que l'écart-type des temps de jeu est de 50, effectuez un test d'hypothèse
    bilatéral sur la variance avec un seuil de signification de 5% afin d'évaluer si les données
    fournissent suffisamment de preuves pour rejeter ou ne pas rejeter l'hypothèse nulle.
    """)
    return


@app.cell
def _(dataset):
    # Bilateral hypothesis test on variance
    # H0: σ² = 50² = 2500 (population variance equals hypothesized value)
    # H1: σ² ≠ 2500 (population variance is different)

    sigma_0 = 50  # hypothesized population standard deviation
    sigma_0_squared = sigma_0**2  # hypothesized population variance
    alpha_vii = 0.05  # significance level

    # Sample statistics
    n_vii = len(dataset)
    s_squared_vii = dataset.var(ddof=1)  # sample variance (unbiased)
    s_vii = dataset.std(ddof=1)  # sample standard deviation

    # Test statistic: Chi-square
    # χ² = (n-1) * s² / σ₀²
    chi2_stat_vii = (n_vii - 1) * s_squared_vii / sigma_0_squared

    # Degrees of freedom
    df_vii = n_vii - 1

    # Critical values for two-tailed test
    chi2_lower = stats.chi2.ppf(alpha_vii / 2, df=df_vii)
    chi2_upper = stats.chi2.ppf(1 - alpha_vii / 2, df=df_vii)

    # P-value for two-tailed test
    # P-value = 2 * min(P(χ² < χ²_stat), P(χ² > χ²_stat))
    p_lower = stats.chi2.cdf(chi2_stat_vii, df=df_vii)
    p_upper = 1 - p_lower
    p_value_vii = 2 * min(p_lower, p_upper)

    # Decision
    reject_h0_vii = chi2_stat_vii < chi2_lower or chi2_stat_vii > chi2_upper

    variance_test_df = pl.DataFrame(
        {
            "Paramètre": [
                "Hypothèse nulle H₀",
                "Hypothèse alternative H₁",
                "Écart-type hypothétisé σ₀",
                "Variance hypothétisée σ₀²",
                "Taille de l'échantillon n",
                "Écart-type de l'échantillon s",
                "Variance de l'échantillon s²",
                "Degrés de liberté (n-1)",
                "Statistique χ²",
                "Valeur critique inférieure χ²_α/2",
                "Valeur critique supérieure χ²_1-α/2",
                "Valeur p",
                "Niveau de signification α",
            ],
            "Valeur": [
                "σ² = 2500",
                "σ² ≠ 2500",
                f"{sigma_0}",
                f"{sigma_0_squared}",
                f"{n_vii}",
                f"{s_vii:.2f}",
                f"{s_squared_vii:.2f}",
                f"{df_vii}",
                f"{chi2_stat_vii:.4f}",
                f"{chi2_lower:.4f}",
                f"{chi2_upper:.4f}",
                f"{p_value_vii:.4f}",
                f"{alpha_vii}",
            ],
        }
    )
    return (
        alpha_vii,
        chi2_lower,
        chi2_stat_vii,
        chi2_upper,
        df_vii,
        reject_h0_vii,
        s_squared_vii,
        sigma_0,
        sigma_0_squared,
        variance_test_df,
    )


@app.cell(hide_code=True)
def _(
    alpha_vii,
    chi2_lower,
    chi2_stat_vii,
    chi2_upper,
    df_vii,
    reject_h0_vii,
    s_squared_vii,
    sigma_0,
    sigma_0_squared,
    variance_test_df,
):
    mo.md(rf"""
    ## Test bilatéral sur la variance (test du Chi-deux)

    **Formulation des hypothèses:**
    - H₀: σ² = {sigma_0_squared} (σ = {sigma_0})
    - H₁: σ² ≠ {sigma_0_squared}

    **Méthode:** Test du Chi-deux pour la variance

    La statistique de test est:
    $$
    \chi^2 = \frac{{(n-1) \cdot s^2}}{{\sigma_0^2}} = \frac{{{df_vii} \cdot {s_squared_vii:.2f}}}{{{sigma_0_squared}}} = {chi2_stat_vii:.4f}
    $$

    **Résultats:**

    {mo.ui.table(variance_test_df, selection=None)}

    **Règle de décision:**

    On rejette H₀ si χ² < χ²_α/2 = {chi2_lower:.4f} ou χ² > χ²_1-α/2 = {chi2_upper:.4f}

    Puisque χ² = {chi2_stat_vii:.4f} {"est dans la région de rejet" if reject_h0_vii else "n'est pas dans la région de rejet"} [{chi2_lower:.4f}, {chi2_upper:.4f}], on {"**rejette**" if reject_h0_vii else "**ne rejette pas**"} H₀.

    **Conclusion:**

    Au niveau de signification α = {alpha_vii}, {"les données fournissent suffisamment de preuves pour rejeter l'hypothèse que l'écart-type de la population est de 50 minutes. La variance de la population est significativement différente de 2500." if reject_h0_vii else "les données ne fournissent pas suffisamment de preuves pour rejeter l'hypothèse nulle. On ne peut pas conclure que la variance de la population est différente de 2500."}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Extra) Variable aléatoire Q du temps de jeu

    On définit ici la variable aléatoire Q représentant le temps de jeu d'un joueur.
    """)
    return


@app.cell
def _(dataset):
    mu_q = dataset.mean()
    sigma_q = dataset.std(ddof=1)
    return mu_q, sigma_q


@app.cell(hide_code=True)
def _(mu_q, sigma_q):
    mo.md(rf"""
    ## Définition de la variable aléatoire Q

    Soit **Q** la variable aléatoire représentant le temps de jeu hebdomadaire d'un joueur (en minutes).

    En supposant que les temps de jeu suivent une distribution normale, on a :

    $$Q \sim \mathcal{{N}}(\mu, \sigma^2)$$

    Où :

    | Paramètre | Description | Valeur (min.) |
    |-----------|-------------|--------------|
    | $\mu$ | Moyenne | {mu_q:.2f} |
    | $\sigma$ | Écart-type | {sigma_q:.2f} |
    | $\sigma^2$ | Variance | {sigma_q**2:.2f} |

    Thus, the probability density function of Q is:

    $$f_Q(q) = \frac{{1}}{{\sigma\sqrt{{2\pi}}}} \exp\left(-\frac{{(q-\mu)^2}}{{2\sigma^2}}\right) = \frac{{1}}{{{sigma_q:.2f}\sqrt{{2\pi}}}} \exp\left(-\frac{{(q-{mu_q:.2f})^2}}{{2 \times {sigma_q**2:.2f}}}\right)$$
    """)
    return


if __name__ == "__main__":
    app.run()
