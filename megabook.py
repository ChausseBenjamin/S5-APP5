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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Introduction

    Dans le contexte actuel du développement de jeux vidéo en ligne, la conception de systèmes interactifs repose largement sur l’utilisation de modèles probabilistes, d’analyses statistiques et de simulations numériques. Ces outils permettent non seulement de structurer les mécaniques de jeu, mais également d’assurer un équilibre entre défi et accessibilité, tout en optimisant les performances techniques des infrastructures sous-jacentes.

    Dans cette optique, l’entreprise ZeldUS, spécialisée dans la création de jeux vidéo, cherche à intégrer ces approches quantitatives afin de mieux comprendre et contrôler certains aspects clés de son produit, notamment les probabilités d’obtention de récompenses, le comportement des joueurs et la gestion des ressources serveur. L’objectif de ce rapport est donc d’appliquer des méthodes issues des probabilités et des statistiques pour analyser différentes situations concrètes rencontrées lors du développement du jeu.

    Pour ce faire, le rapport est structuré en trois parties. La première porte sur l’étude de modèles probabilistes liés à des mécaniques de jeu et à un système de visée. La deuxième traite de l’analyse statistique descriptive et inférentielle des temps de jeu des utilisateurs. Enfin, la troisième partie présente une simulation de type Monte-Carlo visant à estimer le nombre moyen de joueurs connectés simultanément aux serveurs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mandat no. 1 : Probabilités
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Partie 1: Roues de pouvoirs
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### i)
    Le pouvoir est obtenu si toutes les roues s'arrêtent sur le même pictogramme. Deux configurations sont testés:
    - 3 roues avec 8 pictogrammes chacune
    - 4 roues avec 5 pictogrammes chacune

    C'est un cas classique de probabilités basé sur le principe fondamental du dénombrement, avec des événements indépendants.

    #### Premier cas : 3 roues, 8 pictogrammes

    L'espace échantillonnal contient 8 pictogrammes.
    L'événement recherché est d'obtenir le même pictogramme sur les 3 roues.

    La probabilité d'obtenir un pictogramme spécifique sur les 3 roues est:
    $$
    \frac{1}{8} \times \frac{1}{8} \times \frac{1}{8} = \frac{1}{512} \approx 0.195\%
    $$

    Cependant, c'est un seul pictogramme et il y en a 8 possibles, la probabilité d'obtenir un pouvoir (peu importe lequel) est
    donc en fonction du nombre total de possibilités $8^3$ et du nombre de possibilités valides ($8$):
    $$
    \frac{8}{512} \approx 1.56\%
    $$

    #### Deuxième cas : 4 roues, 5 pictogrammes
    Même chose qu'en haut:
    La probabilité d'obtenir un pictogramme spécifique sur les 4 roues est:
    $$
    \frac{1}{5} \times \frac{1}{5} \times \frac{1}{5} \times \frac{1}{5} = \frac{1}{625} = 0.16\%
    $$

    Pour les 5 possibles, la probabilité d'obtenir un pouvoir c'est:
    $$
    \frac{5}{625} \approx 0.8\%
    $$

    #### Résultats
    La configuration avec 3 roues et 8 pictogrammes offre une probabilité plus élevée d'obtenir un pouvoir.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ii)
    C'est le cas où un pouvoir est obtenue si toutes les roues affichent des pictogrammes différents.

    #### Permutations ou combinaisons
    Pour déterminer le nombre de possibilités, il faut choisir entre permutations ou combinaisons.
    Une combinaison ne tient pas compte de l'ordre, tandis qu'une permutation en tient compte.

    Les permutations sont utilisé puisque l'ordre est important: par exemple, (A, B, C) et (C, A, B) sont deux résultats différents.
    $$
    P(n,r)=\frac{n!}{(n-r)!}
    $$

    où $n$ est le nombre de pictogrammes possibles et $r$ le nombre de roues.

    #### Premier cas : 3 roues, 8 pictogrammes
    $$
    P(8,3)=\frac{8!}{(8-3)!}=\frac{8!}{5!}=336
    $$
    Il y a donc 336 arrangements où les pictogrammes sont tous différents.

    Sachant que le nombre total d'arrangement est $8^3=512$, La probabilité d'obtenir uniquement des pictogrammes différents est donc:
    $$
    \frac{336}{512} \approx 65.6\%
    $$

    #### Deuxième cas : 4 roues, 5 pictogrammes
    $$
    P(5,4)=\frac{5!}{(5-4)!}=\frac{5!}{1!}=120
    $$
    Il y a donc 120 arrangements où les pictogrammes sont tous différents.

    Sachant que le nombre total d'arrangement est $5^4=625$, La probabilité d'obtenir uniquement des pictogrammes différents est donc:
    $$
    \frac{120}{625} \approx 19.2\%
    $$

    #### Résultats
    Il est beaucoup plus facile d'obtenir des pictogrammes tous différents dans le premier cas que dans le deuxième.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### iii)

    Dans ce cas, on obtient un pouvoir si toutes les roues affichent le même pictogramme au moins 2 fois sur 5 essais.
    Les succès peuvent être différents (par exemple : (AAA) et (BBB) sont valides).

    On modélise la situation avec une variable aléatoire $X$ représentant le nombre de succès (Nombre de fois où toutes les roues affichent le même pictogramme).

    On utilise la loi binomiale, car :
    - le nombre d'essais est fixé ($n = 5$)
    - chaque essai a deux issues (succès ou échec)
    - la probabilité de succès est constante
    - les essais sont indépendants

    La fonction de masse de la loi binomiale c'est:
    $$
    P(X=k)=\binom{n}{k} p^k (1-p)^{n-k}
    $$

    où:
    $$
    \binom{n}{k}=\frac{n!}{k!(n-k)!}
    $$

    On cherche:
    $$
    P(X \ge 2)
    $$

    #### Probabilitées
    ##### Premier cas: (3 roues, 8 pictogrammes)
    La probabilité de succès d'avoir que des pictogrammes identiques à déjà été calculé en haut. C'est $1.56\%$.

    On calcul les probabilitées d'en avoir moins que 2. C'est plus court et le résultat n'a qu'a être inversé.
    $$
    P(X=0)=\frac{5!}{0!(5-0)!}0.0156^0(1-0.0156)^{5-0}=0.92439593103
    $$
    $$
    P(X=1)=\frac{5!}{1!(5-1)!}0.0156^1(1-0.0156)^{5-1}=0.07324551261
    $$

    l'inverse, est donc:
    $$
    P(X \ge 2)=1 - (P(X=0)+P(X=1))
    $$
    $$
    P(X \ge 2)=1 - (0.9244 + 0.0732) \approx 0.0024
    $$
    Soit environ $0.24\%$

    ##### Deuxième cas: (4 roues, 5 pictogrammes)
    La probabilité de succès d'avoir que des pictogrammes identiques à déjà été calculé en haut. C'est $0.8\%$.
    On calcul les probabilitées d'en avoir moins que 2. C'est plus court et le résultat n'a qu'a être inversé.
    $$
    P(X=0)=\frac{5!}{0!(5-0)!}0.008^0(1-0.008)^{5-0}=0.96063490044
    $$
    $$
    P(X=1)=\frac{5!}{1!(5-1)!}0.008^1(1-0.008)^{5-1}=0.03873527824
    $$

    l'inverse, est donc:
    $$
    P(X \ge 2)=1 - (P(X=0)+P(X=1))
    $$
    $$
    P(X \ge 2)=1 - (0.9606 + 0.0387) \approx 0.0006
    $$
    Soit environ: $0.06\%$

    #### Variance
    Pour une loi binomiale, la variance est donnée par:
    $$
    \mathrm{Var}(X)=n \cdot p (1-p)
    $$

    ##### Premier cas:
    $$
    \mathrm{Var}(X)=5 \cdot 0.0156 \cdot (1-0.0156) \approx 0.0768
    $$

    ##### Deuxième cas:
    $$
    \mathrm{Var}(X)=5 \cdot 0.008 \cdot (1-0.008) \approx 0.0397
    $$

    #### Espérance
    L'espérance de la loi binomiale c'est:
    $$
    E[X]=n \cdot p
    $$

    ##### Premier cas:
    $$
    E[X]=5 \cdot 0.0156 \approx 0.078
    $$

    ##### Deuxième cas:
    $$
    E[X]=5 \cdot 0.008 \approx 0.04
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Partie 2: Probabilitées de la cible
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Covariance entre $X$ et $Y$

    Intuitivement, il n'y a pas de relation entre $X$ et $Y$. Un déplacement selon l'axe $X$ n'affecte pas la position selon $Y$, et inversement.
    On suppose donc que :
    $$
    \mathrm{Cov}(X,Y) = 0
    $$

    #### Corrélation entre $X$ et $Y$
    La corrélation est donnée par :
    $$
    \rho_{XY}=\frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
    $$
    Puisque la covariance est nulle :
    $$
    \rho_{XY} = 0
    $$
    Dans le cas d'une loi normale bivariée, une corrélation nulle implique que les variables sont indépendantes. Donc, $X$ et $Y$ sont indépendantes.

    #### Fonction de densité de probabilité
    La fonction de densité de probabilité conjointe pour deux variables indépendantes $X$ et $Y$ (loi normale bivariée) s'écrit:
    $$
    f_{X,Y}(x,y|z) = f_X(x) \cdot f_Y(y)
    $$

    La PDF d'une loi normale univariée est :
    $$
    f(x) = \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

    Donc, pour la distribution conjointe :

    $$
    f_{X,Y}(x,y|z) =
    \frac{1}{\sigma_X \sqrt{2\pi}} \, e^{-\frac{(x-\mu_X)^2}{2\sigma_X^2}}
    \times
    \frac{1}{\sigma_Y \sqrt{2\pi}} \, e^{-\frac{(y-\mu_Y)^2}{2\sigma_Y^2}}
    $$

    Comme $\mu_X = \mu_Y = 0$, ça se simplifie:

    $$
    f_{X,Y}(x,y|z) =
    \frac{1}{\sigma_X \sqrt{2\pi}} \, e^{-\frac{x^2}{2\sigma_X^2}}
    \times
    \frac{1}{\sigma_Y \sqrt{2\pi}} \, e^{-\frac{y^2}{2\sigma_Y^2}}
    $$

    À noter : $\sigma_Y$ dépend de la variable $z$ (distance par rapport à la cible).

    #### Fonction de densité de probabilité marginale
    Comme $X$ et $Y$ sont indépendant, il est possible d'obtenir les PDF marginales simplement en “coupant” la PDF conjointe.
    ##### Pour $X$:
    $$
    f_X(x) =
    \frac{1}{\sigma_X \sqrt{2\pi}} \, e^{-\frac{x^2}{2\sigma_X^2}}
    $$
    Avec $\sigma_X = 0.1$:
    $$
    f_X(x) =
    \frac{1}{0.1 \sqrt{2\pi}} \, e^{-\frac{x^2}{0.02}}
    $$
    Ce qui donne:
    $$
    f_X(x) \approx 4 \, e^{-\frac{x^2}{0.02}}
    $$

    ##### Pour $Y$:
    Pour $Y$, on garde le $\sigma$ puisqu'il peut changé en fonction de $z$:
    $$
    f_Y(y) =
    \frac{1}{\sigma_Y \sqrt{2\pi}} \, e^{-\frac{y^2}{2\sigma_Y^2}}
    $$
    $$
    f_Y(y|z) =
    \frac{1}{\sigma_Y(z) \sqrt{2\pi}} \, e^{-\frac{y^2}{2\sigma_Y(z)^2}}
    $$

    #### Début de la probabilité d'ouvrir la porte à moins d'un mètre
    Pour réussir, il faut que les coordonnées du lancé tombent à l'intérieur d'un cercle de rayon de $\le 0.1$.
    La condition est simple grâce au théorème de Pythagore:
    $$
    X^2 + Y^2 \le R^2
    $$
    $$
    X^2 + Y^2 \le 0.1^2
    $$

    Comme on connaît $\sigma_Y$ pour cette distance, la PDF de $Y$ peut être simplifiée :
    $$
    f_Y(y|z) = \frac{1}{\sigma_Y(z) \sqrt{2\pi}} \, e^{-\frac{y^2}{2\sigma_Y(z)^2}}
    $$
    $$
    f_Y(y) = \frac{1}{0.05 \sqrt{2\pi}} \, e^{-\frac{y^2}{2\cdot 0.05^2}} \approx 8 \, e^{-\frac{y^2}{0.005}}
    $$

    ##### Pourquoi on ne fait pas l’intégrale à la main

    La probabilité qu'une variable continue tombe dans une région donnée se calcule avec son **CDF**, ça implique une **intégrale**.
    Dans notre cas, il s'agit d'une double intégrale quand même assez complex:

    $$
    P(X,Y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx \, dy
    $$

    $X$ et $Y$ étant indépendant,la PDF conjointe se factorise:
    $$
    F_X(x) = \int_{-\infty}^{\infty} f_X(x) \, dx, \quad
    F_Y(y) = \int_{-\infty}^{\infty} f_Y(y) \, dy
    $$

    Les intégrales restent lourds à faire à la main. Donc Monte-Carlo est utilisé, vue que la condition de réussite
    est très simple, les distributions, variances et moyennes sont connues.

    ##### Estimation par simulation Monte-Carlo
    Avec Monte-Carlo, on peut estimer la probabilité beaucoup plus facilement. La probabilité devient simplement :
    $$
    P \approx \frac{\text{Nombre de succès}}{N}
    $$

    Plus le nombre de tirages $N$ est grand, plus l'estimation est précise.
    De plus, NumPy à déjà une fonction pour généré des valeurs aléatoires dans une distribution normale.

    Hors le processus est simple. N points aléatoire distribués normalements avec une moyenne de $0$ sont généré et avec la variance connue.
    Ensuites, ils sont comparé avec la condition de réussite, qui est que la distance entre le centre du cercle et la coordonée soit égal ou
    moindre que $0.1$. Le taux de succès sur le nombre total d'essaie donne la probabilité de réussite.
    """)
    return


@app.cell(hide_code=True)
def fonctions():
    def random_value(variance, N):
        return np.random.normal(0, variance, N)

    def is_in_circle(x, y, radius):
        return (x**2 + y**2) <= radius**2

    return is_in_circle, random_value


@app.cell
def program(is_in_circle, random_value):
    def _():
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
        return print("z>10: ", probability)


    _()
    return


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

    Cette section calcule les statistiques descriptives des temps de jeu (moyenne, médiane, mode, écart-type, variance, min, max, étendue) conformément à la première exigence du mandat.

    {mo.ui.table(df, selection=None)}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # II) Construction de l'histogramme des données

    Cette section construit l'histogramme des temps de jeu en utilisant la loi de Sturges pour déterminer le nombre de classes, puis calcule les fréquences, fréquences relatives et fréquences cumulées.
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

    Cette section vérifie si les temps de jeu suivent une distribution normale en utilisant un test de Khi-deux d'ajustement, comme exigé dans la troisième partie du mandat. En observant l'histogramme ci-dessus, on peut voir que les données semblent suivre une distribution approximativement normale (forme de cloche). Pour le confirmer quantitativement, nous utilisons le test de Khi-deux.
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

    Cette section calcule l'intervalle de confiance à 95% pour la moyenne des temps de jeu en utilisant la distribution normale centrée réduite, comme requis dans la quatrième partie du mandat.

    Calculez l'intervalle de confiance pour la moyenne des temps de jeu avec un niveau de
    confiance de 95%. Vous ferez alors usage des tables de la distribution normale centrée réduite.
    """)
    return


@app.cell
def _(dataset):
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

    Cette section effectue un test d'hypothèse unilatéral à gauche sur la moyenne pour vérifier si le temps de jeu moyen est significativement inférieur aux 5 heures (300 minutes) estimées par le patron, comme requis dans la cinquième partie du mandat.

    Effectuez un test d'hypothèse approprié sur la moyenne afin d'évaluer si les données
    fournissent suffisamment de preuves pour rejeter ou ne pas rejeter l'hypothèse de votre
    patron énoncée précédemment. Considérez un niveau de confiance de 95%. Quelle est ici
    l'erreur de première espèce?
    """)
    return


@app.cell
def _(dataset):
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

    Cette section calcule l'erreur de type II (β) en supposant que la moyenne échantillonnale est un bon estimé de la vraie moyenne de la population, comme demandé dans la sixième partie du mandat.

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

    Cette section effectue un test du chi-deux bilatéral sur la variance pour vérifier si l'écart-type des temps de jeu est significativement différent de 50 minutes, comme exigé dans la septième partie du mandat.

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

    Cette section définit la variable aléatoire Q représentant le temps de jeu hebdomadaire d'un joueur, qui sera utilisée dans le mandat 3 pour les simulations Monte-Carlo.

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mandat no. 3 : Simulations Monte-Carlo
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fonctions des temps d'arrivés
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Processus de poisson et exponentielle.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    La fonction de masse de la loi de Poisson est donnée par :
    $$
    P(X=x)=\frac{e^{-\lambda T}(\lambda T)^x}{x!}
    $$
    - Où $\lambda$ est le taux d'arrivée.
    - T est une durée de temps.


    Cette fonction donne la probabilité que $x$ arrivées surviennent dans un intervalle de temps $T$. Cependant, ici on s'intéresse plutôt au *temps entre chaque arrivée*.
    Si $x$ représente le nombre d'arrivées dans l'intervalle $[0,T]$, posé $x=0$ devrait donnée la probabilité qu'aucune arrivée ne survienne dans cet intervalle.

    Autrement dit, l'absence d'événement dans $[0,T]$ c'est la probabilité que le temps jusqu'à la prochaine arrivée soit plus grand que $T$:
    $$
    P(T_A > T)
    $$

    $x$ est donc remplacé par $0$ dans la formule de Poisson, et c'est simplifié:
    $$
    P(X=0)=\frac{e^{-\lambda T}(\lambda T)^0}{0!}
    $$
    $$
    P(X=0)=\frac{e^{-\lambda T}\cdot 1}{1}
    $$
    $$
    P(X=0)=e^{-\lambda T}
    $$

    Ça donne une fonction exponentielle décroissante. Plus $T$ augmente, plus la probabilité diminue.
    C'est la fonction de survie:
    $$
    P(T_A > T) = e^{-\lambda T}
    $$

    La CDF s'obtient en prenant le complément, étant donnée la présence d'un $>$ dans la formule de survie.
    $$
    F(T)=P(T_A \le T)=1 - e^{-\lambda T}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Inversion de la fonction
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Chaque joueur arrive à un temps $P$, puis quitte après une session de jeu de durée aléatoire $Q$. Donc, chaque joueur est actif dans un intervalle $[P, P+Q]$.

    L'activité d'un joueur est donc:
    $$
    P \le t \le P + Q
    $$

    Ça permet de déterminer combien de joueurs sont actifs à n'importe quel instant (minute) $t$.
    La génération des temps d'arrivée est fait avec la méthode de transformation inverse.
    Il faut donc appliqué une transformation inverse sur la CDF trouvé plus haut. On résoud pour $T$.
    $$
    F(T)=1 - e^{-\lambda T}
    $$
    $$
    U = 1 - e^{-\lambda T}
    $$
    $$
    U - a = e^{-\lambda T}
    $$
    $$
    \ln(1 - U) = -\lambda T
    $$
    $$
    T = \frac{\ln(1 - U)}{-\lambda}
    $$
    Ici, $U$ est une variable aléatoire continue uniforme entre $[0,1]$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Python
    La fonction python suivante génère les temps d'arrivés $P$ en utilisant la transformation inverse trouvé plus haut.
    Elle prend comme paramètres le taux d'arrivés par minutes et le nombre de données à générés.
    """)
    return


@app.function
def generate_arrival_times(arrivals_per_minutes, sample_size):
    uniform_randoms = np.random.rand(sample_size)
    return -np.log(uniform_randoms) / arrivals_per_minutes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Temps de jeux des joueurs
    La variable aléatoire normale $Q$ a été déterminée dans le mandat 2, avec sa moyenne et sa variance.
    Elle dicte le nombre d'heures qu'un joueur passe sur le jeux.

    La variable est indépendante du temps d'arrivé ou tout autres valeurs associés au mandat-03.

    Pour la générer l'implémentation manuel de l'algorithme de Box-Muller est fesable, mais c'est inutile,
    puisque NumPy a déjà une implémentation dans la fonction `randn`, qui à été confirmé comme équivalente
    lors des laboratoires.
    """)
    return


@app.function
def generate_play_durations(average, variance, sample_size):
    return np.random.normal(average, variance, sample_size)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Paramètres de simulation
    """)
    return


@app.cell
def _():
    tested_arrivals_per_minutes = [
        10,
        67,
        100,
    ]  # Taux d'arrivés testés (branchements / minutes)
    MU_Q = 280.56  # Moyenne de temps de jeux d'un joueur (minutes)
    SIGMA_Q = 50.38  # Variance du temps de jeux (minutes)
    sample_size = 10000  # N (minutes)
    return MU_Q, SIGMA_Q, sample_size, tested_arrivals_per_minutes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fonction de simulation
    Avec le taux de branchements, la transformation inverse calculé plus haut est utilisé pour
    donnée un nombre de nouveau joueurs à chaque minutes de la simulation.
    Leurs temps d'arrivé est calculé progressivement au fur et à mesure que le temps avance.
    Par la suite, un nombre de temps de jeux est généré pour chaque branchements.
    Le temps de départ du joueur est ensuite calculé.

    Le nombre de joueurs actif à n'importe quel instant $t$ est finalement calculé en appliquant
    $$
        T_{\text{arrivé}} \le t \le T_{\text{départ}}
    $$
    """)
    return


@app.cell
def _(MU_Q, SIGMA_Q):
    def simulate(arrivals_per_minutes, simulation_time):
        arrival_times = []
        t = 0

        # Generating a P for each minute of the simulation
        # While making the arrivals later and later.
        while t < simulation_time:
            t += generate_arrival_times(arrivals_per_minutes, 1)[0]
            arrival_times.append(t)
        arrival_times = np.array(arrival_times)

        play_time = generate_play_durations(MU_Q, SIGMA_Q, len(arrival_times))
        departures = arrival_times + play_time

        # This removes the warmup time of the population of servers.
        # Without this, the averages are off because they assume infinite
        # amount of time.
        t_start = 0.3 * simulation_time  # remove transient phase
        t_values = np.linspace(t_start, simulation_time, 500)

        active_players = []
        for t in t_values:
            active = np.sum((arrival_times <= t) & (departures >= t))
            active_players.append(active)

        return arrival_times, play_time, t_values, active_players

    return (simulate,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Roulement de la simulation
    Le code suivant éxecute la simulation monte-carlo pour chaque taux de branchements voulu.
    Elle vient sauvegarder certaines valeurs analysé afin d'être utilisé dans les histogrammes
    et des affichages.

    Elle prouve aussi la réponse au besoin d'être capable de dire combiens de joueurs sont connecté
    à un instant $t$, et fait une comparaison de la moyenne théorique ($\lambda\times\mu_Q$) versus réel.
    """)
    return


@app.cell
def _(MU_Q, sample_size, simulate, tested_arrivals_per_minutes):
    results = []

    # Testing every rates set in the test array up in this document
    for rate in tested_arrivals_per_minutes:
        arrivals, play_time, t, active_players = simulate(rate, sample_size)
        theorical_average = rate * MU_Q
        real_average = np.mean(active_players)

        # Points à différents instants
        t_check = [10, 50, 500]
        active_at_t = {
            ti: np.sum((arrivals <= ti) & (arrivals + play_time >= ti))
            for ti in t_check
        }

        # Sauvegarde des résultats
        results.append(
            {
                "rate": rate,
                "sample_size": sample_size,
                "moyenne_theorique": theorical_average,
                "moyenne_reel": real_average,
                "diff": real_average - theorical_average,
                "peak": np.max(active_players),
                "active_at_t": active_at_t,
                "arrivals": arrivals,
                "play_time": play_time,
            }
        )
    return (results,)


@app.cell(hide_code=True)
def _(results):
    report = ""
    for _r in results:
        report += f"""## {_r["rate"]} branchements / minute
        - Temps de la simulation:\t{_r["sample_size"]} minutes
        - Moyennes de joueurs actifs:
            - Échantillon / réel: {_r["moyenne_reel"]:.2f}
            - Théorique:          {_r["moyenne_theorique"]:.2f}
            - Différence:         {_r["diff"]:.2f}
        - Peak de joueurs: {_r["peak"]}
        - Joueurs actifs à différents instants:
        {"".join([f"\t- t={ti} min : {n}\n" for ti, n in _r["active_at_t"].items()])}\n"""
    mo.md(report)
    return


@app.cell
def _(MU_Q, results, sample_size):
    for _r in results:
        P = generate_arrival_times(_r["rate"], sample_size)
        # Histogram of generated inter-arrival times
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(P, bins=50, density=True, label="Generated P")
        # plt.hist(_r['arrivals'], bins=50, density=True)
        plt.title(f"Arrivées (λ={_r['rate']}/min)")

        plt.subplot(1, 2, 2)
        plt.hist(_r["play_time"], bins=50, density=True)
        plt.title(f"Temps de jeu Q (μ={MU_Q} min)")

        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusion de la simulation
    C'est une belle simulation, mais malheureusement elle est très loin d'une situation réel.
    En effet, elle assume un taux de branchement constant à travers la journée, tandis que le taux change en fonction de l'heure.
    Elle assume aussi une moyenne de temps de jeux constant, alors qu'en réalité, les gens jouent plus les fins de semaines que les journées de travail.

    Tout ceci ne prend pas en compte le plus gros problème potentiel. Lorsqu'un jeux est mis sur le marché, on observe une fonction décroissante du
    nombre de joueur à travers le temps, avec un énorme montant de joueurs au tout début de la mise en marché.
    S'ils utilisent des moyennes pour quantifié leurs besoins en serveurs, la réalité vas vite les rattrapés lors du lancement du jeux.

    Ils ont besoin au minimum du double de l'infrastructure prévus pour les premières semaines de la mises en marché, avec possibilités d'aggrendissements
    temporaires lors d'évènements saisoniers ou des periodes de fêtes nationaux.

    En bref, c'est une bonne simulation de moyenne, mais ça ne devrait pas être la seule métrique de prise de décisions pour l'infrastructure de serveurs.
    """)
    return


if __name__ == "__main__":
    app.run()
