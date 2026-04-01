# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy>=2.4.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mandat no. 1 : Probabilités
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partie 1: Roues de pouvoirs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
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
def _(mo):
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
def _(mo):
    mo.md(r"""
    ## Partie 2: Probabilitées de la cible
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
