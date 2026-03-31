import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Introduction

    Dans le contexte actuel du développement de jeux vidéo en ligne, la conception de systèmes interactifs repose largement sur l’utilisation de modèles probabilistes, d’analyses statistiques et de simulations numériques. Ces outils permettent non seulement de structurer les mécaniques de jeu, mais également d’assurer un équilibre entre défi et accessibilité, tout en optimisant les performances techniques des infrastructures sous-jacentes.

    Dans cette optique, l’entreprise ZeldUS, spécialisée dans la création de jeux vidéo, cherche à intégrer ces approches quantitatives afin de mieux comprendre et contrôler certains aspects clés de son produit, notamment les probabilités d’obtention de récompenses, le comportement des joueurs et la gestion des ressources serveur. L’objectif de ce rapport est donc d’appliquer des méthodes issues des probabilités et des statistiques pour analyser différentes situations concrètes rencontrées lors du développement du jeu.

    Pour ce faire, le rapport est structuré en trois parties. La première porte sur l’étude de modèles probabilistes liés à des mécaniques de jeu et à un système de visée. La deuxième traite de l’analyse statistique descriptive et inférentielle des temps de jeu des utilisateurs. Enfin, la troisième partie présente une simulation de type Monte-Carlo visant à estimer le nombre moyen de joueurs connectés simultanément aux serveurs.
    """)
    return


if __name__ == "__main__":
    app.run()
