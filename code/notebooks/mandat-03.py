# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.20.2",
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(
    width="full",
    app_title="Simulation de joueurs actifs par minute",
)

with app.setup:
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo


@app.cell
def _():
    # Taux d'arrivée à tester (joueurs par minute)
    tested_arrivals_per_minutes = [1, 5, 10]
    MU_Q = 20  # durée moyenne de jeu en minutes
    sample_size = 10000

    mo.md(r"""
    # Simulation des joueurs actifs

    Nous allons simuler le nombre de joueurs actifs à différents taux d'arrivée et comparer
    la moyenne réelle observée à la moyenne théorique.
    """)
    return MU_Q, sample_size, tested_arrivals_per_minutes


@app.cell
def _(MU_Q):
    def simulate(rate, N):
        """
        Simule N joueurs arrivant selon un taux `rate` (par minute)
        et jouant un temps selon une distribution normale.
        """
        arrivals = np.random.exponential(1 / rate, N)  # en minutes
        play_time = np.random.normal(MU_Q, MU_Q / 4, N)  # écart-type arbitraire
        play_time = np.clip(play_time, 0, None)
        t = np.arange(0, np.max(arrivals + play_time) + 1)
        active_players = [np.sum((arrivals <= ti) & (arrivals + play_time >= ti)) for ti in t]
        return arrivals, play_time, t, active_players

    return (simulate,)


@app.cell
def _(MU_Q, sample_size, simulate, tested_arrivals_per_minutes):
    results = []

    for rate in tested_arrivals_per_minutes:
        arrivals, play_time, t, active_players = simulate(rate, sample_size)
        moyenne_theorique = rate * MU_Q
        moyenne_reel = np.mean(active_players)

        # Points à différents instants
        t_check = [10, 50, 500]
        active_at_t = {ti: np.sum((arrivals <= ti) & (arrivals + play_time >= ti)) for ti in t_check}

        # Sauvegarde des résultats
        results.append({
            "rate": rate,
            "sample_size": sample_size,
            "moyenne_theorique": moyenne_theorique,
            "moyenne_reel": moyenne_reel,
            "diff": moyenne_reel - moyenne_theorique,
            "peak": np.max(active_players),
            "active_at_t": active_at_t,
            "arrivals": arrivals,
            "play_time": play_time
        })
    return (results,)


@app.cell(hide_code=True)
def _(results):
    for _r in results:
        mo.md(rf"""
        ## Taux: {_r['rate']} joueurs / minute
        - Nombre de joueurs simulés: {_r['sample_size']}
        - Moyenne réelle de joueurs actifs: {_r['moyenne_reel']:.2f}
        - Moyenne théorique: {_r['moyenne_theorique']:.2f}
        - Différence moyenne réel vs théorique: {_r['diff']:.2f}
        - Peak simultané: {_r['peak']}
        - Joueurs actifs à différents instants:
            {"".join([f"- t={ti} min : {n}\n" for ti, n in _r['active_at_t'].items()])}
        """)
    return


@app.cell
def _(MU_Q, results):
    for _r in results:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.hist(_r['arrivals'], bins=50, density=True)
        plt.title(f"Arrivées (λ={_r['rate']}/min)")

        plt.subplot(1,2,2)
        plt.hist(_r['play_time'], bins=50, density=True)
        plt.title(f"Temps de jeu Q (μ={MU_Q} min)")

        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conclusion de la simulation

    Nous avons simulé différents taux d'arrivée de joueurs et observé:
    - Comment la moyenne réelle se rapproche ou diverge de la moyenne théorique.
    - Les pics de joueurs simultanés.
    - Les distributions des arrivées et des temps de jeu.
    """)
    return


if __name__ == "__main__":
    app.run()
