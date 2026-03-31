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


@app.function
def generate_arrival_times(arrivals_per_minutes, sample_size):
    uniform_randoms = np.random.rand(sample_size)
    return -np.log(uniform_randoms) / arrivals_per_minutes


@app.function
def generate_play_durations(average, variance, sample_size):
    return np.random.normal(average, variance, sample_size)


@app.cell
def _():
    tested_arrivals_per_minutes = [10, 67, 100] # Taux d'arrivés testés
    MU_Q = 280.56                               # Moyenne de temps de jeux d'un joueur
    SIGMA_Q = 50.38                             # Variance du temps de jeux
    sample_size = 10000                         # N
    return MU_Q, SIGMA_Q, sample_size, tested_arrivals_per_minutes


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


        play_time = generate_play_durations(
            MU_Q,
            SIGMA_Q,
            len(arrival_times))
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
        active_at_t = {ti: np.sum((arrivals <= ti) & (arrivals + play_time >= ti)) for ti in t_check}

        # Sauvegarde des résultats
        results.append({
            "rate": rate,
            "sample_size": sample_size,
            "moyenne_theorique": theorical_average,
            "moyenne_reel": real_average,
            "diff": real_average - theorical_average,
            "peak": np.max(active_players),
            "active_at_t": active_at_t,
            "arrivals": arrivals,
            "play_time": play_time
        })
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
