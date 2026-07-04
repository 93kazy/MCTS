# Projet MCTS — gestion énergétique d'une centrale avec stockage

Application de la recherche arborescente de Monte-Carlo (**MCTS**) à un problème
d'arbitrage stockage : à chaque heure, une centrale décide de vendre ou stocker
sa production pour maximiser son profit sur l'horizon, à partir des prix
day-ahead France (EPEX SPOT). Projet académique (M2).

## Structure

```
Projet_MCTS/
├── MCTS_deterministe/     # environnement, heuristiques + optimum (PD), MCTS ; données réelles
├── MCTS_stochastique/     # modèle de prix markovien, SDP, équivalent-certain, MCTS stochastique
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note structure.** Les modules stochastiques et le chargeur de données réutilisent
> `environment.py` (et `baselines.py`). Pour que chaque dossier tourne seul, garde une
> copie de ces modules partagés dans chaque dossier, ou regroupe tout dans un seul
> package. Évite aussi les accents dans les noms de dossiers (`deterministe` plutôt que
> `déterministe`) pour la portabilité entre OS.

## Installation (Windows, environnement virtuel)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(macOS / Linux : `source .venv/bin/activate` à la place de la 2e ligne.)

## Utilisation

```bat
python run_demo.py              :: version déterministe (données synthétiques)
python run_demo_stochastic.py   :: évaluation Monte-Carlo (prix incertains)
python data_loader.py           :: test du chargeur de données réelles
python run_demo_real.py         :: comparaison sur vraies données France
```

## Données

- **Prix day-ahead France** — [energy-charts.info](https://energy-charts.info) (Fraunhofer ISE),
  rediffusés en **CC BY 4.0** (source : Bundesnetzagentur | SMARD.de).
- **Production solaire France** — [ODRÉ / éCO2mix](https://odre.opendatasoft.com) (RTE).
- Alternative « officielle » : ENTSO-E Transparency (client `entsoe-py`, token gratuit).

Les séries sont récupérées à la volée (rien n'est versionné). **Licence à respecter :**
les prix EPEX SPOT restent propriété d'EPEX SPOT SE — usage non commercial, avec mention
de la source. Citer energy-charts.info / SMARD.de et RTE/ODRÉ dans le mémoire.

## Résultats en bref

Sur une semaine réelle de juin 2024 (prix négatifs de midi liés au solaire), en part de
la valeur d'arbitrage captée (optimum = 100 %) : « tout vendre » 0 %, seuil ~44 %,
**MCTS à rollout informé ~96 %**. En stochastique, le MCTS rejoint l'optimum causal (SDP)
en n'utilisant qu'un modèle génératif ; l'équivalent-certain reste une base forte, et
l'intérêt propre du MCTS est sa généralité (il se passe d'un modèle explicite et passe à
l'échelle sur des modèles de prix riches).

## Licence

Code sous licence MIT (voir `LICENSE`). Les données ne sont pas couvertes par cette
licence (voir la section Données ci-dessus).
