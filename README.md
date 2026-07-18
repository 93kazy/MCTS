# Projet MCTS — gestion énergétique d'une centrale avec stockage

Application de la recherche arborescente de Monte-Carlo (**MCTS**) à un problème
de gestion énergétique : à chaque heure, une centrale décide de **vendre**,
**stocker** ou **consommer** l'électricité qu'elle produit, afin de maximiser
son profit sur l'horizon, à partir des prix day-ahead France (EPEX SPOT).
Projet académique (M2).

Le recours à MCTS se justifie par l'explosion combinatoire des séquences de
décisions : avec les 3 actions du sujet, 3^24 ≈ 2,8 × 10¹¹ séquences sur une
journée ; avec la grille de débits de stockage utilisée en pratique (11
actions), 11^24 ≈ 9,8 × 10²⁴ — l'énumération exhaustive est impraticable, alors
que le MCTS concentre l'effort de recherche sur les branches prometteuses.

## Structure

```
Projet_MCTS/
├── core/                    # package commun (une seule source de vérité)
│   ├── environment.py       #   MDP : dynamique de stockage + vendre/stocker/consommer
│   ├── baselines.py         #   heuristiques + optimum par programmation dynamique
│   ├── mcts.py              #   MCTS/UCT déterministe (horizon glissant)
│   ├── price_model.py       #   modèle de prix markovien (stochastique)
│   ├── stochastic.py        #   SDP, équivalent-certain (MPC), borne clairvoyante
│   ├── mcts_stochastic.py   #   MCTS/UCT à transitions aléatoires
│   └── data_loader.py       #   données réelles France (prix day-ahead + solaire)
├── experiments/             # scripts d'expériences (voir experiments/README.md)
│   ├── run_demo.py          #   déterministe, données synthétiques (+ mode 3 actions)
│   ├── run_sensitivity.py   #   sensibilité au budget de simulations et à c
│   ├── run_demo_stochastic.py  # évaluation Monte-Carlo, prix incertains
│   └── run_demo_real.py     #   comparaison sur vraies données France
├── tests/                   # tests unitaires (dynamique, PD vs force brute, MCTS)
├── figures/                 # figures générées par les scripts
├── requirements.txt
└── README.md
```

## Installation (Windows, environnement virtuel)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(macOS / Linux : `source venv/bin/activate` à la place de la 2e ligne.)

## Utilisation (depuis la racine du projet)

```bat
python experiments\run_demo.py              :: déterministe (synthétique), multi-graines
python experiments\run_sensitivity.py       :: étude de sensibilité du MCTS
python experiments\run_demo_stochastic.py   :: évaluation Monte-Carlo (prix incertains)
python experiments\run_demo_real.py         :: comparaison sur vraies données France
python -m unittest discover tests           :: tests unitaires
```

## Modèle en bref

État `(t, soc)` ; action = débit de stockage `u` (charge `u>0`, décharge `u<0`)
sur une grille de `n_actions` valeurs. L'énergie non stockée est répartie entre
**vente** (prix spot) et **consommation interne** (demande `demand[t]`,
valorisée au prix d'évitement `p_consume[t]`) — répartition résolue par
dominance, car elle est instantanée et sans effet sur le stock. Un mode
`action_mode="simple3"` implémente les trois actions littérales du sujet
(VENDRE / STOCKER / CONSOMMER) et donc l'arbre 3^H de l'énoncé. Détails :
[core/README.md](core/README.md).

## Données

- **Prix day-ahead France** — [energy-charts.info](https://energy-charts.info) (Fraunhofer ISE),
  rediffusés en **CC BY 4.0** (source : Bundesnetzagentur | SMARD.de).
- **Production solaire France** — [ODRÉ / éCO2mix](https://odre.opendatasoft.com) (RTE).
- Alternative « officielle » : ENTSO-E Transparency (client `entsoe-py`, token gratuit).

Les séries sont récupérées à la volée (rien n'est versionné). **Licence à respecter :**
les prix EPEX SPOT restent propriété d'EPEX SPOT SE — usage non commercial, avec mention
de la source. Citer energy-charts.info / SMARD.de et RTE/ODRÉ dans le mémoire.

## Résultats en bref

En part de la valeur d'arbitrage captée (0 % = sans stockage, 100 % = optimum
par programmation dynamique) :

- **Synthétique (48 h)** : seuil ~85 %, **MCTS à rollout informé 96,6 ± 0,1 %**
  (5 graines) ; sur l'arbre 3^48 littéral du sujet, MCTS ~96 % également.
- **Semaine réelle de juin 2024** (20 h de prix négatifs) : seuil ~25 %,
  **MCTS ~94 %** de la valeur d'arbitrage (98,7 % de l'optimum).
- **Sensibilité** : le rollout est le levier décisif — informé par l'heuristique
  de seuil, le MCTS atteint ~96–99 % dès 30–100 simulations/pas ; en rollout
  aléatoire il monte lentement avec le budget (~35 → 61 %) et plafonne loin de
  l'optimum. La constante d'exploration `c` a peu d'effet une fois le rollout
  informé.
- **Temps de calcul & passage à l'échelle** : la PD est la moins chère sur
  l'instance à 1 batterie (`run_timing.py`), mais son coût explose en `n_soc^B`
  avec le nombre `B` de batteries. Avec `run_scaling.py`, le **MCTS devient plus
  rapide dès 3 batteries** (0,65 s vs 13 s) puis reste le seul praticable quand
  la PD devient infaisable — c'est le vrai argument en faveur du MCTS.
- **Stochastique** : le MCTS (90 % avec 800 simulations/pas, en hausse avec le
  budget) approche l'optimum causal exact (SDP) en n'utilisant qu'un modèle
  génératif ; l'équivalent-certain reste une base très forte, et l'intérêt
  propre du MCTS est sa généralité (il se passe d'un modèle explicite et passe
  à l'échelle sur des modèles de prix riches).

Chiffres détaillés et protocole : [experiments/README.md](experiments/README.md).
