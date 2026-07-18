# Projet MCTS — gestion énergétique d'une centrale avec stockage

Projet M2. On applique l'algorithme Monte Carlo Tree Search (MCTS) à un problème
de décision séquentielle : à chaque heure, une centrale doit vendre, stocker ou
consommer l'électricité qu'elle produit, pour maximiser son profit sur la
journée. Les prix viennent du marché day-ahead français (EPEX SPOT).

L'intérêt du MCTS vient du nombre de séquences de décisions possibles : avec les
3 actions du sujet on a déjà 3^24 ≈ 2,8 × 10¹¹ séquences pour une journée, et
avec la grille de débits qu'on utilise (11 actions) 11^24 ≈ 10²⁵. On ne peut pas
tout énumérer ; le MCTS explore seulement les branches intéressantes.

## Structure

```
Projet_MCTS/
├── core/                    # code partagé
│   ├── environment.py       #   le MDP (stockage + vendre/stocker/consommer)
│   ├── baselines.py         #   heuristiques + optimum (programmation dynamique)
│   ├── mcts.py              #   MCTS/UCT déterministe
│   ├── price_model.py       #   modèle de prix markovien (partie stochastique)
│   ├── stochastic.py        #   SDP, équivalent-certain, borne clairvoyante
│   ├── mcts_stochastic.py   #   MCTS pour le cas stochastique
│   └── data_loader.py       #   téléchargement des données réelles France
├── experiments/             # scripts qui produisent les résultats et figures
├── tests/                   # tests unitaires
├── figures/                 # figures générées
├── report/                  # rapport LaTeX
├── requirements.txt
└── README.md
```

## Installation (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(macOS / Linux : `source venv/bin/activate` pour la 2e ligne.)

## Lancer (depuis la racine)

```bat
python experiments\run_demo.py              :: données synthétiques
python experiments\run_sensitivity.py       :: sensibilité budget / c
python experiments\run_timing.py            :: temps de calcul
python experiments\run_scaling.py           :: PD vs MCTS quand l'état grandit
python experiments\run_demo_stochastic.py   :: prix incertains
python experiments\run_demo_real.py         :: vraies données France
python -m unittest discover tests           :: tests
```

Voir [experiments/README.md](experiments/README.md) pour le détail de chaque
script et les résultats, et [core/README.md](core/README.md) pour le modèle.

## Le modèle en deux mots

État `(t, soc)` où `soc` est le niveau de la batterie. L'action est un débit de
stockage `u`. L'énergie qui n'est pas stockée est vendue au prix spot, ou
consommée en interne si le prix d'évitement est plus intéressant. Il existe
aussi un mode à 3 actions (vendre / stocker / consommer) qui correspond
exactement à l'énoncé.

## Données

- Prix day-ahead France : [energy-charts.info](https://energy-charts.info)
  (Fraunhofer ISE), données EPEX SPOT rediffusées en CC BY 4.0.
- Production solaire France : [ODRÉ / éCO2mix](https://odre.opendatasoft.com) (RTE).

Les séries sont téléchargées à la volée, rien n'est stocké dans le dépôt. Les
prix EPEX restent propriété d'EPEX SPOT SE (usage académique, avec mention de la
source).

## Principaux résultats

En pourcentage de la valeur d'arbitrage captée (0 % = sans stockage, 100 % =
optimum) :

- Données synthétiques (48 h) : heuristique de seuil ~85 %, MCTS avec rollout
  informé ~96 %.
- Semaine réelle de juin 2024 (avec des prix négatifs) : seuil ~25 %, MCTS ~94 %.
- Le choix du rollout est ce qui compte le plus : un rollout informé rend le
  MCTS bon dès ~100 simulations, un rollout aléatoire plafonne bien plus bas.
- Sur cette petite instance la PD est plus rapide que le MCTS, mais son coût
  explose avec le nombre de batteries : dès 3 batteries le MCTS passe devant,
  puis reste le seul praticable (`run_scaling.py`).
- En stochastique, le MCTS approche l'optimum causal (SDP) sans utiliser la
  matrice de transition, juste en échantillonnant.
