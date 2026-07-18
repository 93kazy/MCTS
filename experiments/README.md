# `experiments/` — les expériences

Tous les scripts se lancent depuis la racine du projet et écrivent leurs figures
dans `figures/`.

## La métrique : le gain d'arbitrage

```
gain = (profit − sans_stockage) / (optimum − sans_stockage)
```

Sans stockage = 0 %, optimum (programmation dynamique) = 100 %. On l'utilise
parce que « tout vendre » fait déjà ~87 % de l'optimum en valeur absolue : cette
métrique isole ce que le stockage apporte vraiment. Le MCTS est moyenné sur
plusieurs graines (moyenne ± écart-type).

## `run_demo.py` — données synthétiques

```bat
python experiments\run_demo.py [n_seeds] [n_simulations]   :: défaut 5 1000
```

48 h synthétiques. Résultats (5 graines, 1000 simulations/pas) :

| Stratégie | % optimum | % arbitrage |
|---|---|---|
| Sans stockage / glouton myope | 87,5 % | 0 % |
| Seuil de prix | 98,1 % | 85,1 % |
| MCTS, rollout aléatoire | 94,9 % | 58,8 % |
| MCTS, rollout informé | 99,6 % | 96,6 % (± 0,1) |
| Optimum (PD) | 100 % | 100 % |

Le script teste aussi le mode `simple3` (les 3 actions du sujet, arbre 3^48) :
le MCTS y capte ~96 % de l'arbitrage. Figure : `comparaison.png`.

## `run_sensitivity.py` — budget et constante c

```bat
python experiments\run_sensitivity.py [n_seeds]            :: défaut 5
```

Même instance 48 h. On fait varier le budget de simulations (30 → 3000) et la
constante `c` (0,25 → 2), pour les deux rollouts. Ce qu'on observe :

- Le rollout compte plus que tout. Informé par le seuil, le MCTS est déjà bon
  partout (~96 % dès 30 simulations, 97–99 % ensuite). Aléatoire, il monte avec
  le budget (~35 % → ~61 %) mais reste loin de l'optimum : les continuations
  aléatoires déchargent rarement l'énergie au bon moment.
- `c` change peu de chose une fois le rollout informé.
- Les courbes ne sont pas parfaitement monotones : avec le rollout informé,
  plus de simulations peut même faire un peu baisser le résultat (98,6 % à 100,
  96,6 % à 1000), l'arbre s'éloignant du bon défaut du rollout. C'est normal,
  MCTS n'est exactement optimal qu'à très grand budget.

Figure : `sensibilite.png`.

## `run_timing.py` — temps de calcul

```bat
python experiments\run_timing.py [n_repeat]               :: défaut 3
```

Compare le coût d'un pas de MCTS à celui de la PD complète (instance 48 h). Sur
cette petite instance, la PD est en fait la moins chère (~0,35 s par épisode),
et le MCTS ne la rejoint qu'autour de 100 simulations/pas. L'intérêt du MCTS
n'est donc pas la vitesse ici — voir `run_scaling.py`.

## `run_scaling.py` — quand le MCTS devient plus rapide

```bat
python experiments\run_scaling.py [dp_max_B] [n_sim_mcts]   :: défaut 3 400
```

On passe à B batteries de rendements différents (état = les B niveaux de charge,
actions = `2B+1`). Le coût de la PD est en `n_soc^B` et explose ; le MCTS ne
visite que les états atteints, donc son coût ne bouge presque pas. Résultats :

| B | états PD | temps PD | temps MCTS | profit PD | profit MCTS |
|---|---|---|---|---|---|
| 1 | 21 | 0,01 s | 0,59 s | 51 667 | 51 530 |
| 2 | 441 | 0,37 s | 0,61 s | 53 318 | 52 942 |
| 3 | 9 261 | 13,1 s | 0,65 s | 53 657 | 52 469 |
| 4 | 194 481 | infaisable | 0,68 s | — | 52 297 |
| 5 | 4 084 101 | infaisable | 0,75 s | — | 52 553 |
| 6 | 85 766 121 | infaisable | 0,71 s | — | 52 288 |

À 1 batterie la PD est ~50× plus rapide ; à 3 batteries le MCTS passe devant
(0,65 s vs 13 s) ; au-delà la PD n'est plus praticable alors que le MCTS tourne
toujours, à ~2 % de l'optimum. C'est la malédiction de la dimension : la PD paie
pour tout l'espace d'état, le MCTS seulement pour ce qu'il explore. Figure :
`scaling.png`.

## `run_demo_stochastic.py` — prix incertains

```bat
python experiments\run_demo_stochastic.py [n_eval] [n_sim_mcts]   :: défaut 40 800
```

Modèle de prix markovien (retour à la moyenne + pics). Évaluation Monte-Carlo
appariée : 40 chemins communs à toutes les politiques, jouées sans voir le futur
(sauf le clairvoyant, qui donne la borne haute). Résultats (40 scénarios) :

| Stratégie | % arbitrage (SDP = 100) |
|---|---|
| Tout vendre | 0 % |
| Seuil | ~83 % |
| Équivalent-certain (MPC) | ~100 % |
| MCTS, rollout aléatoire | ~20 % |
| MCTS, rollout informé | ~90 % (croît avec le budget) |
| SDP (optimum causal) | 100 % |

À retenir pour la soutenance :

1. le MCTS approche l'optimum causal (SDP) en n'utilisant qu'un modèle génératif,
   il ne le bat pas (le SDP est optimal par construction) ;
2. l'équivalent-certain égale ici le SDP, parce que la re-planification corrige
   l'incertitude au fil de l'eau (le coût de l'incertitude est ~1 %) ;
3. l'intérêt du MCTS reste sa généralité : il ne demande qu'un simulateur, là où
   le SDP exige un état de prix discret explicite et l'équivalent-certain une
   prévision optimisable.

Figure : `comparaison_stochastique.png`.

## `run_demo_real.py` — vraies données France

```bat
python experiments\run_demo_real.py [debut] [fin] [n_seeds]
python experiments\run_demo_real.py 2024-06-10 2024-06-16 3       :: défaut
```

Semaine de juin 2024 (166 h, dont 20 h de prix négatifs liés au solaire de midi).
Stockage = 60 % du pic solaire, demande interne = 15 % du pic (prix d'évitement
90 €/MWh), ce qui active la consommation. Résultats (300 simulations/pas,
3 graines) :

| Stratégie | % optimum | % arbitrage |
|---|---|---|
| Sans stockage | 77,5 % | 0 % |
| Glouton myope | 79,1 % | 7,2 % |
| Seuil (médiane) | 83,2 % | 25,3 % |
| MCTS rollout informé | 98,7 % | 94,2 % |
| Optimum (PD) | 100 % | 100 % |

Les prix négatifs rendent le stockage bien plus utile que sur les données
synthétiques, et l'heuristique de seuil décroche (elle ne gère pas les prix
négatifs) alors que le MCTS tient. Figure : `comparaison_reelle.png`.
