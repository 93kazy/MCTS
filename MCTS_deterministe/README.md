# Phases 4 & 5 — Stratégies de référence et MCTS

Cœur algorithmique du projet en version **déterministe** (prévision parfaite),
qui sert d'étape de validation : la programmation dynamique fournit l'optimum
exact, contre lequel le MCTS et les heuristiques sont mesurés. Les données sont
**synthétiques** en attendant les séries EPEX SPOT France.

## Modèle (rappel)

Producteur avec stockage. À chaque heure `t` : production `prod[t]`, prix spot
`price[t]`, niveau de stockage `soc`. Une action = un débit de batterie `u`
(charge `u>0`, décharge `u<0`, tout vendre `u=0`). Seule la production est
dispatchée (pas d'achat) : le stockage décale dans le temps la vente de sa propre
production. Récompense = énergie vendue × prix. Horizon fini, sans actualisation.

> *Hook « consommer »* : se modélise en valorisant une part de l'énergie à un
> prix d'évitement ; mécanisme identique à la vente, laissé en extension pour
> garder l'action en une dimension.

## Fichiers

- `environment.py` — simulateur MDP (`EnergyStorageEnv`) + données synthétiques.
- `baselines.py` — **phase 4** : heuristiques (tout vendre, seuil, glouton,
  aléatoire) et optimum par programmation dynamique (`dp_optimal`).
- `mcts.py` — **phase 5** : `MCTSPlanner` (UCT, horizon glissant, rollout
  aléatoire ou informé).
- `run_demo.py` — comparaison chiffrée + figure `comparaison.png`.

## Lancer

```bash
python run_demo.py
```

## Lecture des résultats

Métrique clé : le **gain d'arbitrage** = part de la valeur du stockage captée
(`(profit − tout_vendre) / (optimum − tout_vendre)`), qui isole l'apport réel du
contrôleur. Résultats typiques sur les données synthétiques :

| Stratégie | % arbitrage |
|---|---|
| Tout vendre / glouton myope | ~0 % |
| Seuil de prix | ~87 % |
| MCTS, rollout aléatoire | ~51 % |
| MCTS, rollout informé (seuil) | ~94 % |
| Optimum (PD) | 100 % |

Enseignements : la myopie ignore toute la valeur du stockage ; une bonne
heuristique en capte l'essentiel ; le **rollout est le levier décisif** du MCTS,
qui une fois informé dépasse l'heuristique et approche l'optimum.

## Prochaines étapes / extensions

- **Version stochastique** (contribution principale) : remplacer le rejeu
  identique des prix dans `_rollout` par un échantillonnage de scénarios. La
  structure horizon-glissant du `MCTSPlanner` ne change pas.
- **Vraies données** : injecter les séries EPEX SPOT France dans
  `EnergyStorageEnv` à la place de `make_synthetic_data` (penser aux prix
  négatifs et aux jours de 23 h/25 h au changement d'heure).
- **Canal de consommation**, réutilisation d'arbre entre pas, étude de
  sensibilité (`c`, budget de simulations, finesse de discrétisation).
