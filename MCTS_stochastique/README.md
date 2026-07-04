# Projet MCTS — gestion énergétique d'une centrale avec stockage

Décision séquentielle : à chaque heure, vendre / stocker la production pour
maximiser le profit sur l'horizon, prix issus (à terme) d'EPEX SPOT France.
Deux versions :

- **Déterministe** (prévision parfaite) — phases 4 & 5, étape de validation.
- **Stochastique** (prix incertains) — la contribution scientifique.

Données **synthétiques** pour l'instant (à remplacer par EPEX SPOT France).

## Modèle

Producteur avec stockage. État `(t, soc)`, action = débit de batterie `u`
(charge `u>0`, décharge `u<0`, tout vendre `u=0`). Seule la production est
dispatchée (pas d'achat) : le stockage décale dans le temps la vente de sa propre
production. Récompense = énergie vendue × prix. Horizon fini, sans actualisation.
La dynamique du stockage est indépendante du prix (`dispatch`), ce qui permet de
brancher le prix déterministe ou le modèle stochastique sur la même dynamique.

## Fichiers

Déterministe :
- `environment.py` — simulateur (`EnergyStorageEnv`), `dispatch` (sans prix) + données synthétiques.
- `baselines.py` — **phase 4** : heuristiques + optimum par programmation dynamique.
- `mcts.py` — **phase 5** : MCTS/UCT (horizon glissant, rollout aléatoire ou informé).
- `run_demo.py` — comparaison + figure `comparaison.png`.

Stochastique :
- `price_model.py` — `MarkovPriceModel` : moyenne saisonnière + déviation à retour
  à la moyenne (chaîne de Markov, pics de prix). Fournit la matrice de transition
  ET un modèle génératif (échantillonnage).
- `stochastic.py` — borne **clairvoyante**, **équivalent-certain** (MPC sur
  prévision moyenne) et **SDP** (optimum causal exact, DP sur `(t, soc, état_prix)`).
- `mcts_stochastic.py` — `StochasticMCTSPlanner` : MCTS à transitions aléatoires
  (états de prix échantillonnés), n'utilise que le modèle génératif.
- `run_demo_stochastic.py` — évaluation Monte Carlo appariée + `comparaison_stochastique.png`.

## Lancer

```bash
python run_demo.py                       # version déterministe
python run_demo_stochastic.py 40 800     # stochastique : n_scenarios, n_sim MCTS
```

## Résultats — déterministe

Métrique : **gain d'arbitrage** = part de la valeur du stockage captée
(`(profit − tout_vendre)/(optimum − tout_vendre)`).

| Stratégie | % arbitrage |
|---|---|
| Tout vendre / glouton myope | ~0 % |
| Seuil de prix | ~87 % |
| MCTS, rollout aléatoire | ~51 % |
| MCTS, rollout informé | ~94 % |
| Optimum (PD) | 100 % |

Le rollout est le levier décisif : informé, le MCTS dépasse l'heuristique et
approche l'optimum.

## Résultats — stochastique (40 scénarios)

| Stratégie | % arbitrage (SDP=100) |
|---|---|
| Tout vendre | 0 % |
| Seuil | ~83 % |
| Équivalent-certain (MPC) | **~100 %** |
| MCTS, rollout aléatoire | ~20 % |
| MCTS, rollout informé | ~90 % (→94 % à 2000 sim) |
| SDP (optimum causal) | 100 % |

Coût de l'incertitude (clairvoyant − SDP) : **~1–2 %**.

### Lecture honnête (importante pour la soutenance)

1. **Le MCTS rejoint l'optimum causal (SDP)** en n'utilisant qu'un modèle
   génératif — sans matrice de transition explicite. C'est le résultat de
   validation propre. Le MCTS ne *bat* pas le SDP (qui est exact et optimal) :
   il l'**approche** (94 % à 2000 simulations, en hausse avec le budget).

2. **L'équivalent-certain (MPC) est une base très forte ici** : il égale le SDP.
   La re-planification à chaque pas corrige l'incertitude au fur et à mesure, et
   le coût de l'incertitude est faible (~1–2 %). C'est un résultat connu et
   robuste — à assumer plutôt qu'à cacher.

3. **Où le MCTS est réellement justifié** : sa *généralité*. Il ne requiert qu'un
   simulateur, là où le SDP exige un état de prix discrétisé et explicite (malédiction
   de la dimension) et où l'équivalent-certain exige un modèle de prévision
   optimisable. Dès que le modèle de prix s'enrichit (mémoire longue, plusieurs
   variables corrélées, modèle génératif appris, marchés multiples), le SDP devient
   intraitable tandis que le coût du MCTS croît doucement. **C'est l'expérience à
   monter pour la contribution : un modèle génératif où le SDP n'est plus calculable.**

## Données réelles (France)

Deux sources ouvertes, **sans inscription**, déjà câblées dans `data_loader.py` :

- **Prix day-ahead France** (€/MWh, horaire) — `energy-charts.info` (Fraunhofer ISE),
  endpoint `api.energy-charts.info/price?bzn=FR`. Données EPEX SPOT / ENTSO-E
  rediffusées en **CC BY 4.0** (source : Bundesnetzagentur | SMARD.de) → à citer.
- **Production solaire France** (MW, ½ h) — **ODRÉ / éCO2mix** (RTE), dataset
  `eco2mix-national-cons-def` sur `odre.opendatasoft.com`, 2012→2025. Open data RTE.

Alternatives recommandées pour le mémoire (provenance « officielle ») :
- **ENTSO-E Transparency** via le client Python `entsoe-py` : `query_day_ahead_prices('FR', ...)`
  pour les prix et `query_generation('FR', ...)` (filière solaire) pour la production.
  Clé gratuite : s'inscrire sur transparency.entsoe.eu puis e-mail à
  transparency@entsoe.eu (objet « Restful API access »).
- **Open Power System Data** (CSV groupé prix + production + charge, plusieurs pays) :
  pratique mais figé (~2020).

Attention licence : via RTE/éCO2mix, les **prix EPEX** restent propriété d'EPEX SPOT SE
(usage non commercial, avec mention de la source) — OK pour un usage académique.

Édges gérés dans le loader : grille **horaire en UTC** (évite les jours de 23 h/25 h
au changement d'heure), **prix négatifs conservés** (ils donnent une vraie valeur au
stockage pour un producteur qui doit écouler sa production).

```bash
python data_loader.py                       # test : une semaine 2024
python run_demo_real.py 2024-06-10 2024-06-16   # comparaison sur vraies données
```

### Résultats — données réelles (semaine de juin 2024, 20 h de prix négatifs)

| Stratégie | % optimum | % arbitrage |
|---|---|---|
| Tout vendre | 53 % | 0 % |
| Seuil | 74 % | 44 % |
| MCTS rollout informé | **98 %** | **96 %** |
| Optimum (PD) | 100 % | 100 % |

Les prix négatifs (excédent solaire de midi) rendent le stockage bien plus utile que
sur les données synthétiques : l'écart naïf → optimal y est beaucoup plus marqué.

## Prochaines étapes

- **Brancher les vraies données sur la version stochastique** : caler le
  `MarkovPriceModel` sur l'historique réel (saisonnalité + écarts), ou passer à un
  **modèle génératif riche / appris** rendant le SDP intraitable — c'est là que le
  MCTS devient indispensable.
- **MCTS à fenêtre glissante bornée** (lookahead de 24–48 h) pour les longues séries
  réelles, au lieu d'un horizon = mois entier.
- Réduction de variance du MCTS (réutilisation d'arbre, double progressive widening),
  étude de sensibilité (`c`, budget, discrétisation), canal de consommation.
