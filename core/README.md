# `core/` — modèle, algorithmes et données

Package commun à toutes les expériences : **une seule implémentation** de
l'environnement, des stratégies de référence et du MCTS (plus de copies par
dossier). Les scripts de `experiments/` importent `core.*`.

## Le MDP (`environment.py`)

Problème de **producteur** : à chaque heure `t`, la centrale produit
`production[t]`, observe le prix spot `prices[t]` et une demande interne
`demand[t]`. L'énergie produite (ou déstockée) peut être :

- **vendue** au prix spot `prices[t]` ;
- **stockée** en batterie (capacité `capacity`, rendements `eta_charge` /
  `eta_discharge`, débits max `max_charge` / `max_discharge`) ;
- **consommée** pour couvrir la demande interne, valorisée au **prix
  d'évitement** `p_consume[t]` (coût de l'électricité qu'on n'a pas achetée,
  p. ex. un tarif de détail).

État `(t, soc)` ; récompense = `vendu × prices[t] + consommé × p_consume[t]` ;
horizon fini `H`, sans actualisation. Pas d'achat sur le marché : le stockage
ne fait que **décaler dans le temps** l'usage de sa propre production.

### Deux espaces d'action (`action_mode`)

- **`"simple3"`** — les trois actions littérales du sujet : `VENDRE`
  (décharge max + tout vendre), `STOCKER` (charge max, surplus vendu),
  `CONSOMMER` (couvrir la demande, surplus vendu). C'est exactement l'arbre
  **3^H** de l'énoncé (3^24 ≈ 2,8 × 10¹¹ pour une journée).
- **`"grid"`** (défaut) — généralisation : l'action est un **débit de
  stockage** `u` parmi `n_actions` valeurs (u > 0 charge, u < 0 décharge,
  u = 0 rien). L'énergie non stockée est répartie entre vente et consommation
  **par dominance** : à énergie dispatchée donnée, la répartition n'affecte pas
  le stock et n'a donc aucun effet futur — l'affectation optimale est immédiate
  (consommer d'abord si `p_consume[t] > prices[t]`, dans la limite de la
  demande, vendre le reste). La seule décision séquentielle non triviale — le
  stockage — reste seule dans l'espace d'action, sans actions dominées, et
  l'arbre `n_actions^H` ≫ 3^H renforce l'argument combinatoire du sujet.

### Interface pour les planificateurs

```
transition(soc, t, a) -> (soc', out, forced)   # dynamique SANS prix
revenue(out, forced, t, price) -> récompense   # valorisation à prix donné
apply(soc, t, a) -> (soc', récompense)         # raccourci prix déterministes
```

La séparation transition/revenue permet aux méthodes stochastiques de brancher
un prix externe (échantillonné dans le modèle de prix) **sans dupliquer la
dynamique** — c'est la même batterie dans tous les cadres.

`make_synthetic_data(days, seed)` fournit des profils journaliers plausibles
(prix en « duck curve », production solaire en cloche, demande avec bosse du
soir), pour garder la maîtrise des paramètres expérimentaux.

## Stratégies de référence (`baselines.py`)

- **Heuristiques** : aléatoire (plancher), sans stockage (`u=0`), glouton myope
  (max de la récompense immédiate — illustre le coût de la myopie), seuil de
  prix (vendre si prix ≥ médiane, sinon stocker — l'arbitrage classique).
- **`dp_optimal`** : optimum exact du MDP discrétisé par **programmation
  dynamique** (induction arrière sur une grille de soc, interpolation
  linéaire). C'est le **plafond théorique** en déterministe ; le MCTS est
  mesuré en pourcentage de cette borne. Vérifiée contre une énumération
  exhaustive dans les tests.

## MCTS déterministe (`mcts.py`)

`MCTSPlanner` : UCT à **horizon glissant** (style MPC) — à chaque pas réel, un
arbre est construit depuis `(t, soc)`, l'action la plus visitée est exécutée,
puis on replanifie. Les quatre phases classiques (sélection UCT, expansion,
rollout, rétropropagation). Deux choix d'implémentation notables :

- **Normalisation locale d'UCT** : les valeurs d'action d'un nœud sont
  ramenées dans [0, 1] (min/max locaux) avant d'ajouter le terme d'exploration,
  pour rester robuste à l'échelle des récompenses (€ vs centaines de k€).
- **Visites des nœuds terminaux comptées** : au dernier pas de l'horizon, les
  enfants de la racine sont terminaux ; sans incrément de leur compteur, le
  choix « action la plus visitée » y serait arbitraire (bug détecté par les
  tests unitaires, corrigé).

Le rollout est soit aléatoire, soit **informé** par une politique passée en
paramètre (typiquement l'heuristique de seuil) — c'est le levier de performance
principal, voir `experiments/run_sensitivity.py`.

## Version stochastique (`price_model.py`, `stochastic.py`, `mcts_stochastic.py`)

- **`MarkovPriceModel`** : prix = moyenne saisonnière connue + déviation à
  retour à la moyenne, discrétisée en chaîne de Markov avec queue haute (pics).
  Fournit la matrice de transition (pour le SDP) **et** un modèle génératif
  (seul requis par le MCTS).
- **`stochastic.py`** : borne **clairvoyante** (DP sur le chemin réalisé),
  **SDP** (optimum causal exact sur l'état augmenté `(t, soc, état_prix)`),
  **équivalent-certain** (MPC : re-optimisation déterministe sur la prévision
  moyenne à chaque pas).
- **`StochasticMCTSPlanner`** : UCT pour MDP à transitions aléatoires — l'état
  de prix suivant est **échantillonné** à chaque passage (pas d'arbre exhaustif
  sur les aléas), les retours sont agrégés par action. N'utilise que
  `sample_next` : extensible à des modèles de prix riches où le SDP explose.

## Données réelles (`data_loader.py`)

Prix day-ahead France (energy-charts.info, CC BY 4.0) + production solaire
France (ODRÉ / éCO2mix, RTE), alignés sur une grille **horaire UTC** (évite les
jours de 23 h / 25 h aux changements d'heure). Les **prix négatifs sont
conservés** : pour un producteur qui doit écouler sa production, ils donnent au
stockage une valeur bien réelle. `build_real_env(...)` dimensionne le stockage
en fraction du pic solaire et peut activer une demande interne
(`demand_frac`, `p_consume`). Test rapide : `python -m core.data_loader`.
