# `core/` — le code partagé

Toute la logique du projet est ici ; les scripts de `experiments/` ne font
qu'appeler ces modules.

## `environment.py` — le MDP

À chaque heure `t`, la centrale produit `production[t]`, voit le prix spot
`prices[t]` et une demande interne `demand[t]`. Son énergie peut être :

- vendue au prix spot ;
- stockée dans la batterie (capacité, rendements charge/décharge, débits max) ;
- consommée pour couvrir la demande interne, valorisée au prix d'évitement
  `p_consume[t]` (le prix qu'on aurait payé pour l'acheter).

L'état est `(t, soc)`, la récompense `vendu × prix + consommé × p_consume`, sur
un horizon fini sans actualisation. La centrale n'achète jamais sur le marché :
le stockage sert seulement à décaler l'usage de sa propre production.

Deux modes d'action (`action_mode`) :

- `"simple3"` : les trois actions de l'énoncé (vendre / stocker / consommer),
  donc l'arbre 3^H.
- `"grid"` (défaut) : l'action est un débit de stockage `u` parmi `n_actions`
  valeurs. Le partage du reste entre vente et consommation ne dépend que du prix
  (on consomme si `p_consume > prix`), donc il est fait directement, ce qui
  laisse le stockage comme seule vraie décision. L'arbre `n_actions^H` est encore
  plus grand.

La dynamique est séparée du prix : `transition()` renvoie le stock suivant et
l'énergie à écouler, `revenue()` applique un prix. Ça permet de brancher le prix
déterministe ou le modèle stochastique sur la même batterie sans dupliquer le
code. `apply()` fait les deux avec les prix réels de l'environnement.

`make_synthetic_data()` génère des profils journaliers (prix creux à midi,
production solaire en cloche, demande le soir).

## `baselines.py` — références

Quelques heuristiques (aléatoire, tout vendre, glouton myope, seuil de prix) et
surtout `dp_optimal`, l'optimum exact par programmation dynamique (induction
arrière sur une grille de soc). C'est le 100 % contre lequel on mesure le MCTS.
Il est vérifié contre une énumération exhaustive dans les tests.

## `mcts.py` — le MCTS

`MCTSPlanner` en horizon glissant : à chaque pas on construit un arbre depuis
l'état courant, on joue l'action la plus visitée, puis on recommence. Les quatre
phases habituelles (sélection UCT, expansion, rollout, rétropropagation). Deux
points à noter :

- les valeurs d'action sont normalisées dans [0,1] avant UCT, pour que la
  constante `c` garde le même sens quelle que soit l'échelle des prix ;
- les nœuds terminaux comptent leurs visites (sans ça, au dernier pas le choix
  « action la plus visitée » devient arbitraire — repéré par les tests).

Le rollout peut être aléatoire ou informé par une politique (le seuil de prix).
C'est ce qui change le plus la performance.

## Cas stochastique

- `price_model.py` : le prix = moyenne saisonnière connue + une déviation
  aléatoire (chaîne de Markov, avec des pics). Donne la matrice de transition
  (pour le SDP) et de quoi échantillonner des scénarios (pour le MCTS).
- `stochastic.py` : borne clairvoyante (DP sur le chemin réalisé), SDP (optimum
  causal exact sur `(t, soc, état_prix)`), équivalent-certain (on re-optimise à
  chaque pas sur la prévision moyenne).
- `mcts_stochastic.py` : même MCTS mais l'état de prix suivant est échantillonné
  après chaque action. N'utilise que `sample_next`, jamais la matrice.

## `data_loader.py` — données réelles

Télécharge les prix day-ahead et le solaire France et les aligne sur une grille
horaire UTC (évite les jours de 23/25 h). Les prix négatifs sont gardés (ils
donnent de la valeur au stockage). `build_real_env()` construit l'environnement
en dimensionnant la batterie par rapport au pic solaire. Test :
`python -m core.data_loader`.
