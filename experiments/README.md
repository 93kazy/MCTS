# `experiments/` — protocole et résultats

Tous les scripts se lancent **depuis la racine** du projet et écrivent leurs
figures dans `figures/`.

## Métrique commune : le gain d'arbitrage

```
gain = (profit − profit_sans_stockage) / (optimum − profit_sans_stockage)
```

« Sans stockage » (u = 0 à chaque pas, répartition vente/conso au fil de l'eau)
vaut 0 % ; l'optimum (programmation dynamique, prévision parfaite) vaut 100 %.
Cette métrique isole ce que la stratégie apporte **via le stockage**, au lieu
de la noyer dans le revenu de base (sans elle, « tout vendre » afficherait déjà
~87 % de l'optimum et tout semblerait bon).

Le MCTS étant stochastique, il est évalué sur **plusieurs graines**
(moyenne ± écart-type).

## `run_demo.py` — déterministe, données synthétiques

```bat
python experiments\run_demo.py [n_seeds] [n_simulations]   :: défaut : 5 1000
```

48 h synthétiques (duck curve + solaire + demande interne, prix d'évitement
70 €/MWh). Résultats (5 graines, 1000 simulations/pas) :

| Stratégie | % optimum | % arbitrage |
|---|---|---|
| Sans stockage / glouton myope | 87,5 % | 0 % |
| Seuil de prix (médiane) | 98,1 % | 85,1 % |
| MCTS, rollout aléatoire | 94,9 % | 58,8 % |
| **MCTS, rollout informé (seuil)** | **99,6 %** | **96,6 %** (± 0,1) |
| Optimum (PD) | 100 % | 100 % |

Le script joue aussi le mode **`simple3`** — les 3 actions littérales du sujet,
soit l'arbre 3^48 : le MCTS y capte ~96 % de l'arbitrage de cet espace
(l'optimum simple3 est lui-même en dessous de l'optimum grille : les débits
fins dominent le tout-ou-rien). Figure : `figures/comparaison.png`.

## `run_sensitivity.py` — sensibilité du MCTS

```bat
python experiments\run_sensitivity.py [n_seeds]            :: défaut : 5
```

Sur la **même instance 48 h** que `run_demo.py` (pour que les chiffres soient
comparables) : gain d'arbitrage vs **budget de simulations** (30 → 3000,
échelle log) et vs **constante d'exploration c** (0,25 → 2), pour les deux
rollouts. Enseignements (5 graines) :

- **Le rollout est le levier décisif** : informé par le seuil, le MCTS est
  déjà excellent partout (~96 % dès 30 simulations/pas, 97–99 % au-delà), quasi
  plat juste au-dessus de l'heuristique de départ ; en rollout aléatoire il
  monte avec le budget (~35 % à 30 → ~61 % à 3000) mais plafonne loin de
  l'optimum — les continuations aléatoires déchargent rarement l'énergie au bon
  moment, ce qui biaise les estimations vers le bas.
- **c a peu d'effet** une fois le rollout informé (~97 % sur toute la plage) ;
  en rollout aléatoire, le signal est trop bruité pour que c le compense.
- **Non-monotonie assumée** : avec le rollout informé, plus de simulations peut
  même faire légèrement *baisser* le résultat (98,6 % à 100, 96,6 % à 1000),
  car l'arbre s'éloigne du bon défaut du rollout vers ses propres estimations
  plus bruitées. MCTS ne converge vers l'optimum qu'à très grand budget ; ici
  il est déjà proche d'un plafond.

Figure : `figures/sensibilite.png` (barres d'erreur = écart-type).

## `run_timing.py` — temps de calcul

```bat
python experiments\run_timing.py [n_repeat]               :: défaut : 3
```

Compare le coût d'un pas de planification MCTS (budgets 100 / 300 / 1000) à
celui de la PD complète, sur l'instance 48 h. Résultat honnête : sur cette
petite instance déterministe, la **PD est en fait la moins chère** (~0,35 s par
épisode) ; le MCTS ne la rejoint qu'à ~100 simulations/pas (~0,22 s/épisode) et
devient plus lent au-delà. L'intérêt du MCTS n'est donc pas la vitesse ici,
mais sa généralité (pas besoin d'état discrétisé explicite ni de prévision
parfaite) — voir `run_scaling.py` et la version stochastique.

## `run_scaling.py` — là où MCTS devient plus rapide que la PD

```bat
python experiments\run_scaling.py [dp_max_B] [n_sim_mcts]   :: défaut : 3 400
```

Généralise à **B batteries de rendements différents** (état = vecteur des B
niveaux de charge, espace d'action gardé petit : `2B+1`). Le coût de la PD est
`O(H · n_soc^B · n_actions)` : le terme `n_soc^B` explose. Le MCTS ne visite que
les états atteints → coût ~indépendant de B. Résultats mesurés :

| B | états PD | temps PD | temps MCTS | profit PD | profit MCTS |
|---|---|---|---|---|---|
| 1 | 21 | 0,01 s | 0,59 s | 51 667 | 51 530 |
| 2 | 441 | 0,37 s | 0,61 s | 53 318 | 52 942 |
| 3 | 9 261 | **13,1 s** | 0,65 s | 53 657 | 52 469 |
| 4 | 194 481 | *infaisable* | 0,68 s | — | 52 297 |
| 5 | 4 084 101 | *infaisable* | 0,75 s | — | 52 553 |
| 6 | 85 766 121 | *infaisable* | 0,71 s | — | 52 288 |

À 1 batterie, la PD est ~50× plus rapide ; à **3 batteries, le MCTS passe
devant** (0,65 s vs 13 s) ; au-delà, la PD est infaisable alors que le MCTS
tourne toujours au même coût, à ~2 % de l'optimum PD là où on peut encore le
calculer. C'est la « malédiction de la dimension » en image (figure
`figures/scaling.png`) : la PD paie pour tout l'espace d'état, le MCTS ne paie
que ce qu'il explore.

## `run_demo_stochastic.py` — prix incertains

```bat
python experiments\run_demo_stochastic.py [n_eval] [n_sim_mcts]   :: défaut : 40 800
```

Modèle de prix markovien (retour à la moyenne + pics). Évaluation Monte-Carlo
**appariée** : 40 chemins de prix communs à toutes les politiques, jouées en
boucle fermée (causales) ; le clairvoyant voit le futur (borne haute).
Résultats (40 scénarios, 800 simulations/pas) :

| Stratégie | % arbitrage (SDP = 100) |
|---|---|
| Tout vendre | 0 % |
| Seuil | ~83 % |
| Équivalent-certain (MPC) | **~100 %** |
| MCTS, rollout aléatoire | ~20 % |
| MCTS, rollout informé | ~90 % (croît avec le budget) |
| SDP (optimum causal) | 100 % |

Coût de l'incertitude (clairvoyant − SDP) : ~1 %.

**Lecture honnête (pour la soutenance)** :
1. le MCTS **approche** l'optimum causal exact (SDP) en n'utilisant qu'un
   modèle génératif — il ne le bat pas (le SDP est optimal par construction) ;
2. l'équivalent-certain égale ici le SDP : la re-planification corrige
   l'incertitude au fil de l'eau et le coût de l'incertitude est faible —
   résultat connu, à assumer ;
3. l'intérêt propre du MCTS est sa **généralité** : il ne requiert qu'un
   simulateur, là où le SDP exige un état de prix discret explicite (malédiction
   de la dimension) et où l'équivalent-certain exige une prévision optimisable.
   L'expérience à monter pour une contribution : un modèle génératif riche où
   le SDP n'est plus calculable.

Figure : `figures/comparaison_stochastique.png`.

## `run_demo_real.py` — vraies données France

```bat
python experiments\run_demo_real.py [debut] [fin] [n_seeds]
python experiments\run_demo_real.py 2024-06-10 2024-06-16 3       :: défaut
```

Semaine réelle de juin 2024 (166 h, **20 h de prix négatifs** liés à l'excédent
solaire de midi). Stockage = 60 % du pic solaire, demande interne = 15 % du pic
(prix d'évitement 90 €/MWh) — le canal « consommer » est actif : aux heures à
prix négatif ou bas, consommer sa production vaut mieux que la vendre.
Résultats (MCTS : 300 simulations/pas, 3 graines) :

| Stratégie | % optimum | % arbitrage |
|---|---|---|
| Sans stockage | 77,5 % | 0 % |
| Glouton myope | 79,1 % | 7,2 % |
| Seuil (médiane) | 83,2 % | 25,3 % |
| **MCTS rollout informé** | **98,7 %** | **94,2 %** |
| Optimum (PD) | 100 % | 100 % |

Les prix négatifs rendent le stockage bien plus précieux que sur les données
synthétiques : l'écart naïf → optimal y est beaucoup plus marqué, et les
heuristiques simples décrochent. Figure : `figures/comparaison_reelle.png`.

## Reproduire tout

```bat
python -m unittest discover tests
python experiments\run_demo.py
python experiments\run_sensitivity.py
python experiments\run_demo_stochastic.py
python experiments\run_demo_real.py
```
