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

Sur 24 h synthétiques : gain d'arbitrage vs **budget de simulations**
(30 → 3000, échelle log) et vs **constante d'exploration c** (0,25 → 2), pour
les deux rollouts. Enseignements (5 graines) :

- **Le rollout est le levier décisif** : informé par le seuil, le MCTS dépasse
  99 % dès ~100 simulations/pas (94 % à 30) ; en rollout aléatoire il reste
  sous ~30 % même à 3000 simulations — les rollouts aléatoires sous-évaluent
  systématiquement le stockage (une charge n'a de valeur que si la suite du
  scénario la décharge au bon moment, ce qu'une continuation aléatoire fait
  rarement).
- **c a peu d'effet** une fois le rollout informé (99,3–99,5 % sur toute la
  plage) ; en rollout aléatoire, le signal est trop bruité pour que c le
  compense.

Figure : `figures/sensibilite.png` (barres d'erreur = écart-type).

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
