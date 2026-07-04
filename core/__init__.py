"""Package coeur du projet MCTS — gestion energetique d'une centrale avec stockage.

Modules :
  environment      : MDP (dynamique de stockage + canaux vendre/stocker/consommer)
  baselines        : heuristiques de reference + optimum par programmation dynamique
  mcts             : MCTS/UCT deterministe (horizon glissant)
  price_model      : modele de prix markovien (version stochastique)
  stochastic       : SDP, equivalent-certain (MPC), borne clairvoyante
  mcts_stochastic  : MCTS/UCT pour transitions aleatoires
  data_loader      : donnees reelles France (prix day-ahead + solaire)
"""

from .environment import EnergyStorageEnv, make_synthetic_data, A_SELL, A_STORE, A_CONSUME
