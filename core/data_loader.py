"""Chargement de VRAIES donnees France pour le projet.

Deux sources ouvertes, sans inscription :
  - PRIX day-ahead France (€/MWh, horaire) via energy-charts.info (Fraunhofer ISE).
    Donnees sous-jacentes EPEX SPOT / ENTSO-E, rediffusees en CC BY 4.0
    (source : Bundesnetzagentur | SMARD.de) -> citer la source.
  - PRODUCTION solaire France (MW, pas 1/2 h) via ODRE / eCO2mix (RTE).

Les deux series sont alignees sur une grille HORAIRE en UTC. Travailler en UTC
evite le probleme des jours de 23 h / 25 h aux changements d'heure (la grille
reste un fil horaire continu, sans trou ni doublon de calendrier local).

Les prix NEGATIFS sont conserves tels quels : pour un producteur qui doit
ecouler sa production (pas d'achat), vendre pendant une heure a prix negatif
fait perdre de l'argent -> le stockage qui decale la vente hors de ces heures
prend une vraie valeur (cas realiste et interessant pour le projet).

API (aucune cle requise) :
  load_real_series(start, end) -> (timestamps, prices, production)
  build_real_env(start, end, ...) -> EnergyStorageEnv

Test rapide du chargeur :  python -m core.data_loader  (depuis la racine)
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np

try:
    from .environment import EnergyStorageEnv
except ImportError:  # execution directe : python core/data_loader.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.environment import EnergyStorageEnv

_HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def _get_json(url):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


# ---------------------------------------------------------------------- #
# Prix day-ahead France (energy-charts.info)                             #
# ---------------------------------------------------------------------- #
def fetch_prices(start, end, bzn="FR"):
    """Renvoie {heure_unix_UTC : prix_eur_mwh}. `end` est inclus (jour entier)."""
    url = ("https://api.energy-charts.info/price?bzn=%s&start=%s&end=%s"
           % (bzn, start, end))
    d = _get_json(url)
    ts = d.get("unix_seconds", [])
    px = d.get("price", [])
    out = {}
    for t, p in zip(ts, px):
        if p is None:
            continue
        out[(t // 3600) * 3600] = float(p)     # ancrage a l'heure pleine
    return out


# ---------------------------------------------------------------------- #
# Production solaire France (ODRE / eCO2mix)                             #
# ---------------------------------------------------------------------- #
def fetch_solar(start, end):
    """Renvoie {heure_unix_UTC : solaire_MW} (moyenne des pas 1/2 h sur l'heure).
    `end` exclu (borne haute de la fenetre)."""
    where = ('date_heure>="%s" AND date_heure<"%s" AND solaire IS NOT NULL'
             % (start, end))
    url = ("https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
           "eco2mix-national-cons-def/exports/json?select=date_heure,solaire&where="
           + urllib.parse.quote(where))
    rows = _get_json(url)
    acc, cnt = {}, {}
    for r in rows:
        dt = datetime.fromisoformat(r["date_heure"]).astimezone(timezone.utc)
        hour = int(dt.timestamp() // 3600) * 3600
        acc[hour] = acc.get(hour, 0.0) + float(r["solaire"])
        cnt[hour] = cnt.get(hour, 0) + 1
    return {h: acc[h] / cnt[h] for h in acc}


# ---------------------------------------------------------------------- #
# Alignement horaire                                                     #
# ---------------------------------------------------------------------- #
def load_real_series(start, end):
    """Aligne prix et production solaire sur les heures communes (UTC).
    Renvoie (timestamps_unix, prices, production) tries chronologiquement."""
    prices = fetch_prices(start, end)
    # borne haute exclusive pour ODRE = lendemain de `end`
    end_excl = (datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
                .timestamp())
    end_excl_str = datetime.fromtimestamp(end_excl + 86400, timezone.utc).strftime("%Y-%m-%d")
    solar = fetch_solar(start, end_excl_str)

    common = sorted(set(prices) & set(solar))
    if not common:
        raise RuntimeError("Aucune heure commune entre prix et production.")
    ts = np.array(common)
    p = np.array([prices[h] for h in common], dtype=float)
    prod = np.array([solar[h] for h in common], dtype=float)
    return ts, p, prod


def build_real_env(start, end, capacity_frac=0.5, power_frac=0.25,
                   eta_charge=0.95, eta_discharge=0.95, n_actions=11,
                   soc0=0.0, demand_frac=0.0, p_consume=90.0):
    """Construit un EnergyStorageEnv a partir de vraies donnees France.

    Le stockage est dimensionne par rapport au pic de production solaire :
      capacity   = capacity_frac * pic_solaire
      max_charge = max_discharge = power_frac * pic_solaire

    Canal consommation (optionnel) : demande interne constante
      demand = demand_frac * pic_solaire, valorisee au prix d'evitement
      p_consume (€/MWh, p.ex. un tarif de detail). demand_frac=0 => pur
      arbitrage vendre/stocker.
    """
    ts, prices, production = load_real_series(start, end)
    peak = float(np.max(production)) if np.max(production) > 0 else 1.0
    demand = np.full(len(prices), demand_frac * peak)
    env = EnergyStorageEnv(prices, production,
                           demand=demand, p_consume=p_consume,
                           capacity=capacity_frac * peak,
                           eta_charge=eta_charge, eta_discharge=eta_discharge,
                           max_charge=power_frac * peak,
                           max_discharge=power_frac * peak,
                           soc0=soc0, n_actions=n_actions)
    return env, ts, prices, production


if __name__ == "__main__":
    # Petit test : une semaine de juin 2024.
    ts, p, prod = load_real_series("2024-06-10", "2024-06-16")
    print("Heures alignees :", len(ts))
    print("Prix  €/MWh  : min %.1f  moy %.1f  max %.1f  (negatifs : %d h)"
          % (p.min(), p.mean(), p.max(), int((p < 0).sum())))
    print("Solaire MW   : min %.0f  moy %.0f  max %.0f"
          % (prod.min(), prod.mean(), prod.max()))
