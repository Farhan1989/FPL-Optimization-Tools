"""
Write a sandbox with sample plans so you can look at the UI before pointing it
at the real thing.

    python demo_data.py
    SOLVER_ROOT=./demo uvicorn app:app --port 8711

Delete ./demo when you're done. Nothing here touches your real data.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "demo"
random.seed(11)

SQUAD = [
    ("Raya", "ARS", "GKP", 5.6),
    ("Sels", "NFO", "GKP", 5.1),
    ("Gabriel", "ARS", "DEF", 6.3),
    ("Gvardiol", "MCI", "DEF", 6.1),
    ("Van Dijk", "LIV", "DEF", 6.2),
    ("Muñoz", "CRY", "DEF", 5.5),
    ("Milenković", "NFO", "DEF", 5.4),
    ("Salah", "LIV", "MID", 14.6),
    ("Palmer", "CHE", "MID", 10.4),
    ("Saka", "ARS", "MID", 10.1),
    ("Semenyo", "BOU", "MID", 7.3),
    ("Rogers", "AVL", "MID", 6.0),
    ("Haaland", "MCI", "FWD", 14.9),
    ("Wood", "NFO", "FWD", 7.4),
    ("Beto", "EVE", "FWD", 5.3),
]

MOVES = {
    5: (("Beto", "EVE", "FWD", 5.3), ("Isak", "NEW", "FWD", 10.6)),
    7: (("Rogers", "AVL", "MID", 6.0), ("Mbeumo", "MUN", "MID", 8.2)),
    9: (("Milenković", "NFO", "DEF", 5.4), ("Timber", "ARS", "DEF", 5.9)),
}


# One shared draw, reused by every variant. Real candidate plans sit within a
# fraction of a point of each other over a horizon; independent random draws per
# variant would fake a spread that doesn't exist.
BASE = {gw: [round(max(0.8, random.gauss(4.4 if i < 11 else 2.4, 1.3)), 2) for i in range(15)] for gw in range(3, 11)}


def build(variant, label, drift, chip_gw=None, chip="Bench Boost", skip=()):
    """drift = total xPts this variant gives up across the whole horizon."""
    squad = [dict(name=n, team=t, pos=p, price=v) for n, t, p, v in SQUAD]
    gameweeks = []

    for gw in range(3, 11):
        transfers = []
        if gw in MOVES and gw not in skip:
            out_t, in_t = MOVES[gw]
            transfers = [
                {
                    "out": dict(zip(("name", "team", "pos", "price"), out_t)),
                    "in": dict(zip(("name", "team", "pos", "price"), in_t)),
                }
            ]
            for i, pl in enumerate(squad):
                if pl["name"] == out_t[0]:
                    squad[i] = dict(zip(("name", "team", "pos", "price"), in_t))

        rows, total = [], 0.0
        for i, pl in enumerate(squad):
            xp = BASE[gw][i]
            starting = i < 11
            captain = pl["name"] in ("Haaland", "Salah") and starting and i < 2
            rows.append(
                {
                    **pl,
                    "xpts": xp,
                    "starting": starting,
                    "captain": captain,
                    "vice": pl["name"] == "Palmer",
                    "bench_order": None if starting else i - 10,
                }
            )
            if starting:
                total += xp * (2 if captain else 1)

        gameweeks.append(
            {
                "gw": gw,
                "chip": chip if gw == chip_gw else None,
                "itb": round(random.uniform(0.1, 1.4), 1),
                "ft": 1 if transfers else 2,
                "hits": 0,
                "xpts": round(total - drift / 8, 2),
                "transfers": transfers,
                "squad": rows,
            }
        )

    return {
        "variant": variant,
        "label": label,
        "gameweeks": gameweeks,
        "totals": {
            "xpts": round(sum(g["xpts"] for g in gameweeks), 2),
            "hits": 0,
            "transfers": sum(len(g["transfers"]) for g in gameweeks),
            "gameweeks": len(gameweeks),
        },
    }


def main() -> None:
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)

    plans = [
        build("ev", "Expected points", 0.00),
        build("cvar_a20", "CVaR alpha=0.2", 0.17),
        build("cvar_a10", "CVaR alpha=0.1", 1.42, skip=(9,)),
        build("stochastic", "Two-stage stochastic", 0.31, chip_gw=8),
    ]
    for plan in plans:
        (results / f"{plan['variant']}.plan.json").write_text(json.dumps(plan, indent=2))

    (ROOT / "user_settings.json").write_text(
        json.dumps(
            {
                "team_id": 1234567,
                "horizon": 8,
                "gap": 0,
                "decay_base": 0.87,
                "blend_review_weight": 0.5,
                "blend_solio_weight": 0.5,
                "no_transfer_last_gws": 0,
                "bench_weights": [0.03, 0.21, 0.06, 0.002],
                "banned": [],
                "locked": [],
                "use_cmt": False,
                "solver": "highs",
                "time_limit_seconds": 900,
            },
            indent=2,
        )
        + "\n"
    )

    (ROOT / "data" / "fplreview_gw3.csv").write_text("id,name,team,pos,3_xmins,3_pts\n1,Salah,LIV,MID,88,6.4\n")
    (ROOT / "data" / "solio_gw3.csv").write_text("id,name,team,pos,3_xmins,3_pts\n1,Salah,LIV,MID,90,6.1\n")

    print(f"Demo sandbox written to {ROOT}")
    print(f"Run:  SOLVER_ROOT={ROOT} uvicorn app:app --port 8711")


if __name__ == "__main__":
    main()
