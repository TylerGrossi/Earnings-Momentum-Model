"""Build the CRSP/Compustat data set (needs a WRDS account and `pip install wrds`).

    python scripts/build_crsp.py --probe            # list tables your account can see
    python scripts/build_crsp.py                    # build data/crsp/ (CIZ format)
    python scripts/build_crsp.py --legacy           # use legacy crsp.msf + msedelist instead
    python scripts/build_crsp.py --user tgrossi     # WRDS username (otherwise prompted)

The first connection asks for your WRDS password (and Duo approval) and offers to save a
.pgpass file so later runs don't ask. Run it once yourself in a terminal for that step.
Then run the research on CRSP data with FACTORLAB_SOURCE=crsp:
    FACTORLAB_SOURCE=crsp python scripts/validate_factors.py
    FACTORLAB_SOURCE=crsp python scripts/run_research.py              # exploration 1990-2017
    FACTORLAB_SOURCE=crsp python scripts/run_test_c2.py               # SUE event drift, exploration
"""
import os
import sys
from pathlib import Path

os.environ["FACTORLAB_SOURCE"] = "crsp"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factorlab import crsp  # noqa: E402


def main() -> None:
    user = sys.argv[sys.argv.index("--user") + 1] if "--user" in sys.argv else None
    db = crsp.connect(user)
    found = crsp.probe(db)
    if "--probe" in sys.argv:
        return
    ciz = "--legacy" not in sys.argv
    need = ["crsp.msf_v2"] if ciz else ["crsp.msf", "crsp.msenames", "crsp.msedelist"]
    need += ["crsp.ccmxpf_lnkhist", "comp.funda", "comp.fundq"]
    missing = [t for t in need if not found.get(t)]
    if missing:
        sys.exit(f"Missing tables for this account: {missing}. Try --legacy, or ask the Stevens WRDS reps.")
    crsp.build(db, ciz=ciz)


if __name__ == "__main__":
    main()
