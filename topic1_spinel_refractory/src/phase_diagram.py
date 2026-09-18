"""
phase_diagram.py
-----------------
Builds and plots the MgO-Al2O3-MgAl2O4 phase stability diagram using
pymatgen PDEntry/PhaseDiagram objects and literature/CALPHAD formation
enthalpies. The code documents how to swap in Materials Project entries.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

OUT_DIR = "/home/claude/topic1_spinel_refractory/outputs"
KJ_PER_MOL_TO_EV = 1 / 96.485


def dHf_MgAl2O4_kJmol_global():
    return -601.6 - 1675.7 - 25.0


def build_literature_entries():
    mg_entry = PDEntry(Composition("Mg1"), 0.0, name="Mg (metal, ref)")
    al_entry = PDEntry(Composition("Al1"), 0.0, name="Al (metal, ref)")
    o_entry = PDEntry(Composition("O2"), 0.0, name="O2 (gas, ref)")

    mgo_entry = PDEntry(
        Composition("Mg1 O1"), -601.6 * KJ_PER_MOL_TO_EV,
        name="MgO (periclase)")
    al2o3_entry = PDEntry(
        Composition("Al2 O3"), -1675.7 * KJ_PER_MOL_TO_EV,
        name="Al2O3 (corundum)")
    spinel_entry = PDEntry(
        Composition("Mg1 Al2 O4"),
        dHf_MgAl2O4_kJmol_global() * KJ_PER_MOL_TO_EV,
        name="MgAl2O4 (spinel)")
    )
    return [mg_entry, al_entry, o_entry, mgo_entry, al2o3_entry, spinel_entry]


def build_entries_from_mp(api_key):
    """Reference implementation for swapping in Materials Project data."""
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        return mpr.get_entries_in_chemsys(["Mg", "Al", "O"])


def analyze_phase_diagram():
    entries = build_literature_entries()
    pd_obj = PhaseDiagram(entries)

    print("Stable phases:")
    for entry in pd_obj.stable_entries:
        print(f"  {entry.name:20s} E_above_hull={pd_obj.get_e_above_hull(entry):.5f}")

    spinel = next(e for e in entries if "MgAl2O4" in e.name)
    e_hull = pd_obj.get_e_above_hull(spinel)
    print(f"\nSpinel E_above_hull={e_hull:.5f} eV/atom")

    x_mgo = [1.0, 1 / 3, 0.0]
    y = [-601.6, dHf_MgAl2O4_kJmol_global(), -1675.7]
    labels = ["MgO", "MgAl2O4\n(spinel)", "Al2O3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_mgo, y, "o-", markersize=9, linewidth=2)
    for x, yy, label in zip(x_mgo, y, labels):
        ax.annotate(label, (x, yy), textcoords="offset points", xytext=(0, 12), ha="center")
    ax.set_xlabel("Mole fraction MgO")
    ax.set_ylabel("Formation enthalpy (kJ/mol formula unit)")
    ax.set_title("MgO-Al2O3 Pseudo-Binary: Spinel Intermediate")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mgo_al2o3_phase_diagram.png", dpi=150)
    return pd_obj


if __name__ == "__main__":
    analyze_phase_diagram()
