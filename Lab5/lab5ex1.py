from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

player_model = BayesianNetwork(
    [
        ("C1", "C2"),
        ("C1", "D1"),
        ("C1", "D3"),
        ("C2", "D2"),
        ("D1", "D2"),
        ("D2", "D3")
    ]
)

cpd_c1 = TabularCPD(
    variable="C1", variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
)

cpd_c2 = TabularCPD(
    variable="C2",
    variable_card=5,
    # Sansa de alegere a doua carte:
    # Prima carte este As.
    values=[
        [0, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0]
    ],
    evidence=["C1"],
    evidence_card=5
)

cpd_d1 = TabularCPD(
    variable="D1",
    variable_card=2,
    # Sansa pentru cand jucatorul 1 trebuie sa parieze/astepte.
    # Primul rand de valori: sansa de pariat, unde prima carte este As.
    # Al doilea rand: sansa de asteptat, prima carte este As.
    values=[
        [1, 0.75, 0.5, 0.25, 0],
        [0, 0.25, 0.5, 0.75, 1]
    ],
    evidence=["C1"],
    evidence_card=5
)

cpd_d2 = TabularCPD(
    variable="D2",
    variable_card=2,
    # Sansa pentru cand jucatorul 2 trebuie sa parieze/astepte daca
    # jucatorul 1 nu a pariat.
    # Primul rand de valori: sansa de pariat, unde prima carte este As.
    # Al doilea rand: sansa de asteptat/iesit din joc, prima carte este As.
    values=[
        [1, 0.75, 0.5, 0.25, 0],
        [0, 0.25, 0.5, 0.75, 1]
    ],
    evidence=["C2", "D1"],
    evidence_card=[5, 2]
)

cpd_d3 = TabularCPD(
    variable="D3",
    variable_card=2,
    # Sansa pentru cand jucatorul 1 este obligat sa parieze/sa iasa.
    # Primul rand de valori: sansa de pariat, unde prima carte este As.
    # Al doilea rand: sansa de iesit din joc, prima carte este As.
    values=[
        [1, 0.75, 0.5, 0.25, 0],
        [0, 0.25, 0.5, 0.75, 1]
    ],
    evidence=["C1", "D2"],
    evidence_card=[5, 2]
)