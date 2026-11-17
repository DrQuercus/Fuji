"""Mode d'emploi des outils d'analyse (menus, résumés, démarches) et calcul du contraste."""

from __future__ import annotations

import numpy as np

from fuji.analysis import list_operations
from fuji.analysis.guide import HELP, MENUS, OTHER_MENU, RECIPES, help_of, menu_of


def test_every_operation_has_a_menu_and_a_plain_explanation():
    names = {op.name for op in list_operations()}
    in_menus = [n for m in MENUS for n in m.operations]
    assert len(in_menus) == len(set(in_menus)), "un outil est rangé dans deux menus"
    assert names == set(in_menus), "chaque outil a sa place dans un menu, et un seul"
    assert names == set(HELP), "chaque outil a son explication"
    for name, h in HELP.items():
        assert h.summary and h.when, name
        assert not h.summary.endswith("."), name  # la fiche ajoute le point
        assert len(h.summary) <= 70, f"résumé trop long pour tenir sur une ligne : {name}"
    assert len({m.key for m in MENUS}) == len(MENUS)


def test_recipes_point_to_real_tools():
    names = {op.name for op in list_operations()}
    for recipe in RECIPES:
        assert recipe.steps, recipe.key
        for step in recipe.steps:
            assert step.operation is None or step.operation in names, step.operation
            for param, _ in step.params:
                op = next(o for o in list_operations() if o.name == step.operation)
                assert param in {p.name for p in op.params}, (step.operation, param)
    count = next(r for r in RECIPES if r.key == "compter")
    assert [s.operation for s in count.steps][-1] == "Analyse de particules"


def test_tools_added_by_extensions_land_in_other_tools():
    assert menu_of("Outil d'une extension") is OTHER_MENU
    assert (
        help_of("Outil d'une extension", "Fait quelque chose.\n\nDétails.").summary == "Fait quelque chose."
    )
    assert menu_of("Seuillage automatique").key == "isoler"


def test_auto_and_full_contrast():
    from fuji.ui.display_bar import auto_limits, full_limits

    rng = np.random.default_rng(0)
    data = rng.normal(1000, 50, (200, 200))
    data[0, :5] = 60000  # quelques pixels chauds ne doivent pas écraser le contraste
    lo, hi = auto_limits(data)
    assert 800 < lo < 900 and 1100 < hi < 1200
    assert full_limits(data)[1] == 60000
    assert auto_limits(np.full((10, 10), 7.0)) == (7.0, 8.0)  # image uniforme
    flat = np.array([[np.nan, 1.0], [2.0, np.inf]])
    assert full_limits(flat) == (1.0, 2.0)
