"""Écriture en français : pluriels et nombres à virgule."""

from __future__ import annotations

from fuji.text import count, number, plural


def test_plurals_follow_french_rules():
    assert count(0, "calque") == "0 calque"  # zéro et un : singulier
    assert count(1, "image") == "1 image" and count(3, "image") == "3 images"
    assert count(2, "canal") == "2 canaux" and count(1, "canal") == "1 canal"
    assert count(2, "manipulation manuelle") == "2 manipulations manuelles"
    assert count(4, "champ") == "4 champs" and plural(5, "résultat") == "résultats"
    assert plural(2, "sera", "seront") == "seront" and plural(1, "sera", "seront") == "sera"
    assert count(3, "prix") == "3 prix"  # déjà terminé par s, x ou z


def test_numbers_are_written_with_a_comma():
    assert number(12.5) == "12,5" and number(0.29067) == "0,2907" and number(1234.4) == "1234"
    assert number(123456.7) == "123457" and number(10, 2) == "10,00" and number(-0.5, 1) == "-0,5"
    assert number(float("nan")) == "—" and number(3) == "3"
