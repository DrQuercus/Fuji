"""Menus des outils d'analyse : rangés par tâche, avec une ligne d'explication par outil.

- :class:`ToolMenu` : la recherche, les démarches guidées et les menus repliables
  (« Améliorer l'image », « Isoler les objets »…) ;
- :class:`RecipePage` : une démarche guidée, étape par étape.

Le contenu (menus, résumés, démarches) vient de :mod:`fuji.analysis.guide`.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from fuji.analysis import list_operations
from fuji.analysis.guide import MENUS, OTHER_MENU, RECIPES, Menu, Recipe, help_of, menu_of

from ..components import button, label, section_title, set_icon


def fold(text: str) -> str:
    """Texte sans accents ni majuscules, pour une recherche tolérante (« seuil » = « Seuil »)."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class ClickableRow(QFrame):
    """Ligne cliquable : une icône, un titre en gras et une explication en dessous."""

    clicked = Signal()

    def __init__(self, title: str, summary: str, icon_name: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ToolRow")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(8)
        if icon_name:
            ic = QLabel()
            set_icon(ic, icon_name, "accent", 18)
            ic.setAlignment(Qt.AlignmentFlag.AlignTop)
            lay.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        self.title = label(title, "ToolTitle")
        self.summary = label(summary, "hint")
        texts.addWidget(self.title)
        texts.addWidget(self.summary)
        lay.addLayout(texts, 1)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class MenuSection(QWidget):
    """Un menu repliable : son titre (clic pour ouvrir / fermer) et ses outils."""

    toggled = Signal(bool)

    def __init__(self, menu: Menu, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.menu = menu
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self.header = ClickableRow(menu.title, menu.intro, menu.icon)
        self.header.setObjectName("MenuHeader")
        self.chevron = QLabel()
        self.header.layout().addWidget(self.chevron, 0, Qt.AlignmentFlag.AlignVCenter)
        self.header.clicked.connect(lambda: self.set_open(not self.is_open()))
        lay.addWidget(self.header)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(22, 0, 0, 4)
        self.body_lay.setSpacing(1)
        lay.addWidget(self.body)
        self.rows: dict[str, ClickableRow] = {}
        self.set_open(False)

    def add_tool(self, name: str, summary: str, imagej: str | None) -> ClickableRow:
        row = ClickableRow(name, summary)
        if imagej:
            row.setToolTip(f"Équivalent ImageJ : {imagej}")
        self.rows[name] = row
        self.body_lay.addWidget(row)
        return row

    def is_open(self) -> bool:
        return not self.body.isHidden()

    def set_open(self, on: bool) -> None:
        self.body.setVisible(on)
        set_icon(self.chevron, "down" if on else "right", "muted", 14)
        self.toggled.emit(on)


class ToolMenu(QWidget):
    """Recherche + démarches guidées + menus des outils."""

    operation_chosen = Signal(str)
    recipe_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher un outil (ex. seuil, flou, compter, Threshold)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        self.no_match = label("Aucun outil ne correspond.", "hint")
        self.no_match.hide()
        lay.addWidget(self.no_match)

        # démarches guidées
        self.recipes_box = QWidget()
        rlay = QVBoxLayout(self.recipes_box)
        rlay.setContentsMargins(0, 2, 0, 4)
        rlay.setSpacing(1)
        rlay.addWidget(section_title("Démarches guidées"))
        self.recipe_rows: dict[str, ClickableRow] = {}
        for recipe in RECIPES:
            row = ClickableRow(recipe.title, "", "sparkle")
            row.summary.hide()  # le titre suffit ; l'explication est dans l'infobulle
            row.setToolTip(recipe.intro)
            row.clicked.connect(lambda k=recipe.key: self.recipe_chosen.emit(k))
            self.recipe_rows[recipe.key] = row
            rlay.addWidget(row)
        lay.addWidget(self.recipes_box)

        # menus par tâche
        lay.addWidget(section_title("Tous les outils, par tâche"))
        self.sections: dict[str, MenuSection] = {}
        ops = list_operations()
        for menu in (*MENUS, OTHER_MENU):
            names = [o.name for o in ops if menu_of(o.name) is menu]
            if not names:
                continue
            # ordre des menus du guide (du plus courant au plus rare) ; puis les extensions
            order = {n: i for i, n in enumerate(menu.operations)}
            section = MenuSection(menu)
            for op in sorted((o for o in ops if o.name in names), key=lambda o: order.get(o.name, 999)):
                row = section.add_tool(op.name, help_of(op.name, op.description).summary, op.imagej)
                row.clicked.connect(lambda n=op.name: self.operation_chosen.emit(n))
            self.sections[menu.key] = section
            lay.addWidget(section)
        self._haystack = {
            o.name: fold(
                " ".join(
                    (
                        o.name,
                        o.category,
                        o.description,
                        o.imagej or "",
                        help_of(o.name, o.description).summary,
                        help_of(o.name).when,
                        menu_of(o.name).title,
                    )
                )
            )
            for o in ops
        }
        self._recipe_haystack = {
            r.key: fold(" ".join((r.title, r.intro, *(s.text for s in r.steps)))) for r in RECIPES
        }
        self._open_before_search: set[str] = set()

    # ---------------------------------------------------------------- recherche
    def _filter(self, text: str) -> None:
        words = fold(text).split()
        searching = bool(words)
        if searching and not self._open_before_search:
            self._open_before_search = {k for k, s in self.sections.items() if s.is_open()} | {"_"}
        found = False
        for section in self.sections.values():
            hits = 0
            for name, row in section.rows.items():
                ok = all(w in self._haystack.get(name, fold(name)) for w in words)
                row.setVisible(ok)
                hits += ok
            section.setVisible(hits > 0)
            if searching:
                section.set_open(hits > 0)
            found |= hits > 0
        recipes = 0
        for key, row in self.recipe_rows.items():
            ok = all(w in self._recipe_haystack[key] for w in words)
            row.setVisible(ok)
            recipes += ok
        self.recipes_box.setVisible(recipes > 0)
        self.no_match.setVisible(searching and not found and not recipes)
        if not searching and self._open_before_search:  # fin de la recherche : menus comme avant
            for key, section in self.sections.items():
                section.set_open(key in self._open_before_search)
            self._open_before_search = set()

    def visible_operations(self) -> list[str]:
        """Outils affichés (après la recherche), dans l'ordre des menus."""
        return [
            name
            for section in self.sections.values()
            if not section.isHidden()
            for name, row in section.rows.items()
            if not row.isHidden()
        ]

    def open_menu_of(self, name: str) -> None:
        """Déplie le menu qui contient cet outil (en revenant de sa fiche)."""
        key = menu_of(name).key
        if key in self.sections:
            self.sections[key].set_open(True)


class RecipePage(QWidget):
    """Une démarche guidée : la liste des étapes, chacune avec son bouton."""

    back = Signal()
    open_step = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        back = button("Tous les outils", "left", object_name="ghost")
        back.clicked.connect(self.back.emit)
        lay.addWidget(back, 0, Qt.AlignmentFlag.AlignLeft)
        self.title = label("", "step")
        self.intro = label("", "hint")
        lay.addWidget(self.title)
        lay.addWidget(self.intro)
        self.steps_box = QWidget()
        self.steps_lay = QVBoxLayout(self.steps_box)
        self.steps_lay.setContentsMargins(0, 4, 0, 0)
        self.steps_lay.setSpacing(6)
        lay.addWidget(self.steps_box)
        lay.addStretch(1)
        self.recipe: Recipe | None = None
        self.step_buttons: list[Any] = []
        self._rows: list[QFrame] = []

    def show_recipe(self, recipe: Recipe, current: int = 0, done: set[int] | None = None) -> None:
        self.recipe = recipe
        self.title.setText(recipe.title)
        self.intro.setText(recipe.intro)
        while self.steps_lay.count():
            item = self.steps_lay.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.step_buttons, self._rows = [], []
        done = done or set()
        for i, step in enumerate(recipe.steps):
            row = QFrame()
            row.setObjectName("StepCurrent" if i == current else "Step")
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 6, 8, 6)
            rl.setSpacing(8)
            num = QLabel("✓" if i in done else str(i + 1))
            num.setObjectName("StepNumberDone" if i in done else "StepNumber")
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rl.addWidget(num, 0, Qt.AlignmentFlag.AlignTop)
            texts = QVBoxLayout()
            texts.setSpacing(4)
            text = ("Si besoin — " if step.optional else "") + step.text
            texts.addWidget(label(text))
            if step.operation:
                b = button(step.operation, "right", object_name="primary" if i == current else None)
                b.setToolTip(help_of(step.operation).summary)
            elif step.tool:
                b = button("Dessiner sur l'image", "edit", object_name="primary" if i == current else None)
            else:
                b = None
            if b is not None:
                b.clicked.connect(lambda _=False, k=i: self.open_step.emit(k))
                texts.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
            self.step_buttons.append(b)
            rl.addLayout(texts, 1)
            self._rows.append(row)
            self.steps_lay.addWidget(row)


__all__ = ["ClickableRow", "MenuSection", "RecipePage", "ToolMenu", "fold"]
