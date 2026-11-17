"""Thème visuel de Fuji : palettes (sombre / clair), couleur d'accent, taille du texte.

Tout le style de l'application est généré ici à partir de quelques couleurs
(« jetons ») : changer le thème ou l'accent restyle instantanément toute
l'interface, y compris les widgets de napari (thème napari enregistré à la volée).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import QApplication

from .prefs import settings

ACCENTS: dict[str, str] = {
    "Bleu": "#4f8cff",
    "Indigo": "#7b6cf6",
    "Turquoise": "#14b8a6",
    "Vert": "#22c55e",
    "Ambre": "#f5a524",
    "Corail": "#ff6b57",
    "Rose": "#ec4899",
}
FONT_SIZES = {"Petit": 12, "Normal": 13, "Grand": 15}


@dataclass(frozen=True)
class Palette:
    mode: str
    bg: str
    surface: str
    surface2: str
    surface3: str
    input: str
    border: str
    border_strong: str
    text: str
    muted: str
    faint: str
    canvas: str
    success: str = "#2fbf71"
    warning: str = "#f2a93b"
    danger: str = "#f05252"
    accent: str = ACCENTS["Bleu"]
    font_size: int = 13

    # dérivés de l'accent -------------------------------------------------------
    def _mix(self, a: str, b: str, t: float) -> str:
        ca, cb = QColor(a), QColor(b)
        return QColor(
            round(ca.red() + (cb.red() - ca.red()) * t),
            round(ca.green() + (cb.green() - ca.green()) * t),
            round(ca.blue() + (cb.blue() - ca.blue()) * t),
        ).name()

    @property
    def accent_hover(self) -> str:
        return self._mix(self.accent, "#ffffff", 0.12)

    @property
    def accent_pressed(self) -> str:
        return self._mix(self.accent, "#000000", 0.15)

    @property
    def accent_soft(self) -> str:
        """Accent très atténué (fond des éléments sélectionnés)."""
        return self._mix(self.surface2, self.accent, 0.22)

    @property
    def accent_text(self) -> str:
        """Texte lisible sur fond d'accent."""
        c = QColor(self.accent)
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        return "#0b0d10" if lum > 165 else "#ffffff"

    @property
    def hover(self) -> str:
        return self._mix(self.surface3, self.text, 0.06)


DARK = Palette(
    mode="sombre",
    bg="#0f1115",
    surface="#151920",
    surface2="#1b2028",
    surface3="#232a34",
    input="#11151b",
    border="#29303b",
    border_strong="#3a4352",
    text="#e8ebf0",
    muted="#98a2b3",
    faint="#636d7c",
    canvas="#07090c",
)

LIGHT = Palette(
    mode="clair",
    bg="#e9edf2",
    surface="#f6f7f9",
    surface2="#ffffff",
    surface3="#eef1f5",
    input="#ffffff",
    border="#dbe0e7",
    border_strong="#c3cbd6",
    text="#18202c",
    muted="#5b6677",
    faint="#8b95a5",
    canvas="#1b1f25",  # le canevas reste sombre : les images s'y lisent mieux
)

PALETTES = {"sombre": DARK, "clair": LIGHT}


class ThemeManager(QObject):
    """Thème courant + notification des changements (icônes, histogramme...)."""

    changed = Signal(object)  # Palette

    _instance: ThemeManager | None = None

    def __init__(self) -> None:
        super().__init__()
        s = settings()
        mode = s.value("theme/mode", "sombre")
        accent = s.value("theme/accent", "Bleu")
        size = s.value("theme/font", "Normal")
        self.palette = self._build(mode, accent, size)

    @classmethod
    def get(cls) -> ThemeManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _build(mode: str, accent: str, size: str) -> Palette:
        base = PALETTES.get(mode, DARK)
        return replace(
            base,
            accent=ACCENTS.get(accent, accent if str(accent).startswith("#") else ACCENTS["Bleu"]),
            font_size=FONT_SIZES.get(size, 13),
        )

    def settings(self) -> tuple[str, str, str]:
        s = settings()
        return (
            s.value("theme/mode", "sombre"),
            s.value("theme/accent", "Bleu"),
            s.value("theme/font", "Normal"),
        )

    def set(self, *, mode: str | None = None, accent: str | None = None, size: str | None = None) -> None:
        cur_mode, cur_accent, cur_size = self.settings()
        mode, accent, size = mode or cur_mode, accent or cur_accent, size or cur_size
        s = settings()
        s.setValue("theme/mode", mode)
        s.setValue("theme/accent", accent)
        s.setValue("theme/font", size)
        self.palette = self._build(mode, accent, size)
        app = QApplication.instance()
        if app is not None:
            apply(app, self.palette)  # type: ignore[arg-type]
        self.changed.emit(self.palette)

    def toggle_mode(self) -> None:
        self.set(mode="clair" if self.palette.mode == "sombre" else "sombre")


def palette() -> Palette:
    return ThemeManager.get().palette


# ------------------------------------------------------------- ressources SVG
def _asset_dir(p: Palette) -> Path:
    folder = Path(tempfile.gettempdir()) / "fuji-theme" / f"{p.mode}-{p.accent.strip('#')}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_assets(p: Palette) -> dict[str, str]:
    """Petites images (flèches, coche) utilisées par la feuille de style."""
    folder = _asset_dir(p)
    svgs = {
        "down": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{p.muted}" stroke-width="2.4" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>',
        "up": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{p.muted}" stroke-width="2.4" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>',
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{p.accent_text}" stroke-width="3.2" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    }
    out = {}
    for name, svg in svgs.items():
        path = folder / f"{name}.svg"
        if not path.exists():
            path.write_text(svg, encoding="utf-8")
        out[name] = path.as_posix()
    return out


# ------------------------------------------------------------- feuille de style
def build_qss(p: Palette) -> str:
    a = _write_assets(p)
    fs = p.font_size
    return f"""
* {{ font-size: {fs}px; }}
QMainWindow, QDialog, QMessageBox {{ background: {p.bg}; }}
QDialog QScrollArea > QWidget > QWidget {{ background: transparent; }}
QAbstractScrollArea {{ background: transparent; }}
QWidget {{ color: {p.text}; }}
QLabel {{ background: transparent; }}

/* ---------- cadres de l'application ---------- */
#TopBar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
#Sidebar {{ background: {p.surface}; border-right: 1px solid {p.border}; }}
#SidePanel {{ background: {p.surface}; border-left: 1px solid {p.border}; }}
#SidePanel QScrollArea, #SidePanel QScrollArea > QWidget > QWidget {{ background: {p.surface}; }}
#QuickBar {{ background: {p.surface}; border-top: 1px solid {p.border}; }}
#CanvasFrame {{ background: {p.canvas}; }}
#AppName {{ font-size: {fs + 3}px; font-weight: 800; letter-spacing: 0.5px; }}
#PanelTitle {{ font-size: {fs + 5}px; font-weight: 700; }}
#PanelSubtitle {{ color: {p.muted}; }}
#SectionTitle {{ color: {p.muted}; font-size: {fs - 1}px; font-weight: 700; letter-spacing: 0.6px; }}
#Card {{ background: {p.surface2}; border: 1px solid {p.border}; border-radius: 12px; }}
#Card QLabel#CardTitle {{ font-weight: 700; }}
#Separator {{ background: {p.border}; border: none; max-height: 1px; min-height: 1px; }}

/* ---------- menus des outils d'analyse ---------- */
#ToolRow, #MenuHeader {{ background: transparent; border: 1px solid transparent; border-radius: 8px; }}
#ToolRow:hover, #MenuHeader:hover {{ background: {p.surface3}; border-color: {p.border}; }}
#MenuHeader QLabel#ToolTitle {{ font-weight: 700; }}
QLabel#ToolTitle {{ font-weight: 600; }}
#Step {{ background: {p.surface2}; border: 1px solid {p.border}; border-radius: 10px; }}
#StepCurrent {{ background: {p.accent_soft}; border: 1px solid {p.accent}; border-radius: 10px; }}
QLabel#StepNumber, QLabel#StepNumberDone {{ min-width: 22px; max-width: 22px; min-height: 22px;
    max-height: 22px; border-radius: 11px; font-weight: 700; }}
QLabel#StepNumber {{ background: {p.surface3}; color: {p.text}; }}
QLabel#StepNumberDone {{ background: {p.success}; color: #04130a; }}
#Banner {{ background: {p.accent_soft}; border: 1px solid {p.accent}; border-radius: 10px; }}
#VSeparator {{ background: {p.border}; border: none; max-width: 1px; min-width: 1px; }}

/* ---------- textes ---------- */
QLabel#hint {{ color: {p.muted}; font-size: {fs - 1}px; }}
QLabel#faint {{ color: {p.faint}; font-size: {fs - 1}px; }}
QLabel#warning {{ color: {p.warning}; }}
QLabel#error {{ color: {p.danger}; }}
QLabel#success {{ color: {p.success}; }}
QLabel#step {{ font-size: {fs + 3}px; font-weight: 700; padding-top: 4px; }}
QLabel#metric {{ font-size: {fs + 8}px; font-weight: 800; }}
QLabel#badge {{ background: {p.surface3}; color: {p.muted}; border-radius: 9px;
                padding: 2px 8px; font-size: {fs - 2}px; font-weight: 600; }}
QLabel#badgeAccent {{ background: {p.accent_soft}; color: {p.accent}; border-radius: 9px;
                      padding: 2px 8px; font-size: {fs - 2}px; font-weight: 700; }}
QLabel#badgeWarning {{ background: {p._mix(p.surface2, p.warning, 0.2)}; color: {p.warning};
                       border-radius: 9px; padding: 2px 8px; font-size: {fs - 2}px; font-weight: 700; }}

/* ---------- boutons ---------- */
QPushButton, QToolButton {{
    background: {p.surface3}; border: 1px solid {p.border}; border-radius: 8px;
    padding: 6px 12px; color: {p.text};
}}
QPushButton:hover, QToolButton:hover {{ background: {p.hover}; border-color: {p.border_strong}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {p.border}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {p.faint}; background: {p.surface2}; }}
QPushButton:checked {{ background: {p.accent_soft}; border-color: {p.accent}; color: {p.text}; }}
QPushButton#primary {{ background: {p.accent}; border-color: {p.accent}; color: {p.accent_text};
                       font-weight: 700; }}
QPushButton#primary:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton#primary:pressed {{ background: {p.accent_pressed}; }}
QPushButton#primary:disabled {{ background: {p.surface3}; border-color: {p.border}; color: {p.faint}; }}
QPushButton#danger {{ color: {p.danger}; }}
QPushButton#danger:hover {{ background: {p.danger}; border-color: {p.danger}; color: white; }}
QPushButton#ghost, QToolButton#ghost {{ background: transparent; border: 1px solid transparent; }}
QPushButton#ghost:hover, QToolButton#ghost:hover {{ background: {p.surface3}; }}
QPushButton#chip {{ border-radius: 13px; padding: 4px 11px; background: {p.surface3}; }}
QPushButton#chip:checked {{ background: {p.accent}; border-color: {p.accent}; color: {p.accent_text};
                            font-weight: 700; }}
#ModeSwitch {{ background: {p.surface3}; border: 1px solid {p.border}; border-radius: 17px; }}
QPushButton#mode {{ background: transparent; border: none; border-radius: 14px; padding: 5px 13px;
                    color: {p.muted}; font-weight: 600; }}
QPushButton#mode:hover {{ color: {p.text}; }}
QPushButton#mode:checked {{ background: {p.accent}; color: {p.accent_text}; font-weight: 700; }}
QPushButton#live {{ border-radius: 17px; padding: 7px 16px; font-weight: 700; }}
QPushButton#live:checked {{ background: {p.success}; border-color: {p.success}; color: #04130a; }}
QPushButton#capture {{ border-radius: 17px; padding: 7px 16px; font-weight: 700;
                       background: {p.accent}; border-color: {p.accent}; color: {p.accent_text}; }}
QPushButton#capture:hover {{ background: {p.accent_hover}; }}
QPushButton#swatch {{ border-radius: 10px; min-width: 20px; max-width: 20px; min-height: 20px;
                      max-height: 20px; padding: 0; }}

/* ---------- navigation latérale ---------- */
#Sidebar QToolButton {{ background: transparent; border: none; border-radius: 10px;
                        color: {p.muted}; padding: 7px 2px 5px 2px; font-size: {fs - 3}px;
                        font-weight: 600; }}
#Sidebar QToolButton:hover {{ background: {p.surface3}; color: {p.text}; }}
#Sidebar QToolButton:checked {{ background: {p.accent_soft}; color: {p.accent}; }}

/* ---------- champs ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QDateTimeEdit {{
    background: {p.input}; border: 1px solid {p.border}; border-radius: 8px;
    padding: 5px 8px; selection-background-color: {p.accent}; selection-color: {p.accent_text};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus,
QTextEdit:focus {{ border-color: {p.accent}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {p.faint}; background: {p.surface2}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{a["down"]}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {p.surface2}; border: 1px solid {p.border_strong};
    border-radius: 8px; padding: 4px; selection-background-color: {p.accent};
    selection-color: {p.accent_text}; outline: none; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border;
    subcontrol-position: top right; width: 18px; border: none; background: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border;
    subcontrol-position: bottom right; width: 18px; border: none; background: transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{a["up"]}"); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{a["down"]}"); width: 10px;
    height: 10px; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px;
    border: 1.5px solid {p.border_strong}; background: {p.input}; }}
QCheckBox::indicator:hover {{ border-color: {p.accent}; }}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent};
    image: url("{a["check"]}"); }}
QRadioButton::indicator {{ width: 15px; height: 15px; border-radius: 8px;
    border: 1.5px solid {p.border_strong}; background: {p.input}; }}
QRadioButton::indicator:checked {{ background: {p.accent}; border: 4px solid {p.input}; }}

QSlider::groove:horizontal {{ height: 4px; background: {p.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; border: 2px solid {p.accent}; width: 12px;
    height: 12px; margin: -6px 0; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ background: {p.accent}; }}

QProgressBar {{ background: {p.surface3}; border: none; border-radius: 6px; min-height: 12px;
    max-height: 12px; text-align: center; color: {p.text}; font-size: {fs - 3}px; }}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 6px; }}

/* ---------- listes, tableaux ---------- */
QListWidget, QTreeWidget, QTreeView, QTableWidget, QTableView {{
    background: {p.surface2}; border: 1px solid {p.border}; border-radius: 10px;
    alternate-background-color: {p.surface}; gridline-color: {p.border}; outline: none; }}
QListWidget::item, QTreeView::item {{ padding: 5px 8px; border-radius: 6px; }}
QListWidget::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background: {p.accent_soft}; color: {p.text}; }}
QListWidget::item:hover, QTreeView::item:hover {{ background: {p.surface3}; }}
QHeaderView::section {{ background: {p.surface}; color: {p.muted}; border: none;
    border-bottom: 1px solid {p.border}; border-right: 1px solid {p.border}; padding: 6px;
    font-weight: 700; }}
QTableCornerButton::section {{ background: {p.surface}; border: none; }}

/* ---------- groupes = cartes ---------- */
QGroupBox {{ background: {p.surface2}; border: 1px solid {p.border}; border-radius: 12px;
    margin-top: 24px; padding: 12px 10px 10px 10px; font-weight: 700; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 6px;
    top: 4px; padding: 0 2px; color: {p.muted}; font-size: {fs - 1}px; }}
QGroupBox QWidget {{ font-weight: 400; }}

/* ---------- onglets « pilule » ---------- */
QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
QTabBar::tab {{ background: transparent; color: {p.muted}; padding: 6px 12px; margin: 2px;
    border-radius: 8px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {p.surface3}; color: {p.text}; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}
QTabBar::close-button {{ subcontrol-position: right; }}

/* ---------- défilement ---------- */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.border_strong}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p.border_strong}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---------- menus, info-bulles, barre d'état ---------- */
QMenuBar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {p.surface3}; }}
QMenu {{ background: {p.surface2}; border: 1px solid {p.border_strong}; border-radius: 10px;
    padding: 5px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 5px 8px; }}
QToolTip {{ background: {p.surface3}; color: {p.text}; border: 1px solid {p.border_strong};
    border-radius: 6px; padding: 5px 8px; }}
QStatusBar {{ background: {p.surface}; border-top: 1px solid {p.border}; color: {p.muted}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {p.muted}; padding: 0 6px; }}
QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QDockWidget {{ color: {p.muted}; }}
QDockWidget::title {{ background: {p.surface}; padding: 6px 8px; border-bottom: 1px solid {p.border}; }}
QMessageBox QLabel {{ min-width: 280px; }}
"""


def napari_theme_id(p: Palette) -> str:
    return f"fuji-{'dark' if p.mode == 'sombre' else 'light'}-{p.accent.strip('#')}"


def register_napari_theme(p: Palette) -> str:
    """Enregistre un thème napari aux couleurs de Fuji (calques, contrôles, canevas)."""
    from napari.resources._icons import _theme_path
    from napari.utils.theme import available_themes, get_theme, register_theme
    from qtpy.QtCore import QDir

    theme_id = napari_theme_id(p)
    if theme_id in available_themes():
        QDir.addSearchPath(f"theme_{theme_id}", str(_theme_path(theme_id)))
        return theme_id
    base = get_theme("dark" if p.mode == "sombre" else "light")
    data = base.model_dump()
    data.update(
        id=theme_id,
        label=f"Fuji {p.mode}",
        background=p.surface,
        foreground=p.surface2,
        primary=p.surface3,
        secondary=p.border_strong,
        highlight=p.accent_soft,
        text=p.text,
        icon=p.muted,
        current=p.accent,
        canvas=p.canvas,
        console=p.bg,
        font_size=f"{max(8, p.font_size - 4)}pt",
    )
    register_theme(theme_id, data, "fuji")  # génère aussi les icônes napari colorées
    # napari ne déclare le dossier de ces icônes que dans sa propre application : on le fait ici
    QDir.addSearchPath(f"theme_{theme_id}", str(_theme_path(theme_id)))
    return theme_id


def _transparent_containers(qss: str) -> str:
    """napari donne un fond opaque à *tous* les QWidget ; on le rend transparent pour que
    les conteneurs prennent la couleur de la carte ou du panneau qui les contient."""
    import re

    match = re.search(r"(^|\n)QWidget\s*\{([^}]*)\}", qss)
    if not match:
        return qss
    body = re.sub(r"background-color:[^;]+;", "background-color: transparent;", match.group(2), count=1)
    return qss[: match.start(2)] + body + qss[match.end(2) :]


def apply(app: QApplication, p: Palette | None = None) -> str:
    """Applique le thème à toute l'application ; renvoie l'identifiant du thème napari."""
    p = p or palette()
    font = QFont(app.font())
    font.setPixelSize(p.font_size)
    app.setFont(font)
    theme_id = "dark"
    napari_qss = ""
    try:
        from napari.qt import get_stylesheet

        theme_id = register_napari_theme(p)
        napari_qss = _transparent_containers(get_stylesheet(theme_id))
    except Exception:  # pragma: no cover - napari absent ou thème refusé
        pass
    app.setStyleSheet(napari_qss + build_qss(p))
    return theme_id
