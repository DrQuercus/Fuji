"""Liste des images prises : chaque capture y arrive, déjà enregistrée sur le disque.

On y retrouve aussi les images composées en direct (profondeur de champ étendue,
panoramas), les mosaïques et les acquisitions. Un double-clic ouvre l'image dans l'espace
Analyse : la prise de vue et l'analyse restent séparées.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QSize, Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.io.captures import KINDS, CaptureInfo, CaptureStore, thumbnail

from . import icons
from .components import EmptyState, button, hbox, hint
from .prefs import settings
from .theme import palette

ICON = QSize(104, 70)
KIND_LABELS = {
    "capture": "capture",
    "edf": "profondeur de champ étendue",
    "panorama": "panorama",
    "mosaique": "mosaïque",
    "acquisition": "acquisition",
}
#: nombre d'images gardées aussi en mémoire (ouverture instantanée dans Analyse)
_KEEP_IN_MEMORY = 2
_INFO = Qt.ItemDataRole.UserRole


def to_pixmap(thumb: np.ndarray) -> QPixmap:
    """Vignette 8 bits (gris ou RVB) → image Qt (copiée : le tableau peut disparaître)."""
    thumb = np.ascontiguousarray(thumb)
    h, w = thumb.shape[:2]
    if thumb.ndim == 3:
        image = QImage(thumb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        image = QImage(thumb.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(image.copy())


def _icon(pixmap: QPixmap) -> QIcon:
    """Vignette identique, sélectionnée ou non (Qt la teinterait de la couleur de sélection)."""
    icon = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Selected, QIcon.Mode.Active):
        icon.addPixmap(pixmap, mode)
    return icon


def pretty_path(path: Path) -> str:
    """Chemin lisible : ``~/Fuji/Captures/…`` plutôt que le chemin complet."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def captures_root() -> Path | None:
    value = settings().value("captures/folder", "")
    return Path(value) if value else None


class GalleryPanel(QWidget):
    #: ouvrir une image dans l'espace Analyse : (CaptureInfo, données en mémoire ou None,
    #: garder les images déjà ouvertes)
    open_requested = Signal(object, object, bool)
    #: images supprimées (liste de CaptureInfo) : à fermer si elles sont ouvertes
    deleted = Signal(list)
    #: image renommée : (ancien nom, nouvelle CaptureInfo)
    renamed = Signal(str, object)
    open_file_requested = Signal()
    message = Signal(str)
    count_changed = Signal(int)
    #: fin d'un enregistrement en arrière-plan (interne : ramène le résultat dans le thread Qt)
    _saved = Signal(object, object, object)

    def __init__(self, store: CaptureStore | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store or CaptureStore(captures_root())
        self._memory: dict[str, np.ndarray] = {}
        #: images dont l'enregistrement a échoué : gardées tant que Fuji est ouvert
        self._unsaved: dict[str, np.ndarray] = {}
        self._reserved: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._saved.connect(self._on_saved)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.folder = hint()
        self.folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        show = button("Dossier", "folder", tooltip="Ouvrir le dossier des images dans le Finder")
        show.clicked.connect(self.show_folder)
        change = button("Changer…", tooltip="Choisir où enregistrer les images prises")
        change.clicked.connect(self.change_folder)
        other = button("Ouvrir…", "open", tooltip="Ouvrir une autre image (fichier) dans Analyse (Ctrl+O)")
        other.clicked.connect(self.open_file_requested.emit)
        lay.addWidget(self.folder)
        lay.addWidget(hbox(show, change, other))

        self.list = QListWidget()
        self.list.setIconSize(ICON)
        self.list.setSpacing(2)
        self.list.setWordWrap(True)
        self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.itemDoubleClicked.connect(lambda item: self.open_item(item))  # remplace
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.empty = EmptyState(
            "gallery",
            "Aucune image aujourd'hui",
            "Chaque capture (bouton Capturer ou Ctrl+K) arrive ici, déjà enregistrée. "
            "Double-clic sur une image : l'ouvrir dans Analyse.",
        )
        self.views = QStackedWidget()
        self.views.addWidget(self.empty)
        self.views.addWidget(self.list)
        lay.addWidget(self.views, 1)

        self.open_btn = button(
            "Ouvrir dans Analyse",
            "chart",
            primary=True,
            tooltip="Remplace l'image ouverte dans Analyse. Pour en garder plusieurs ouvertes : "
            "clic droit → « Ouvrir à côté »",
        )
        self.open_btn.clicked.connect(self.open_selected)
        self.rename_btn = button("Renommer", "edit")
        self.rename_btn.clicked.connect(self.rename_selected)
        self.delete_btn = button("Supprimer", "trash", object_name="danger")
        self.delete_btn.clicked.connect(self.delete_selected)
        lay.addWidget(self.open_btn)
        lay.addWidget(hbox(self.rename_btn, self.delete_btn))
        self.reload()

    # ------------------------------------------------------------------ liste
    def reload(self) -> None:
        """Relit le dossier du jour (au démarrage, ou après un changement de dossier)."""
        self.list.clear()
        self.folder.setText(f"Enregistrées dans : {pretty_path(self.store.day_folder())}")
        for info in self.store.list():
            self._insert(info, None)
        self._changed()

    def entries(self) -> list[CaptureInfo]:
        return [self.list.item(i).data(_INFO) for i in range(self.list.count())]

    def _insert(self, info: CaptureInfo, pixmap: QPixmap | None, pending: bool = False) -> QListWidgetItem:
        item = QListWidgetItem()
        item.setData(_INFO, info)
        self.list.insertItem(0, item)  # la plus récente en haut
        self._describe(item, pending)
        if pixmap is None and info.thumbnail_path.exists():
            pixmap = QPixmap(str(info.thumbnail_path))
        if pixmap is not None and not pixmap.isNull():
            item.setIcon(_icon(pixmap))
        else:
            item.setIcon(icons.icon("gallery", palette().faint, 48))
            if info.path.exists():
                self._thumbnail_later(item)
        return item

    def _describe(self, item: QListWidgetItem, pending: bool = False) -> None:
        info: CaptureInfo = item.data(_INFO)
        state = "enregistrement…" if pending else info.time_text
        kind = KIND_LABELS.get(info.kind, info.kind)
        lines = [info.name, info.summary or kind, f"{state} · {kind}" if info.summary else state]
        item.setText("\n".join(lines))
        item.setToolTip(str(info.path))

    def _item_of(self, info: CaptureInfo) -> QListWidgetItem | None:
        for i in range(self.list.count()):
            if self.list.item(i).data(_INFO).path == info.path:
                return self.list.item(i)
        return None

    def _changed(self) -> None:
        self.views.setCurrentWidget(self.list if self.list.count() else self.empty)
        self._update_buttons()
        self.count_changed.emit(self.list.count())

    def _update_buttons(self) -> None:
        selected = self.list.selectedItems()
        self.open_btn.setEnabled(bool(selected))
        self.delete_btn.setEnabled(bool(selected))
        single = len(selected) == 1 and selected[0].data(_INFO).kind != "acquisition"
        self.rename_btn.setEnabled(single)

    def _thumbnail_later(self, item: QListWidgetItem) -> None:
        """Vignette d'une image sans résumé (acquisition, fichier déposé à la main)."""
        info: CaptureInfo = item.data(_INFO)

        def work() -> np.ndarray:
            import tifffile

            with tifffile.TiffFile(info.path) as tif:
                page = tif.pages[0]
                return thumbnail(page.asarray())

        worker = create_worker(work, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(lambda thumb: self._set_thumb(info, thumb))
        worker.start()

    def _set_thumb(self, info: CaptureInfo, thumb: np.ndarray) -> None:
        item = self._item_of(info)
        if item is not None:
            item.setIcon(_icon(to_pixmap(thumb)))

    # ------------------------------------------------------------------ ajout
    def add_image(
        self,
        data: np.ndarray,
        *,
        kind: str = "capture",
        metadata: dict[str, Any] | None = None,
        pixel_size_um: float | None = None,
        summary: str | None = None,
    ) -> CaptureInfo:
        """Ajoute une image à la liste et l'enregistre en arrière-plan (l'interface ne se
        fige pas pendant l'écriture d'une image 4K). Renvoie ce qui la décrit."""
        from fuji.io.captures import describe

        folder = self.store.day_folder()
        name = self._reserve(KINDS.get(kind, kind), folder)
        info = CaptureInfo(
            path=folder / f"{name}.ome.tif",
            kind=kind,
            summary=describe(metadata) if summary is None else summary,
            shape=tuple(int(s) for s in np.shape(data)),
            pixel_size_um=pixel_size_um,
            metadata=dict(metadata or {}),
        )
        thumb = thumbnail(data)
        self._remember(info.name, data)
        item = self._insert(info, to_pixmap(thumb), pending=True)
        self.list.clearSelection()
        self.list.setCurrentItem(item)
        self._changed()

        def work() -> None:
            try:
                saved = self.store.save(
                    data,
                    kind=kind,
                    metadata=metadata,
                    pixel_size_um=pixel_size_um,
                    name=name,
                    summary=info.summary,
                    thumb=thumb,
                )
                self._saved.emit(info, saved, None)
            except Exception as exc:
                self._unsaved[name] = data
                self._saved.emit(info, None, exc)

        thread = threading.Thread(target=work, name=f"fuji-enregistre-{name}", daemon=False)
        self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()
        return info

    def add_existing(self, path: str | Path, kind: str = "acquisition", summary: str = "") -> CaptureInfo:
        """Ajoute une image déjà enregistrée ailleurs (fichier d'acquisition)."""
        info = self.store.add_existing(path, kind, summary)
        self._insert(info, None)
        self._changed()
        return info

    def _reserve(self, prefix: str, folder: Path) -> str:
        name = self.store.next_name(prefix, folder)
        while name in self._reserved:  # deux captures rapprochées : la première s'écrit encore
            number = int(name.rsplit("_", 1)[1]) + 1
            name = f"{prefix}_{number:03d}"
        self._reserved.add(name)
        return name

    def _remember(self, name: str, data: np.ndarray) -> None:
        self._memory[name] = data
        while len(self._memory) > _KEEP_IN_MEMORY:
            self._memory.pop(next(iter(self._memory)))

    def _on_saved(self, info: CaptureInfo, saved: CaptureInfo | None, error: Exception | None) -> None:
        self._reserved.discard(info.name)
        item = self._item_of(info)
        if error is not None:
            if item is not None:
                item.setText(f"{info.name}\n⚠ NON enregistrée : {error}")
            QMessageBox.warning(
                self,
                "Image non enregistrée",
                f"« {info.name} » n'a pas pu être enregistrée dans\n{info.path.parent}\n\n{error}\n\n"
                "Elle reste ouverte dans Fuji : choisissez un autre dossier (« Changer… ») ou "
                "ouvrez-la dans Analyse pour l'enregistrer ailleurs.",
            )
            return
        if item is not None and saved is not None:
            item.setData(_INFO, saved)
            self._describe(item)

    def wait_pending(self, timeout: float = 30.0) -> None:
        """Attend la fin des enregistrements en cours (avant de quitter)."""
        for thread in self._threads:
            thread.join(timeout)

    # ------------------------------------------------------------------ actions
    def selected(self) -> list[CaptureInfo]:
        return [item.data(_INFO) for item in self.list.selectedItems()]

    def open_item(self, item: QListWidgetItem, keep: bool = False) -> None:
        """Ouvre l'image dans Analyse, à la place de celle qui y est (``keep`` : à côté)."""
        info: CaptureInfo = item.data(_INFO)
        data = self._memory.get(info.name)
        self.open_requested.emit(info, data if data is not None else self._unsaved.get(info.name), keep)

    def open_selected(self, keep: bool = False) -> None:
        """Plusieurs images choisies : la première remplace l'image ouverte, les autres
        s'ouvrent à côté."""
        for i, item in enumerate(reversed(self.list.selectedItems())):
            self.open_item(item, keep=keep or i > 0)

    def rename_selected(self) -> None:
        items = self.list.selectedItems()
        if len(items) != 1:
            return
        info: CaptureInfo = items[0].data(_INFO)
        if info.name in self._reserved:
            self.message.emit("Patientez : l'image est en cours d'enregistrement")
            return
        name, ok = QInputDialog.getText(self, "Renommer", "Nouveau nom :", text=info.name)
        if not ok or not name.strip() or name.strip() == info.name:
            return
        try:
            renamed = self.store.rename(info, name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Renommer", str(exc))
            return
        if info.name in self._memory:
            self._memory[renamed.name] = self._memory.pop(info.name)
        items[0].setData(_INFO, renamed)
        self._describe(items[0])
        self.message.emit(f"Renommée : {renamed.name}")
        self.renamed.emit(info.name, renamed)

    def delete_selected(self) -> None:
        items = self.list.selectedItems()
        if not items:
            return
        infos = [item.data(_INFO) for item in items]
        if any(info.name in self._reserved for info in infos):
            self.message.emit("Patientez : une image est en cours d'enregistrement")
            return
        files = [i for i in infos if i.kind != "acquisition"]
        text = (
            f"Mettre {len(files)} image(s) à la corbeille ?"
            if files
            else "Retirer ces acquisitions de la liste ? (leurs fichiers ne sont pas supprimés)"
        )
        if QMessageBox.question(self, "Supprimer", text) != QMessageBox.StandardButton.Yes:
            return
        from qtpy.QtCore import QFile

        for item, info in zip(items, infos, strict=True):
            if info.kind != "acquisition" and info.path.exists():
                trashed = QFile.moveToTrash(str(info.path))
                if not (trashed[0] if isinstance(trashed, tuple) else trashed):
                    info.path.unlink()  # pas de corbeille (certains disques réseau)
                self.store.forget(info)
            self._memory.pop(info.name, None)
            self.list.takeItem(self.list.row(item))
        self._changed()
        self.deleted.emit(infos)

    def _menu(self, pos: Any) -> None:
        if not self.list.selectedItems():
            return
        menu = QMenu(self)
        menu.addAction("Ouvrir dans Analyse (remplace l'image ouverte)", self.open_selected)
        menu.addAction("Ouvrir à côté (garder les images ouvertes)", lambda: self.open_selected(keep=True))
        if self.rename_btn.isEnabled():
            menu.addAction("Renommer…", self.rename_selected)
        menu.addAction("Afficher dans le Finder", lambda: self.show_folder(self.selected()[0].path.parent))
        menu.addSeparator()
        menu.addAction("Supprimer…", self.delete_selected)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def show_folder(self, folder: Path | None = None) -> None:
        folder = folder or self.store.day_folder()
        if not folder.exists():
            folder = self.store.root
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def change_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier des images prises", str(self.store.root))
        if not folder:
            return
        self.wait_pending()
        self.store = CaptureStore(folder)
        settings().setValue("captures/folder", folder)
        self.reload()
        self.message.emit(f"Les images seront enregistrées dans {pretty_path(self.store.day_folder())}")
