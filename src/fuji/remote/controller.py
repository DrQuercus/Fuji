"""Façade de commande du microscope pour un programme ou une IA externe.

Toutes les méthodes prennent et renvoient des valeurs simples (nombres, textes,
dictionnaires) : elles se transposent directement en outils MCP, en API web ou en
scripts. Elles peuvent être appelées depuis n'importe quel thread.

Garde-fous (:class:`SafetyPolicy`) : mode lecture seule, moteurs interdits, taille
maximale d'un déplacement, exposition et intensité bornées, fichiers confinés à un dossier
de travail. Les limites de course du matériel restent vérifiées par les pilotes.

Éléments manuels : quand une action demande la main de l'utilisateur (changer d'objectif
sur un microscope sans moteur), la méthode lève :class:`ManualActionRequired` avec la
consigne à transmettre ; on la rappelle avec ``manual_done=True`` une fois la manipulation
faite. Dans l'interface graphique, la consigne s'affiche directement à l'écran.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from psygnal import Signal

from fuji.analysis import Image, Labels, Points, Rois, Table, get_operation, list_operations
from fuji.core.acquisition import AcquisitionEngine, AcquisitionSummary, AutofocusPlan, plan_acquisition
from fuji.core.manual import ManualAction, ManualActionDeclined
from fuji.core.microscope import Microscope

log = logging.getLogger(__name__)

MAX_ITEMS = 60  # nombre d'images et de résultats gardés en mémoire


class ControlError(RuntimeError):
    """Erreur à rapporter telle quelle à l'appelant (message clair, en français)."""


class NotAllowed(ControlError):
    """Action refusée par les garde-fous."""


class ManualActionRequired(ControlError):
    """Une manipulation de l'utilisateur est nécessaire avant de continuer."""


@dataclass
class SafetyPolicy:
    """Ce qu'un programme externe a le droit de faire."""

    #: observation et analyse seulement : aucun réglage du matériel n'est modifié
    read_only: bool = False
    #: autoriser les moteurs (platine, mise au point, roues)
    allow_motion: bool = True
    #: plus grand déplacement de platine accepté en un appel (µm)
    max_xy_step_um: float = 5000.0
    #: plus grand déplacement de mise au point accepté en un appel (µm)
    max_z_step_um: float = 100.0
    max_exposure_ms: float = 5000.0
    max_light_percent: float = 100.0
    #: autoriser l'enregistrement de calibrations dans le profil du microscope
    allow_profile_changes: bool = False
    #: nombre maximal d'images par acquisition
    max_frames: int = 2000
    #: seul dossier où lire et écrire des fichiers
    workspace: Path = field(default_factory=lambda: Path.home() / "Fuji" / "ia")

    def describe(self) -> dict[str, Any]:
        return {
            "lecture_seule": self.read_only,
            "moteurs_autorises": self.allow_motion and not self.read_only,
            "deplacement_max_platine_um": self.max_xy_step_um,
            "deplacement_max_mise_au_point_um": self.max_z_step_um,
            "exposition_max_ms": self.max_exposure_ms,
            "intensite_max_pourcent": self.max_light_percent,
            "modification_du_profil": self.allow_profile_changes,
            "dossier_de_travail": str(self.workspace),
        }


def _plain(value: Any) -> Any:
    """Convertit les types NumPy en types Python (pour JSON)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


class MicroscopeController:
    """Commande d'un microscope par un programme externe (voir le module).

    ``microscope`` : un :class:`~fuji.core.microscope.Microscope` connecté, ou une fonction
    qui renvoie le microscope courant (l'interface graphique peut en changer).
    ``own_manual_handler`` : vrai si le contrôleur gère lui-même les consignes manuelles
    (mode sans interface) ; faux pour les laisser à l'interface graphique.
    """

    #: une image ou un résultat a été ajouté (nom)
    item_added = Signal(str)
    #: journal des actions, pour l'afficher à l'utilisateur
    action = Signal(str)

    def __init__(
        self,
        microscope: Microscope | Callable[[], Microscope | None],
        policy: SafetyPolicy | None = None,
        *,
        own_manual_handler: bool = True,
        engine: AcquisitionEngine | Callable[[], AcquisitionEngine | None] | None = None,
    ) -> None:
        self._get_scope = microscope if callable(microscope) else (lambda: microscope)
        self._get_engine = engine if callable(engine) else (lambda: engine)
        self.policy = policy or SafetyPolicy()
        self._own_manual = own_manual_handler
        self._lock = threading.RLock()
        self._items: dict[str, Any] = {}
        self._counters: dict[str, int] = {}
        self._manual_confirmed = False
        self._pending_manual: str | None = None
        self._engine: AcquisitionEngine | None = None
        self._acquisition: dict[str, Any] = {"etat": "aucune"}
        self._summary: AcquisitionSummary | None = None

    # ------------------------------------------------------------------ accès
    @property
    def microscope(self) -> Microscope:
        scope = self._get_scope()
        if scope is None or not scope.connected:
            raise ControlError("Aucun microscope n'est connecté")
        return scope

    def _log(self, text: str) -> None:
        log.info("IA : %s", text)
        try:
            self.action.emit(text)
        except Exception:  # un abonné défaillant ne doit pas faire échouer la commande
            log.debug("Abonné du journal en erreur", exc_info=True)

    def _require_write(self, what: str) -> None:
        if self.policy.read_only:
            raise NotAllowed(f"Mode lecture seule : impossible de {what}. L'utilisateur peut l'autoriser.")

    def _require_motion(self) -> None:
        self._require_write("déplacer un moteur")
        if not self.policy.allow_motion:
            raise NotAllowed("Les moteurs ne sont pas autorisés pour cette session.")

    def _manual_handler(self, action: ManualAction) -> bool:
        if self._manual_confirmed:
            return True
        self._pending_manual = action.message
        return False

    def _with_manual(self, fn: Callable[[], Any], manual_done: bool) -> Any:
        """Exécute ``fn`` ; transforme une consigne manuelle en erreur explicite à relayer."""
        scope = self.microscope
        previous = scope.manual_handler
        if self._own_manual:
            scope.manual_handler = self._manual_handler
        self._manual_confirmed, self._pending_manual = manual_done, None
        try:
            return fn()
        except ManualActionDeclined as exc:
            if not self._own_manual:  # la consigne a été affichée à l'écran, et refusée
                raise ControlError(
                    f"L'utilisateur a annulé la manipulation demandée à l'écran ({exc})."
                ) from None
            message = self._pending_manual or str(exc)
            raise ManualActionRequired(
                f"Action manuelle nécessaire : {message} Demandez à l'utilisateur de la faire, puis "
                "rappelez la même commande avec manual_done=true."
            ) from None
        finally:
            self._manual_confirmed = False
            if self._own_manual:
                scope.manual_handler = previous

    def _path(self, filename: str, *, must_exist: bool = False) -> Path:
        """Chemin dans le dossier de travail (aucun accès en dehors)."""
        root = self.policy.workspace.expanduser().resolve()
        path = (root / filename).resolve() if not Path(filename).is_absolute() else Path(filename).resolve()
        if root != path and root not in path.parents:
            raise NotAllowed(f"Les fichiers doivent rester dans le dossier de travail : {root}")
        if must_exist and not path.exists():
            raise ControlError(f"Fichier introuvable : {path}")
        return path

    # ------------------------------------------------------------------- état
    def status(self) -> dict[str, Any]:
        """État complet du microscope : matériel, optique, positions, réglages."""
        scope = self.microscope
        caps = scope.capabilities()
        cam, obj = scope.camera, scope.objective
        out: dict[str, Any] = {
            "microscope": scope.profile.name,
            "peripheriques": [
                {"nom": d.name, "type": d.kind.value, "manuel": bool(d.manual), "libelle": d.label}
                for d in scope.devices.values()
            ],
            "objectif": obj.label if obj else None,
            "objectifs": [
                {"nom": o.label, "grossissement": o.magnification, "ouverture_numerique": o.na}
                for o in scope.profile.objectives
            ],
            "changement_objectif": (
                None
                if scope.objective_changer is None
                else ("manuel" if scope.objective_changer.manual else "motorisé")
            ),
            "canal": scope.channel,
            "canaux": scope.channel_names,
            "taille_pixel_um": _round(scope.pixel_size_um(), 5),
            "origine_taille_pixel": scope.pixel_size_source(),
            "acquisition": dict(self._acquisition),
            "garde_fous": self.policy.describe(),
            "elements_en_memoire": len(self._items),
        }
        if cam is not None:
            h, w = cam.sensor_shape()
            lo, hi = cam.exposure_limits()
            px = scope.pixel_size_um()
            out["camera"] = {
                "nom": cam.label,
                "largeur_px": w,
                "hauteur_px": h,
                "couleur": bool(cam.is_color),
                "bits": cam.bit_depth,
                "exposition_ms": _round(cam.get_exposure()),
                "exposition_limites_ms": [_round(lo), _round(min(hi, self.policy.max_exposure_ms))],
                "gain": _round(cam.get_gain()) if caps["gain"] else None,
                "champ_um": [_round(w * px, 1), _round(h * px, 1)] if px else None,
            }
        stage, focus = scope.xy_stage, scope.focus
        if stage is not None:
            x, y = stage.get_position()
            out["platine"] = {
                "x_um": _round(x, 2),
                "y_um": _round(y, 2),
                "motorisee": not stage.manual,
                "limites_um": _plain(stage.limits()),
                "orientation_calibree": scope.profile.stage_orientation is not None,
            }
        if focus is not None:
            out["mise_au_point"] = {
                "z_um": _round(focus.get_position(), 3),
                "motorisee": not focus.manual,
                "limites_um": _plain(focus.limits()),
            }
        from fuji.core.devices import LightSource

        out["lampes"] = [
            {
                "nom": lamp.name,
                "allumee": bool(lamp.is_on()),
                "intensite_pourcent": _round(lamp.get_intensity(), 1),
                "manuelle": bool(lamp.manual),
            }
            for lamp in scope.devices_of(LightSource)
        ]
        return _plain(out)

    def device_properties(self, device: str) -> list[dict[str, Any]]:
        """Tous les réglages d'un périphérique (nom, valeur, bornes, choix)."""
        dev = self._device(device)
        out = []
        for info in dev.property_infos():
            try:
                value = dev.get_property(info.name)
            except Exception as exc:
                value = f"illisible ({exc})"
            out.append(
                {
                    "nom": info.name,
                    "valeur": value,
                    "type": info.type.value,
                    "lecture_seule": info.read_only,
                    "limites": info.limits,
                    "choix": info.choices,
                    "unite": info.unit,
                    "description": info.description,
                }
            )
        return _plain(out)

    def _device(self, name: str) -> Any:
        scope = self.microscope
        if name not in scope.devices:
            raise ControlError(f"Périphérique inconnu {name!r}. Disponibles : {sorted(scope.devices)}")
        return scope.devices[name]

    def set_device_property(self, device: str, name: str, value: Any) -> dict[str, Any]:
        self._require_write("modifier un réglage")
        dev = self._device(device)
        with self._lock:
            try:
                dev.set_property(name, value)
            except (KeyError, ValueError, PermissionError) as exc:
                raise ControlError(str(exc)) from exc
        self._log(f"{device}.{name} = {value}")
        return {"peripherique": device, "nom": name, "valeur": _plain(dev.get_property(name))}

    # ----------------------------------------------------------------- réglages
    def set_exposure(self, exposure_ms: float) -> dict[str, Any]:
        self._require_write("changer l'exposition")
        cam = self.microscope.require_camera()
        lo, hi = cam.exposure_limits()
        hi = min(hi, self.policy.max_exposure_ms)
        if not lo <= exposure_ms <= hi:
            raise ControlError(f"Exposition hors limites : {exposure_ms} ms (autorisé : {lo:g} à {hi:g} ms)")
        with self._lock:
            cam.set_exposure(float(exposure_ms))
        self._log(f"exposition {exposure_ms:g} ms")
        return {"exposition_ms": _round(cam.get_exposure())}

    def auto_exposure(self, target: float = 0.7) -> dict[str, Any]:
        """Règle l'exposition pour que les hautes lumières atteignent ``target`` de la dynamique."""
        self._require_write("changer l'exposition")
        cam = self.microscope.require_camera()
        if not 0.05 <= target <= 0.95:
            raise ControlError("La cible doit être entre 0,05 et 0,95")
        with self._lock:
            value = cam.auto_expose(target)
            if value > self.policy.max_exposure_ms:
                cam.set_exposure(self.policy.max_exposure_ms)
        self._log(f"exposition automatique → {cam.get_exposure():.1f} ms")
        return {"exposition_ms": _round(cam.get_exposure())}

    def set_gain(self, gain: float) -> dict[str, Any]:
        self._require_write("changer le gain")
        cam = self.microscope.require_camera()
        if not cam.supports_gain:
            raise ControlError("Cette caméra n'a pas de gain réglable")
        lo, hi = cam.gain_limits()
        if not lo <= gain <= hi:
            raise ControlError(f"Gain hors limites : {gain} (autorisé : {lo:g} à {hi:g})")
        with self._lock:
            cam.set_gain(float(gain))
        self._log(f"gain {gain:g}")
        return {"gain": _round(cam.get_gain())}

    def set_objective(self, label: str, manual_done: bool = False) -> dict[str, Any]:
        """Change d'objectif. Tourelle manuelle : l'utilisateur doit le placer lui-même."""
        self._require_write("changer d'objectif")
        scope = self.microscope
        labels = [o.label for o in scope.profile.objectives]
        if label not in labels:
            raise ControlError(f"Objectif inconnu {label!r}. Objectifs : {labels}")
        with self._lock:
            self._with_manual(lambda: scope.set_objective(label), manual_done)
        self._log(f"objectif {label}")
        return {"objectif": label, "taille_pixel_um": _round(scope.pixel_size_um(), 5)}

    def set_channel(self, name: str, manual_done: bool = False) -> dict[str, Any]:
        """Applique un canal (cube, lampes, exposition)."""
        self._require_write("changer de canal")
        scope = self.microscope
        if name not in scope.channel_names:
            raise ControlError(f"Canal inconnu {name!r}. Canaux : {scope.channel_names}")
        with self._lock:
            self._with_manual(lambda: scope.apply_channel(name), manual_done)
        self._log(f"canal {name}")
        return {"canal": scope.channel, "exposition_ms": _round(scope.require_camera().get_exposure())}

    def set_light(
        self,
        on: bool | None = None,
        intensity: float | None = None,
        name: str | None = None,
        manual_done: bool = False,
    ) -> dict[str, Any]:
        """Allume, éteint ou règle une lampe (par défaut la lumière transmise)."""
        self._require_write("régler une lampe")
        from fuji.core.devices import LightSource

        scope = self.microscope
        lamp = scope.transmitted_light if name is None else self._device(name)
        if lamp is None or not isinstance(lamp, LightSource):
            raise ControlError("Aucune lampe à ce nom (voir « lampes » dans l'état du microscope)")
        if intensity is not None and not 0 <= intensity <= self.policy.max_light_percent:
            raise ControlError(f"Intensité autorisée : 0 à {self.policy.max_light_percent:g} %")

        def apply() -> None:
            if intensity is not None:
                lamp.set_intensity(float(intensity))
            if on is not None:
                lamp.set_on(bool(on))

        with self._lock:
            self._with_manual(apply, manual_done)
        self._log(
            f"lampe {lamp.name} : {'allumée' if lamp.is_on() else 'éteinte'}, {lamp.get_intensity():.0f} %"
        )
        return {
            "nom": lamp.name,
            "allumee": bool(lamp.is_on()),
            "intensite_pourcent": _round(lamp.get_intensity(), 1),
        }

    # ------------------------------------------------------------------- images
    def _store(self, item: Any, stem: str, name: str | None = None) -> str:
        with self._lock:
            if not name:
                self._counters[stem] = self._counters.get(stem, 0) + 1
                name = f"{stem}_{self._counters[stem]}"
                while name in self._items:
                    self._counters[stem] += 1
                    name = f"{stem}_{self._counters[stem]}"
            self._items.pop(name, None)
            self._items[name] = item
            if hasattr(item, "name"):
                item.name = name
            while len(self._items) > MAX_ITEMS:  # les plus anciens sont oubliés
                self._items.pop(next(iter(self._items)))
        try:
            self.item_added.emit(name)
        except Exception:
            log.debug("Abonné des éléments en erreur", exc_info=True)
        return name

    def item(self, name: str) -> Any:
        if name not in self._items:
            raise ControlError(f"Aucun élément nommé {name!r}. En mémoire : {list(self._items)}")
        return self._items[name]

    def describe_item(self, name: str) -> dict[str, Any]:
        """Résumé d'une image, d'un masque, d'un tableau, de points ou de ROI en mémoire."""
        item = self.item(name)
        out: dict[str, Any] = {"nom": name}
        if isinstance(item, Table):
            df = item.data
            out.update(
                type="tableau",
                lignes=len(df),
                colonnes=[str(c) for c in df.columns],
                unites=item.units,
                resume=item.summary,
            )
        elif isinstance(item, Points):
            out.update(type="points", nombre=len(item), taille_pixel_um=item.pixel_size_um)
        elif isinstance(item, Rois):
            out.update(type="roi", nombre=len(item), formes=[item.kind(i) for i in range(len(item))])
        else:
            data = item.data
            out.update(
                type="masque" if isinstance(item, Labels) else "image",
                forme=list(data.shape),
                type_pixel=str(data.dtype),
                taille_pixel_um=item.pixel_size_um,
            )
            if isinstance(item, Labels):
                out["objets"] = item.count if data.ndim == 2 else None
                out["fraction_couverte"] = _round(float(np.mean(data > 0)), 4)
            elif data.size:
                finite = data[np.isfinite(data)] if data.dtype.kind == "f" else data
                out.update(
                    couleur=bool(item.is_rgb),
                    minimum=float(finite.min()),
                    maximum=float(finite.max()),
                    moyenne=_round(float(finite.mean()), 2),
                )
                meta = {k: v for k, v in item.metadata.items() if k in _META_KEYS and v is not None}
                if meta:
                    out["metadonnees"] = meta
        return _plain(out)

    def list_items(self) -> list[dict[str, Any]]:
        return [self.describe_item(name) for name in self._items]

    def delete_item(self, name: str) -> dict[str, Any]:
        self.item(name)
        with self._lock:
            del self._items[name]
        return {"supprime": name, "restants": list(self._items)}

    def snap(self, name: str | None = None) -> dict[str, Any]:
        """Capture une image avec les réglages actuels et la garde en mémoire."""
        scope = self.microscope
        with self._lock:
            frame = scope.snap()
        image = Image.from_frame(frame)
        cam = scope.require_camera()
        stored = self._store(image, "capture", name)
        info = self.describe_item(stored)
        top = float(2**cam.bit_depth - 1)
        info["sature_pourcent"] = _round(100.0 * float(np.mean(frame.data >= top)), 3)
        info["niveau_maximal_possible"] = top
        info["diagnostic"] = None  # rien à signaler : exposition correcte
        bright = float(np.percentile(frame.data[::4, ::4], 99.5))
        if info["sature_pourcent"] > 1.0:
            info["diagnostic"] = "Image surexposée : réduisez l'exposition (auto_exposure) avant de mesurer."
        elif bright < 0.05 * top:
            info["diagnostic"] = (
                "Image très sombre : la lampe est-elle allumée (set_light, ou set_channel pour appliquer un "
                "canal) ? Sinon augmentez l'exposition (auto_exposure)."
            )
        self._log(f"capture → {stored}")
        return info

    def preview(self, name: str, max_side: int = 1024, auto_contrast: bool = True) -> tuple[bytes, str]:
        """Aperçu compressé d'une image en mémoire : (octets, format « jpeg » ou « png »).

        L'image est réduite (plus grand côté ≤ ``max_side``) et son contraste étalé pour
        être lisible ; les mesures, elles, se font toujours sur l'image d'origine."""
        from PIL import Image as PILImage

        item = self.item(name)
        if isinstance(item, (Table, Points, Rois)):
            raise ControlError(f"{name!r} n'est pas une image")
        data = np.asarray(item.data)
        rgb = isinstance(item, Image) and item.is_rgb
        while data.ndim > (3 if rgb else 2):  # pile : plan du milieu
            data = data[data.shape[0] // 2]
        step = max(1, int(np.ceil(max(data.shape[:2]) / max(64, max_side))))
        data = data[::step, ::step]
        if isinstance(item, Labels):
            lut = np.zeros((256, 3), np.uint8)
            lut[1:] = np.random.default_rng(0).integers(60, 256, (255, 3))
            out = lut[np.where(data > 0, (data.astype(np.int64) - 1) % 255 + 1, 0)]
            fmt = "png"
        else:
            values = data.astype(np.float64)
            if auto_contrast or values.max() > 255 or values.min() < 0:
                lo, hi = np.percentile(values, (0.1, 99.9))
                hi = hi if hi > lo else lo + 1.0
                values = (values - lo) / (hi - lo) * 255.0
            out = np.clip(values, 0, 255).astype(np.uint8)
            fmt = "jpeg"
        buffer = io.BytesIO()
        PILImage.fromarray(out[..., :3] if out.ndim == 3 else out).save(
            buffer, format=fmt.upper(), quality=85
        )
        return buffer.getvalue(), fmt

    def open_image(self, filename: str, name: str | None = None) -> dict[str, Any]:
        """Charge une image du dossier de travail (TIFF, OME-TIFF, PNG, JPEG)."""
        path = self._path(filename, must_exist=True)
        try:
            image = Image.from_file(path)
        except Exception as exc:
            raise ControlError(f"Lecture impossible : {exc}") from exc
        return self.describe_item(self._store(image, path.stem.replace(".ome", ""), name))

    def save_item(self, name: str, filename: str | None = None) -> dict[str, Any]:
        """Enregistre une image (OME-TIFF) ou un tableau (CSV) dans le dossier de travail."""
        from fuji.io import save_image

        item = self.item(name)
        if isinstance(item, (Points, Rois)):
            raise ControlError("Seuls les images, masques et tableaux s'enregistrent")
        suffix = ".csv" if isinstance(item, Table) else ".ome.tif"
        path = self._path(filename or f"{name}{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(item, Table):
            item.to_csv(path)
        else:
            save_image(
                path,
                item.data,
                pixel_size_um=item.pixel_size_um,
                metadata=dict(getattr(item, "metadata", {})),
            )
        self._log(f"{name} enregistré dans {path}")
        return {"fichier": str(path)}

    def table(self, name: str, max_rows: int = 50) -> dict[str, Any]:
        """Contenu d'un tableau de résultats (les premières lignes, et des statistiques par colonne)."""
        item = self.item(name)
        if not isinstance(item, Table):
            raise ControlError(f"{name!r} n'est pas un tableau")
        df = item.data
        numeric = df.select_dtypes("number")
        stats = {
            str(c): {
                "moyenne": float(numeric[c].mean()),
                "ecart_type": float(numeric[c].std()) if len(df) > 1 else 0.0,
                "min": float(numeric[c].min()),
                "max": float(numeric[c].max()),
            }
            for c in numeric.columns
            if len(df)
        }
        return _plain(
            {
                "nom": name,
                "lignes": len(df),
                "colonnes": [str(c) for c in df.columns],
                "unites": item.units,
                "resume": item.summary,
                "statistiques": stats,
                "donnees": df.head(max_rows).round(4).to_dict(orient="records"),
                "tronque": len(df) > max_rows,
            }
        )

    def create_roi(self, kind: str, coordinates: list[float], name: str | None = None) -> dict[str, Any]:
        """Crée une région d'intérêt en pixels (x vers la droite, y vers le bas, comme ImageJ).

        ``rectangle`` / ``ellipse`` : [x, y, largeur, hauteur] ; ``line`` : [x1, y1, x2, y2] ;
        ``polygon`` : [x1, y1, x2, y2, x3, y3, ...]."""
        c = [float(v) for v in coordinates]
        try:
            if kind == "rectangle":
                rois = Rois.rectangle(*c)
            elif kind == "ellipse":
                rois = Rois.oval(*c)
            elif kind == "line":
                rois = Rois.line(*c)
            elif kind == "polygon" and len(c) >= 6 and len(c) % 2 == 0:
                rois = Rois.polygon(c[0::2], c[1::2])
            else:
                raise TypeError
        except TypeError:
            raise ControlError(
                "ROI invalide. rectangle/ellipse : [x, y, largeur, hauteur] ; line : [x1, y1, x2, y2] ; "
                "polygon : au moins trois points [x1, y1, x2, y2, ...]"
            ) from None
        return self.describe_item(self._store(rois, "roi", name))

    def create_points(self, coordinates: list[list[float]], name: str | None = None) -> dict[str, Any]:
        """Crée un ensemble de points en pixels : [[x, y], ...]."""
        try:
            xy = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
        except ValueError:
            raise ControlError("Points invalides : attendu [[x, y], ...]") from None
        return self.describe_item(self._store(Points(xy[:, ::-1].copy()), "points", name))

    # ------------------------------------------------------------------ moteurs
    def _stage(self) -> Any:
        stage = self.microscope.xy_stage
        if stage is None or stage.manual:
            raise ControlError("Ce microscope n'a pas de platine motorisée")
        return stage

    def _focus(self) -> Any:
        focus = self.microscope.focus
        if focus is None or focus.manual:
            raise ControlError("Ce microscope n'a pas de mise au point motorisée")
        return focus

    def move_stage(self, x_um: float, y_um: float, relative: bool = False) -> dict[str, Any]:
        """Déplace la platine (µm). ``relative`` : déplacement depuis la position actuelle."""
        self._require_motion()
        stage = self._stage()
        x0, y0 = stage.get_position()
        x, y = (x0 + x_um, y0 + y_um) if relative else (float(x_um), float(y_um))
        if max(abs(x - x0), abs(y - y0)) > self.policy.max_xy_step_um:
            raise NotAllowed(
                f"Déplacement trop grand ({max(abs(x - x0), abs(y - y0)):.0f} µm) : au plus "
                f"{self.policy.max_xy_step_um:g} µm par commande. Procédez par étapes."
            )
        with self._lock:
            try:
                stage.move_to(x, y)
            except ValueError as exc:  # hors course
                raise ControlError(str(exc)) from exc
        x1, y1 = stage.get_position()
        self._log(f"platine → X {x1:.1f}, Y {y1:.1f} µm")
        return {"x_um": _round(x1, 2), "y_um": _round(y1, 2)}

    def move_focus(self, z_um: float, relative: bool = False) -> dict[str, Any]:
        """Déplace la mise au point (µm). Positif = selon l'axe Z du profil."""
        self._require_motion()
        focus = self._focus()
        z0 = focus.get_position()
        z = z0 + z_um if relative else float(z_um)
        if abs(z - z0) > self.policy.max_z_step_um:
            raise NotAllowed(
                f"Déplacement trop grand ({abs(z - z0):.1f} µm) : au plus {self.policy.max_z_step_um:g} µm "
                "par commande, pour protéger l'objectif et la lame."
            )
        with self._lock:
            try:
                focus.move_to(z)
            except ValueError as exc:
                raise ControlError(str(exc)) from exc
        self._log(f"mise au point → Z {focus.get_position():.2f} µm")
        return {"z_um": _round(focus.get_position(), 3)}

    def stop(self) -> dict[str, Any]:
        """Arrêt d'urgence : interrompt les moteurs et l'acquisition en cours. Toujours autorisé."""
        engine = self._engine
        if engine is not None and engine.is_running:
            engine.cancel()
        self.microscope.stop_motion()
        self._log("ARRÊT D'URGENCE")
        return {"arrete": True}

    def home(self) -> dict[str, Any]:
        """Prise d'origine de la platine sur ses fins de course."""
        self._require_motion()
        stage = self._stage()
        with self._lock:
            try:
                stage.home()
            except NotImplementedError as exc:
                raise ControlError(str(exc)) from exc
        x, y = stage.get_position()
        return {"x_um": _round(x, 2), "y_um": _round(y, 2)}

    def set_origin(self, axis: str = "xy") -> dict[str, Any]:
        """Déclare la position actuelle comme zéro (« xy » : platine, « z » : mise au point)."""
        self._require_motion()
        device = self._stage() if axis == "xy" else self._focus() if axis == "z" else None
        if device is None:
            raise ControlError("axis vaut « xy » ou « z »")
        try:
            device.set_origin()
        except NotImplementedError as exc:
            raise ControlError(str(exc)) from exc
        self._log(f"origine redéfinie ({axis})")
        return {"origine": axis}

    def center_on(self, x_px: float, y_px: float, image: str | None = None) -> dict[str, Any]:
        """Amène au centre du champ le point (x, y) d'une image (pixels, origine en haut à gauche).

        ``image`` : image en mémoire où le point a été repéré (par défaut : la taille du capteur).
        La platine ne doit pas avoir bougé depuis cette image."""
        self._require_motion()
        from fuji.core.stage import offset_to_stage

        scope = self.microscope
        shape = self.item(image).data.shape if image else scope.require_camera().sensor_shape()
        h, w = shape[:2]
        try:
            dx, dy = offset_to_stage(scope, x_px - (w - 1) / 2.0, y_px - (h - 1) / 2.0)
        except RuntimeError as exc:
            raise ControlError(str(exc)) from exc
        return self.move_stage(dx, dy, relative=True)

    def autofocus(self, range_um: float = 20.0, step_um: float = 2.0) -> dict[str, Any]:
        """Cherche la meilleure mise au point autour de la position actuelle."""
        self._require_motion()
        from fuji.assistants.focus import FocusNotFound, autofocus

        self._focus()
        if range_um / 2 > self.policy.max_z_step_um:
            raise NotAllowed(f"Plage trop grande : au plus {2 * self.policy.max_z_step_um:g} µm")
        with self._lock:
            try:
                z = autofocus(self.microscope, range_um=range_um, step_um=step_um)
            except FocusNotFound as exc:
                raise ControlError(str(exc)) from exc
            except ValueError as exc:
                raise ControlError(str(exc)) from exc
        self._log(f"autofocus → Z {z:.2f} µm")
        return {"z_um": _round(z, 3)}

    # --------------------------------------------------------------- assistants
    def focus_score(self) -> dict[str, Any]:
        """Netteté de l'image actuelle (plus grand = plus net). Sert à comparer deux réglages."""
        from fuji.assistants.focus import sharpness

        with self._lock:
            frame = self.microscope.snap()
        return {"nettete": float(sharpness(frame.data)), "z_um": _round(frame.metadata.z_um, 3)}

    def check_illumination(self) -> dict[str, Any]:
        """Uniformité de l'éclairage (sur une zone vide de la lame)."""
        from fuji.assistants import illumination_uniformity

        with self._lock:
            frame = self.microscope.snap()
        rep = illumination_uniformity(frame.data)
        return {
            "rapport_min_max": _round(rep.min_max_ratio),
            "rapport_coins_centre": _round(rep.corner_center_ratio),
            "pente_pourcent": _round(rep.tilt_percent, 1),
            "diagnostic": rep.message,
        }

    def check_field_diaphragm(self) -> dict[str, Any]:
        """Analyse du diaphragme de champ pour le réglage de Köhler (diaphragme fermé en partie)."""
        from fuji.assistants import analyze_field_diaphragm

        with self._lock:
            frame = self.microscope.snap()
        rep = analyze_field_diaphragm(frame.data)
        return _plain(
            {
                "diaphragme_visible": rep.visible,
                "entierement_dans_le_champ": rep.fully_inside,
                "fraction_eclairee": _round(rep.coverage),
                "decentrage_fraction": _round(rep.offset_fraction),
                "decentrage_px_x_y": None if rep.offset_px is None else [rep.offset_px[1], rep.offset_px[0]],
                "nettete_du_bord_px": _round(rep.edge_width_px, 1),
                "conseil": (
                    "Fermez le diaphragme de champ jusqu'à voir son bord dans l'image."
                    if not rep.visible
                    else "Bord net et centré = condenseur réglé. Bord flou : réglez la hauteur du "
                    "condenseur ; décentrage : vis de centrage du condenseur."
                ),
            }
        )

    def calibrate_pixel_size(self, spacing_um: float = 10.0, save: bool = False) -> dict[str, Any]:
        """Mesure la taille du pixel sur un micromètre objet (graduations espacées de ``spacing_um``)."""
        from fuji.assistants import calibrate_micrometer

        scope = self.microscope
        with self._lock:
            frame = scope.snap()
        try:
            cal = calibrate_micrometer(frame.data, spacing_um)
        except ValueError as exc:
            raise ControlError(str(exc)) from exc
        out = {
            "taille_pixel_um": _round(cal.pixel_size_um, 5),
            "traits": cal.n_lines,
            "incertitude_pourcent": _round(cal.relative_uncertainty * 100, 3),
            "fiable": cal.reliable,
            "valeur_actuelle_um": _round(scope.pixel_size_um(), 5),
            "enregistre": False,
        }
        if save:
            self._require_profile_change()
            if not cal.reliable:
                raise ControlError(
                    "Mesure peu fiable (trop peu de graduations nettes) : calibration non enregistrée"
                )
            scope.set_pixel_calibration(cal.pixel_size_um)
            out["enregistre"] = True
            out["fichier"] = self._save_profile()
        return out

    def _require_profile_change(self) -> None:
        self._require_write("modifier le profil")
        if not self.policy.allow_profile_changes:
            raise NotAllowed(
                "L'enregistrement de calibrations dans le profil n'est pas autorisé pour cette session."
            )

    def _save_profile(self) -> str | None:
        try:
            return str(self.microscope.save_profile())
        except Exception as exc:
            log.warning("Profil non enregistré : %s", exc)
            return None

    def calibrate_stage(self, save: bool = False) -> dict[str, Any]:
        """Mesure l'orientation de la platine par rapport à la caméra (deux petits pas)."""
        self._require_motion()
        from fuji.core.stage import calibrate_stage, save_stage_calibration

        self._stage()
        with self._lock:
            try:
                cal = calibrate_stage(self.microscope)
            except RuntimeError as exc:
                raise ControlError(str(exc)) from exc
        out = {
            "rotation_degres": _round(cal.rotation_deg, 2),
            "miroir": cal.mirrored,
            "taille_pixel_selon_platine_um": _round(cal.pixel_size_um, 5),
            "ecart_avec_pixel_calibre_pourcent": None
            if cal.scale_error is None
            else _round(cal.scale_error * 100, 2),
            "resume": cal.summary,
            "enregistre": False,
        }
        if save:
            self._require_profile_change()
            save_stage_calibration(self.microscope, cal)
            out["enregistre"] = True
            out["fichier"] = self._save_profile()
        return out

    def measure_backlash(self, axis: str = "x") -> dict[str, Any]:
        """Jeu mécanique d'un axe de la platine (µm)."""
        self._require_motion()
        from fuji.core.stage import measure_backlash

        self._stage()
        with self._lock:
            try:
                value = measure_backlash(self.microscope, axis)
            except (RuntimeError, ValueError) as exc:
                raise ControlError(str(exc)) from exc
        return {"axe": axis, "jeu_um": _round(value, 2)}

    # -------------------------------------------------------------- acquisition
    def acquire(
        self,
        name: str = "acquisition",
        time_points: int = 1,
        interval_s: float = 0.0,
        z_range_um: float = 0.0,
        z_step_um: float = 1.0,
        channels: list[str] | None = None,
        positions: list[list[float]] | None = None,
        mosaic: list[int] | None = None,
        mosaic_overlap: float = 0.15,
        autofocus: bool = False,
        wait_s: float = 30.0,
        manual_done: bool = False,
    ) -> dict[str, Any]:
        """Lance une acquisition enregistrée en OME-TIFF dans le dossier de travail.

        ``positions`` : [[x, y], ...] en µm ; ``mosaic`` : [colonnes, rangées] autour de la
        position actuelle (les tuiles sont assemblées à la fin). Attend la fin jusqu'à
        ``wait_s`` secondes ; au-delà l'acquisition continue : suivre avec
        :meth:`acquisition_status`."""
        import useq

        from fuji.io import OMETiffWriter

        self._require_write("lancer une acquisition")
        scope = self.microscope
        engine = self._get_engine() or self._engine or AcquisitionEngine(scope)
        if engine.microscope is not scope:
            engine = AcquisitionEngine(scope)
        if engine.is_running:
            raise ControlError("Une acquisition est déjà en cours (acquisition_status, cancel_acquisition)")
        kwargs: dict[str, Any] = {"axis_order": "tpcz"}
        if channels:
            unknown = [c for c in channels if c not in scope.channel_names]
            if unknown:
                raise ControlError(f"Canaux inconnus {unknown}. Canaux : {scope.channel_names}")
            kwargs["channels"] = list(channels)
        if time_points > 1:
            kwargs["time_plan"] = {"interval": float(interval_s), "loops": int(time_points)}
        if z_range_um > 0:
            if scope.focus is None:
                raise ControlError("Pile Z impossible : aucune mise au point déclarée")
            kwargs["z_plan"] = {"range": float(z_range_um), "step": float(z_step_um)}
        plan = None
        if mosaic:
            self._require_motion()
            from fuji.core.stage import plan_mosaic

            try:
                plan = plan_mosaic(scope, int(mosaic[0]), int(mosaic[1]), float(mosaic_overlap))
            except (RuntimeError, ValueError, IndexError) as exc:
                raise ControlError(f"Mosaïque impossible : {exc}") from exc
            kwargs["stage_positions"] = [{"x": x, "y": y} for x, y in plan.positions]
        elif positions:
            self._require_motion()
            self._stage()
            kwargs["stage_positions"] = [{"x": float(p[0]), "y": float(p[1])} for p in positions]
        if len(kwargs) == 1:
            kwargs["time_plan"] = {"interval": 0, "loops": 1}
        sequence = useq.MDASequence(**kwargs)
        folder = self._path("acquisitions")
        estimate = plan_acquisition(scope, sequence, str(folder))
        if estimate.n_frames > self.policy.max_frames:
            raise NotAllowed(
                f"{estimate.n_frames} images demandées : au plus {self.policy.max_frames} par acquisition"
            )
        if estimate.manual_actions and not manual_done and self._own_manual:
            raise ManualActionRequired(
                f"Cette acquisition demandera {estimate.manual_actions} manipulation(s) manuelle(s) "
                "(changements de canal, de mise au point ou de position à la main). Prévenez l'utilisateur ; "
                "pour lancer quand même, rappelez avec manual_done=true : les consignes seront alors "
                "considérées comme faites sans attendre. Préférez une séquence sans élément manuel."
            )
        writer = OMETiffWriter(folder, name)
        tiles: dict[int, dict[int, Any]] = {}

        def collect(event: Any, frame: Any) -> None:
            index = {str(getattr(k, "value", k)): int(v) for k, v in event.index.items()}
            self._acquisition.update(faites=self._acquisition.get("faites", 0) + 1)
            if plan is not None and index.get("t", 0) == 0 and index.get("z", 0) == 0:
                tiles.setdefault(index.get("c", 0), {})[index.get("p", 0)] = frame

        def finished(summary: AcquisitionSummary) -> None:
            engine.frame_acquired.disconnect(collect)
            engine.finished.disconnect(finished)
            self._summary = summary
            result = {
                "etat": "annulée" if summary.cancelled else "erreur" if summary.error else "terminée",
                "faites": summary.acquired,
                "total": summary.total,
                "duree_s": _round(summary.duration_s, 1),
                "erreur": summary.error,
                "fichiers": [str(p) for p in summary.files],
            }
            if plan is not None and not summary.error and not summary.cancelled:
                result["mosaiques"] = self._assemble(plan, tiles)
            self._acquisition = result
            self._log(f"acquisition {result['etat']} : {summary.acquired}/{summary.total} images")

        previous = scope.manual_handler
        if self._own_manual:
            scope.manual_handler = (lambda action: True) if manual_done else self._manual_handler
        engine.frame_acquired.connect(collect)
        engine.finished.connect(finished)
        if self._own_manual:
            engine.finished.connect(lambda *_: setattr(scope, "manual_handler", previous), max_args=1)
        self._engine = engine
        self._acquisition = {
            "etat": "en cours",
            "faites": 0,
            "total": estimate.n_frames,
            "duree_prevue_s": _round(estimate.duration_s, 1),
            "avertissements": list(estimate.warnings),
        }
        self._log(f"acquisition « {name} » : {estimate.n_frames} images")
        engine.start(sequence, writer=writer, autofocus=AutofocusPlan() if autofocus else None)
        return self.acquisition_status(wait_s)

    def _assemble(self, plan: Any, tiles: dict[int, dict[int, Any]]) -> list[str]:
        from fuji.core.stage import refine_offsets, stitch

        names = []
        for _channel, frames in sorted(tiles.items()):
            if len(frames) != len(plan.positions):
                continue
            ordered = [frames[i] for i in range(len(plan.positions))]
            data = [f.data for f in ordered]
            md = ordered[0].metadata
            image = Image(
                stitch(data, refine_offsets(data, plan.offsets_px)),
                md.pixel_size_um,
                "mosaique",
                md.to_dict(),
            )
            names.append(self._store(image, "mosaique"))
        return names

    def acquisition_status(self, wait_s: float = 0.0) -> dict[str, Any]:
        """État de l'acquisition ; attend sa fin jusqu'à ``wait_s`` secondes."""
        deadline = time.monotonic() + max(0.0, min(float(wait_s), 300.0))
        while self._acquisition.get("etat") == "en cours" and time.monotonic() < deadline:
            time.sleep(0.05)
        return _plain(dict(self._acquisition))

    def cancel_acquisition(self) -> dict[str, Any]:
        engine = self._engine
        if engine is None or not engine.is_running:
            return {"etat": self._acquisition.get("etat", "aucune")}
        engine.cancel()
        return self.acquisition_status(10.0)

    # ------------------------------------------------------------------ analyse
    def operations(self, search: str = "", category: str = "") -> list[dict[str, Any]]:
        """Catalogue des opérations d'analyse (filtré par mot-clé ou catégorie)."""
        out = []
        needle = search.strip().lower()
        for op in list_operations():
            text = f"{op.name} {op.category} {op.description} {op.imagej or ''}".lower()
            if (needle and needle not in text) or (category and op.category.lower() != category.lower()):
                continue
            out.append(
                {
                    "nom": op.name,
                    "categorie": op.category,
                    "resume": op.description.split("\n")[0],
                    "imagej": op.imagej,
                    "entrees": [s.name for s in op.inputs],
                }
            )
        return out

    def describe_operation(self, name: str) -> dict[str, Any]:
        """Entrées et réglages d'une opération (noms exacts à utiliser dans run_operation)."""
        op = self._operation(name)
        kinds = {Image: "image", Labels: "masque", Rois: "roi", Points: "points"}
        return _plain(
            {
                "nom": op.name,
                "categorie": op.category,
                "description": op.description,
                "imagej": op.imagej,
                "pile_entiere": op.stack,
                "entrees": [
                    {"nom": s.name, "type": kinds[s.kind], "facultative": s.optional} for s in op.inputs
                ],
                "reglages": [
                    {
                        "nom": p.name,
                        "type": getattr(p.type, "__name__", str(p.type)),
                        "defaut": p.default,
                        "choix": p.meta.choices,
                        "min": p.meta.min,
                        "max": p.meta.max,
                        "unite": p.meta.unit,
                        "description": p.meta.description,
                    }
                    for p in op.params
                ],
            }
        )

    def _operation(self, name: str) -> Any:
        try:
            return get_operation(name)
        except KeyError:
            import difflib

            names = [o.name for o in list_operations()]
            close = [n for n in names if name.lower() in n.lower()]
            close += [n for n in difflib.get_close_matches(name, names, n=4, cutoff=0.5) if n not in close]
            hint = f" Vouliez-vous dire : {close[:5]} ?" if close else " Voir list_operations."
            raise ControlError(f"Opération inconnue {name!r}.{hint}") from None

    def run_operation(
        self, name: str, inputs: dict[str, str] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Exécute une opération d'analyse sur des éléments en mémoire.

        ``inputs`` : nom de l'entrée → nom de l'élément (ex. {"image": "capture_1"}).
        ``params`` : réglages (voir describe_operation). Les résultats sont gardés en
        mémoire ; leurs noms et leur résumé sont renvoyés."""
        op = self._operation(name)
        inputs, params = dict(inputs or {}), dict(params or {})
        kinds = {Image: "une image", Labels: "un masque", Rois: "des ROI", Points: "des points"}
        args = []
        for spec in op.inputs:
            ref = inputs.pop(spec.name, None)
            if ref is None:
                if not spec.optional:
                    raise ControlError(f"{op.name} : il manque l'entrée {spec.name!r} ({kinds[spec.kind]})")
                args.append(None)
                continue
            item = self.item(ref)
            if spec.kind is Labels and isinstance(item, Image) and item.data.ndim == 2:
                item = Labels(item.data, item.pixel_size_um, item.name)  # image binaire utilisée comme masque
            if not isinstance(item, spec.kind):
                raise ControlError(
                    f"{op.name} : l'entrée {spec.name!r} attend {kinds[spec.kind]}, pas {ref!r}"
                )
            args.append(item)
        if inputs:
            raise ControlError(
                f"{op.name} : entrées inconnues {sorted(inputs)} ; attendues {[s.name for s in op.inputs]}"
            )
        known = {p.name: p for p in op.params}
        unknown = set(params) - set(known)
        if unknown:
            raise ControlError(f"{op.name} : réglages inconnus {sorted(unknown)} ; attendus {sorted(known)}")
        for key, value in params.items():
            choices = known[key].meta.choices
            if choices and str(value) not in choices:
                raise ControlError(f"{op.name} : {key} = {value!r} ; choix possibles : {list(choices)}")
        if op.category == "Fiji / ImageJ" and self.policy.read_only:
            raise NotAllowed("Les macros ImageJ ne sont pas autorisées en lecture seule")
        try:
            result = op.func(*args, **params)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise ControlError(f"{op.name} : {exc}") from exc
        stem = "".join(ch if ch.isalnum() else "_" for ch in op.name.lower()).strip("_")[:24]
        outputs = []
        for item in result if isinstance(result, tuple) else (result,):
            if item is not None:
                outputs.append(self.describe_item(self._store(item, stem)))
        self._log(f"{op.name} → {[o['nom'] for o in outputs]}")
        return {"operation": op.name, "resultats": outputs}


_META_KEYS = ("objective", "channel", "exposure_ms", "x_um", "y_um", "z_um", "timestamp", "gain")
