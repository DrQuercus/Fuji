"""Caméras ToupTek (et marques OEM basées sur le même SDK) via le SDK « toupcam ».

Le SDK est propriétaire : il n'est pas inclus dans Fuji. Pour l'installer :

1. télécharger « ToupTek SDK » (toupcamsdk.zip) sur le site de ToupTek ;
2. lancer ``fuji touptek install chemin/vers/toupcamsdk.zip`` : Fuji copie
   ``toupcam.py`` et la bibliothèque de ta plateforme (macOS : libtoupcam.dylib)
   dans ``~/.fuji/touptek`` ;
3. vérifier avec ``fuji touptek info``.

Fonctionnement :

- **live** : la caméra tourne en mode vidéo, chaque image est publiée ;
- **capture / acquisition** : déclenchement logiciel (une image par demande, avec le
  temps d'exposition exact) si la caméra le permet, sinon attente d'une image fraîche ;
- **12 bits** : ``bit_depth: 12`` active la haute profondeur (RGB48 / niveaux de gris
  16 bits) pour des mesures d'intensité fiables.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from fuji.core.devices import Camera
from fuji.core.profile import DeviceSpec
from fuji.core.properties import PropertyInfo, PropertyType

from . import DriverContext, OptionSpec, register_driver

log = logging.getLogger(__name__)


class ToupTekSDKNotFound(ImportError):
    pass


# ------------------------------------------------------------------- SDK
def default_sdk_dir() -> Path:
    home = Path(os.environ.get("FUJI_HOME", Path.home() / ".fuji")).expanduser()
    return home / "touptek"


def load_toupcam(sdk_path: str | Path | None = None) -> ModuleType:
    """Importe ``toupcam.py`` depuis le dossier du SDK."""
    path = Path(sdk_path or os.environ.get("FUJI_TOUPCAM_SDK") or default_sdk_dir()).expanduser()
    module_file = path / "toupcam.py"
    if not module_file.exists():
        raise ToupTekSDKNotFound(
            f"SDK ToupTek introuvable dans {path}.\n"
            "Téléchargez le SDK sur le site de ToupTek, puis : Préférences → Caméras ToupTek → "
            "« Installer le SDK… » (ou, dans un terminal : fuji touptek install /chemin/vers/toupcamsdk.zip)"
        )
    if "toupcam" in sys.modules and getattr(sys.modules["toupcam"], "__file__", None) == str(module_file):
        return sys.modules["toupcam"]
    spec = importlib.util.spec_from_file_location("toupcam", module_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["toupcam"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("toupcam", None)  # ne pas garder un module à moitié chargé
        raise
    return module


def _platform_library() -> tuple[str, tuple[str, ...]]:
    """Nom de la bibliothèque et dossiers d'architecture à privilégier."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch: tuple[str, ...] = ("x64", "x86_64", "amd64")
    elif machine in ("arm64", "aarch64"):
        arch = ("arm64", "aarch64", "armv8")
    elif machine.startswith("armv7") or machine == "armhf":
        arch = ("armhf", "armel", "armv7")
    else:
        arch = ("x86", "i386")
    if sys.platform == "darwin":
        return "libtoupcam.dylib", arch
    if sys.platform.startswith("win"):
        return "toupcam.dll", arch
    return "libtoupcam.so", arch


def _library_rank(path: Path, arch: tuple[str, ...]) -> tuple[bool, bool, bool, str]:
    """Classe les bibliothèques du SDK : la bonne plateforme et la bonne architecture d'abord.

    Le SDK contient aussi des bibliothèques Android portant le même nom que celles de
    Linux : elles ne doivent jamais être choisies.
    """
    parts = [x.lower() for x in path.parts]
    wanted = {"darwin": ("mac", "macos", "osx"), "win32": ("win", "windows")}.get(sys.platform, ("linux",))
    foreign = any(x in parts for x in ("android", "ios", "harmony", "uwp"))
    return (foreign, not any(x in parts for x in wanted), not any(a in parts for a in arch), str(path))


def install_sdk(source: str | Path, dest: str | Path | None = None) -> Path:
    """Copie ``toupcam.py`` + la bibliothèque native depuis le SDK (dossier ou .zip)."""
    source = Path(source).expanduser()
    dest = Path(dest).expanduser() if dest else default_sdk_dir()
    with tempfile.TemporaryDirectory() as tmp:
        root = source
        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as zf:
                zf.extractall(tmp)
            root = Path(tmp)
        if not root.is_dir():
            raise FileNotFoundError(f"{source} n'est ni un dossier ni un .zip")
        modules = sorted(root.rglob("toupcam.py"), key=lambda p: ("python" not in str(p), str(p)))
        if not modules:
            raise FileNotFoundError(f"toupcam.py introuvable dans {source}")
        libname, arch = _platform_library()
        libs = list(root.rglob(libname))
        if not libs:
            raise FileNotFoundError(f"{libname} introuvable dans {source} (plateforme {sys.platform})")
        libs.sort(key=lambda p: _library_rank(p.relative_to(root), arch))
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(modules[0], dest / "toupcam.py")
        shutil.copy2(libs[0], dest / libname)
    return dest


# ------------------------------------------------------------------ caméra
class ToupTekCamera(Camera):
    supports_gain = True
    supports_auto_exposure = True
    gain_unit = "%"

    def __init__(
        self,
        name: str,
        *,
        sdk_path: str | None = None,
        camera_id: str | None = None,
        resolution_index: int = 0,
        bit_depth: int = 8,
        toupcam_module: ModuleType | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._sdk_path = sdk_path
        self._camera_id = camera_id
        self._res_index = int(resolution_index)
        self._want_bits = int(bit_depth)
        self._tc = toupcam_module
        self._hcam: Any = None
        self._info: Any = None
        self._mono = False
        self._high = False
        self._bits = 8
        self._pull_bits = 24
        self._size = (0, 0)  # (h, w)
        self._buf: bytes = b""
        self._buf_lock = threading.Lock()
        self._live_flag = False
        self._has_trigger = False
        self._trigger_failed = False
        self._last_error: str | None = None
        #: Erreur définitive (caméra débranchée) : toute capture échoue aussitôt.
        self._fatal: str | None = None
        self.supports_auto_white_balance = True

    # -- utilitaires SDK ---------------------------------------------------------
    def _c(self, name: str, default: int | None = None) -> int:
        value = getattr(self._tc, name, default)
        if value is None:
            raise AttributeError(f"Constante {name} absente du SDK ToupTek")
        return value

    @property
    def model_name(self) -> str:
        return getattr(self._info, "displayname", "ToupTek") if self._info else "ToupTek"

    # -- cycle de vie ---------------------------------------------------------------
    def initialize(self) -> None:
        if self._tc is None:
            self._tc = load_toupcam(self._sdk_path)
        tc = self._tc
        found = tc.Toupcam.EnumV2()
        if not found:
            raise RuntimeError(
                "Aucune caméra ToupTek détectée. Vérifie le câble USB 3 et qu'aucun autre "
                "logiciel (ToupView...) n'utilise la caméra."
            )
        info = found[0]
        if self._camera_id:
            matches = [
                d
                for d in found
                if self._camera_id in (d.id, d.displayname) or self._camera_id in d.displayname
            ]
            if not matches:
                names = [d.displayname for d in found]
                raise RuntimeError(f"Caméra {self._camera_id!r} introuvable ; détectées : {names}")
            info = matches[0]
        self._info = info
        self._fatal = None
        self._hcam = tc.Toupcam.Open(info.id)
        if not self._hcam:
            raise RuntimeError(
                f"Impossible d'ouvrir {info.displayname} : la caméra est peut-être utilisée "
                "par un autre logiciel (ToupView...)."
            )
        try:
            flag = info.model.flag
            self._mono = bool(flag & self._c("TOUPCAM_FLAG_MONO", 0x10))
            trig = self._c("TOUPCAM_FLAG_TRIGGER_SOFTWARE", 0x00080000)
            self._has_trigger = bool(flag & trig)
            if self._mono:
                self.supports_auto_white_balance = False
            self._configure()
            self._start_stream()
            self._register_props()
        except Exception:
            self._close_handle()  # sinon la caméra resterait « occupée » jusqu'au redémarrage
            raise
        super().initialize()

    def _start_stream(self) -> None:
        self._hcam.StartPullModeWithCallback(ToupTekCamera._event_callback, self)

    def _close_handle(self) -> None:
        handle, self._hcam = self._hcam, None
        if handle is not None:
            try:
                handle.Close()
            except Exception:  # pragma: no cover - dépend du matériel
                log.exception("Erreur à la fermeture de la caméra ToupTek")

    def _configure(self) -> None:
        """Résolution, profondeur de bits, format de pixel (caméra arrêtée)."""
        h, tc = self._hcam, self._tc
        model = self._info.model
        n_res = max(1, int(getattr(model, "preview", 1)))
        self._res_index = min(max(self._res_index, 0), n_res - 1)
        h.put_eSize(self._res_index)
        raw_flags = 0
        for flag_name in (
            "TOUPCAM_FLAG_RAW10",
            "TOUPCAM_FLAG_RAW12",
            "TOUPCAM_FLAG_RAW14",
            "TOUPCAM_FLAG_RAW16",
        ):
            raw_flags |= getattr(tc, flag_name, 0)
        self._high = self._want_bits > 8 and bool(model.flag & raw_flags)
        h.put_Option(self._c("TOUPCAM_OPTION_BITDEPTH"), 1 if self._high else 0)
        if self._high:
            try:
                self._bits = int(h.MaxBitDepth())
            except Exception:
                self._bits = 12
        else:
            self._bits = 8
        # 0 = RGB24, 1 = RGB48, 3 = gris 8 bits, 4 = gris 16 bits
        rgb_mode = (4 if self._high else 3) if self._mono else (1 if self._high else 0)
        h.put_Option(self._c("TOUPCAM_OPTION_RGB"), rgb_mode)
        try:
            h.put_Option(self._c("TOUPCAM_OPTION_BYTEORDER"), 0)  # ordre RGB (pas BGR)
        except Exception:  # pragma: no cover - anciens SDK
            log.debug("TOUPCAM_OPTION_BYTEORDER non supporté")
        h.put_AutoExpoEnable(0)
        if self._has_trigger:
            h.put_Option(self._c("TOUPCAM_OPTION_TRIGGER"), 1)  # déclenchement logiciel
        w, hh = h.get_Size()
        self._size = (int(hh), int(w))
        self._pull_bits = (16 if self._high else 8) if self._mono else (48 if self._high else 24)
        stride = ((w * self._pull_bits + 31) // 32) * 4
        self._buf = bytes(stride * hh)

    def _reconfigure(self, **changes: Any) -> None:
        """Change résolution ou profondeur : la caméra doit être arrêtée entre-temps."""
        was_live = self._live_flag
        self._live_flag = False  # _configure() remet la caméra en mode déclenché
        self._hcam.Stop()
        if "resolution_index" in changes:
            self._res_index = int(changes["resolution_index"])
        if "bit_depth" in changes:
            self._want_bits = int(changes["bit_depth"])
        self._configure()
        self._start_stream()
        if was_live:
            self.start_live()

    def shutdown(self) -> None:
        self._live_flag = False
        self._close_handle()
        super().shutdown()

    # -- réception des images -----------------------------------------------------
    @staticmethod
    def _event_callback(event: int, ctx: ToupTekCamera) -> None:
        ctx._on_event(event)

    def _on_event(self, event: int) -> None:
        tc = self._tc
        hcam = self._hcam
        if hcam is None:  # événement tardif pendant la fermeture
            return
        if event == self._c("TOUPCAM_EVENT_IMAGE", 0x0004):
            try:
                with self._buf_lock:
                    if hasattr(hcam, "PullImageV3"):
                        hcam.PullImageV3(self._buf, 0, self._pull_bits, 0, None)
                    else:  # anciens SDK
                        hcam.PullImageV2(self._buf, self._pull_bits, None)
                    img = self._buffer_to_array()
            except Exception as exc:  # pragma: no cover - dépend du matériel
                self._last_error = f"lecture d'image : {exc}"
                log.warning("ToupTek : %s", self._last_error)
                return
            if self._high and self._bits < 16 and int(img[::16, ::16].max()) >> self._bits:
                # certains modèles étalent leurs valeurs sur 16 bits au lieu de 0..2^n − 1
                self._bits = 16
            self._publish(img)
        elif event == getattr(tc, "TOUPCAM_EVENT_TRIGGERFAIL", 0x0007):
            self._trigger_failed = True
        elif event in (
            getattr(tc, "TOUPCAM_EVENT_ERROR", 0x0080),
            getattr(tc, "TOUPCAM_EVENT_DISCONNECTED", 0x0081),
        ):
            self._fatal = "caméra débranchée ou en erreur : vérifiez le câble puis reconnectez"
            self._last_error = self._fatal
            log.error("ToupTek : %s (événement 0x%x)", self._fatal, event)
            with self._frame_cond:
                self._frame_cond.notify_all()

    def _check_alive(self) -> None:
        if self._fatal:
            raise RuntimeError(f"{self.label} : {self._fatal}")

    def _buffer_to_array(self) -> np.ndarray:
        h, w = self._size
        bpp = self._pull_bits // 8
        stride = ((w * self._pull_bits + 31) // 32) * 4
        raw = np.frombuffer(self._buf, dtype=np.uint8, count=stride * h).reshape(h, stride)
        # copie obligatoire : le tampon est réécrit par l'image suivante (sans le bourrage
        # de fin de ligne quand la largeur n'est pas un multiple de 4 octets)
        arr = raw[:, : w * bpp].copy()
        if self._pull_bits in (16, 48):
            arr = arr.view("<u2")
        return arr.reshape(h, w) if self._mono else arr.reshape(h, w, 3)

    # -- interface Camera ---------------------------------------------------------
    def sensor_shape(self) -> tuple[int, int]:
        return self._size

    @property
    def is_color(self) -> bool:
        return not self._mono

    @property
    def bit_depth(self) -> int:
        return self._bits

    @property
    def sensor_pixel_size_um(self) -> float | None:
        px = getattr(self._info.model, "xpixsz", 0) if self._info else 0
        return float(px) if px else None

    @property
    def binning(self) -> float:
        """En résolution réduite, chaque pixel image couvre plusieurs photosites : la
        taille du pixel (estimée ou calibrée en pleine résolution) est multipliée d'autant."""
        res = getattr(self._info.model, "res", None) if self._info else None
        i = self._res_index
        if not res or not 0 < i < len(res) or not res[i].width:
            return 1
        ratio = res[0].width / res[i].width
        return int(round(ratio)) if abs(ratio - round(ratio)) < 0.01 else ratio

    def get_exposure(self) -> float:
        return self._hcam.get_ExpoTime() / 1000.0

    def set_exposure(self, exposure_ms: float) -> None:
        lo, hi = self.exposure_limits()
        self._hcam.put_ExpoTime(int(round(float(np.clip(exposure_ms, lo, hi)) * 1000)))

    def exposure_limits(self) -> tuple[float, float]:
        lo, hi, _ = self._hcam.get_ExpTimeRange()
        return lo / 1000.0, hi / 1000.0

    def get_gain(self) -> float:
        return float(self._hcam.get_ExpoAGain())

    def set_gain(self, gain: float) -> None:
        lo, hi = self.gain_limits()
        self._hcam.put_ExpoAGain(int(round(float(np.clip(gain, lo, hi)))))

    def gain_limits(self) -> tuple[float, float]:
        lo, hi, _ = self._hcam.get_ExpoAGainRange()
        return float(lo), float(hi)

    def auto_white_balance(self) -> None:
        if self._mono:
            return
        self._hcam.AwbOnce()

    def set_auto_exposure(self, enabled: bool) -> None:
        self._hcam.put_AutoExpoEnable(1 if enabled else 0)

    # -- capture et live ------------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self._live_flag

    def start_live(self) -> None:
        if self._live_flag:
            return
        if self._has_trigger:
            self._hcam.put_Option(self._c("TOUPCAM_OPTION_TRIGGER"), 0)  # mode vidéo
        self._live_flag = True

    def stop_live(self) -> None:
        if not self._live_flag:
            return
        self._live_flag = False
        if self._has_trigger:
            self._hcam.put_Option(self._c("TOUPCAM_OPTION_TRIGGER"), 1)
            flush = getattr(self._tc, "TOUPCAM_OPTION_FLUSH", None)
            if flush is not None:  # jeter les images vidéo encore en transit
                try:
                    self._hcam.put_Option(flush, 3)
                except Exception:  # pragma: no cover - option absente sur certains modèles
                    log.debug("TOUPCAM_OPTION_FLUSH non supportée")

    def snap(self) -> np.ndarray:
        """Renvoie une image exposée **après** la demande.

        En mode déclenché, une image est demandée à la caméra ; en live, on attend dans
        le flux. Dans les deux cas, une image arrivée moins d'un temps de pose après la
        demande a forcément commencé avant (image vidéo en transit, ancien réglage) :
        elle est écartée. Sans ce contrôle, toute une acquisition pourrait être décalée
        d'une image.
        """
        with self._lock:
            self._check_alive()
            exposure_s = self.get_exposure() / 1000.0
            timeout = exposure_s * 3 + 5.0
            triggered = self._has_trigger and not self._live_flag
            try:
                auto = bool(self._hcam.get_AutoExpoEnable())
            except Exception:  # pragma: no cover
                auto = False
            count, _ = self.latest_frame()
            t0 = time.monotonic()
            if triggered:
                self._trigger_failed = False
                self._hcam.Trigger(1)
            # avec l'exposition automatique, le temps de pose réel est inconnu : pas de contrôle
            earliest = t0 if auto else t0 + 0.9 * exposure_s
            deadline = t0 + timeout
            candidate: np.ndarray | None = None
            grace = deadline
            retried = False
            while True:
                now = time.monotonic()
                if candidate is not None and now >= grace:
                    return candidate  # rien d'autre n'est venu : c'était bien la bonne image
                if now >= deadline:
                    raise TimeoutError(f"{self.name} : aucune image reçue en {timeout:.1f} s")
                if triggered and self._trigger_failed and not retried:
                    retried, self._trigger_failed = True, False
                    self._hcam.Trigger(1)
                try:
                    count, img, arrived = self._wait_frame(count, min(deadline - now, 0.2))
                except TimeoutError:
                    continue
                if arrived >= earliest:
                    return img
                candidate, grace = img, time.monotonic() + exposure_s * 1.5 + 0.5

    def _snap(self) -> np.ndarray:  # pragma: no cover - snap() est surchargée
        return self.snap()

    def _max_bits(self) -> int:
        try:
            return int(self._hcam.MaxBitDepth())
        except Exception:
            return 12 if self._high else 8

    # -- propriétés génériques -------------------------------------------------------
    def _register_props(self) -> None:
        h, tc, model = self._hcam, self._tc, self._info.model
        img, col, adv = "Image", "Couleur", True

        res = [f"{r.width}x{r.height}" for r in getattr(model, "res", [])][: max(1, model.preview)]
        if res:
            self._add_property(
                PropertyInfo("Résolution", PropertyType.STRING, choices=tuple(res), category=img),
                lambda: res[self._res_index],
                lambda v: self._reconfigure(resolution_index=res.index(v)),
            )
        self._add_property(
            PropertyInfo(
                "Profondeur de bits",
                PropertyType.STRING,
                choices=("8", "12"),
                category=img,
                description="12 bits : meilleure dynamique (mesures)",
            ),
            lambda: "12" if self._high else "8",
            lambda v: self._reconfigure(bit_depth=int(v)),
        )
        self._add_property(
            PropertyInfo("Exposition automatique", PropertyType.BOOL, category=img),
            lambda: bool(h.get_AutoExpoEnable()),
            lambda v: h.put_AutoExpoEnable(1 if v else 0),
        )

        def optional(info: PropertyInfo, getter, setter) -> None:
            try:
                getter()
            except Exception:
                return  # non supporté par ce modèle
            self._add_property(info, getter, setter)

        optional(
            PropertyInfo("Miroir horizontal", PropertyType.BOOL, category=img),
            lambda: bool(h.get_HFlip()),
            lambda v: h.put_HFlip(1 if v else 0),
        )
        optional(
            PropertyInfo("Miroir vertical", PropertyType.BOOL, category=img),
            lambda: bool(h.get_VFlip()),
            lambda v: h.put_VFlip(1 if v else 0),
        )
        optional(
            PropertyInfo(
                "Gamma",
                PropertyType.INT,
                limits=(getattr(tc, "TOUPCAM_GAMMA_MIN", 20), getattr(tc, "TOUPCAM_GAMMA_MAX", 180)),
                category=img,
                advanced=adv,
                description="100 = réponse linéaire (recommandé pour mesurer)",
            ),
            h.get_Gamma,
            h.put_Gamma,
        )
        optional(
            PropertyInfo(
                "Contraste",
                PropertyType.INT,
                limits=(getattr(tc, "TOUPCAM_CONTRAST_MIN", -100), getattr(tc, "TOUPCAM_CONTRAST_MAX", 100)),
                category=img,
                advanced=adv,
            ),
            h.get_Contrast,
            h.put_Contrast,
        )
        optional(
            PropertyInfo(
                "Luminosité",
                PropertyType.INT,
                limits=(
                    getattr(tc, "TOUPCAM_BRIGHTNESS_MIN", -64),
                    getattr(tc, "TOUPCAM_BRIGHTNESS_MAX", 64),
                ),
                category=img,
                advanced=adv,
            ),
            h.get_Brightness,
            h.put_Brightness,
        )
        optional(
            PropertyInfo(
                "Vitesse", PropertyType.INT, limits=(0, max(0, model.maxspeed)), category=img, advanced=adv
            ),
            h.get_Speed,
            h.put_Speed,
        )
        bl = getattr(tc, "TOUPCAM_OPTION_BLACKLEVEL", None)
        if bl is not None:
            optional(
                PropertyInfo(
                    "Niveau de noir",
                    PropertyType.INT,
                    limits=(0, 31 << max(0, self._max_bits() - 8)),
                    category=img,
                    advanced=adv,
                ),
                lambda: h.get_Option(bl),
                lambda v: h.put_Option(bl, int(v)),
            )
        cg = getattr(tc, "TOUPCAM_OPTION_CG", None)
        if cg is not None and model.flag & (
            getattr(tc, "TOUPCAM_FLAG_CG", 0) | getattr(tc, "TOUPCAM_FLAG_CGHDR", 0)
        ):
            modes = ("LCG", "HCG", "HDR")
            optional(
                PropertyInfo(
                    "Gain de conversion",
                    PropertyType.STRING,
                    choices=modes,
                    category=img,
                    advanced=adv,
                    description="HCG : moins de bruit en faible lumière",
                ),
                lambda: modes[int(h.get_Option(cg))],
                lambda v: h.put_Option(cg, modes.index(v)),
            )
        if not self._mono:

            def temp_tint() -> tuple[int, int]:
                return tuple(h.get_TempTint())  # type: ignore[return-value]

            optional(
                PropertyInfo(
                    "Température de couleur",
                    PropertyType.INT,
                    limits=(getattr(tc, "TOUPCAM_TEMP_MIN", 2000), getattr(tc, "TOUPCAM_TEMP_MAX", 15000)),
                    unit="K",
                    category=col,
                ),
                lambda: temp_tint()[0],
                lambda v: h.put_TempTint(int(v), temp_tint()[1]),
            )
            optional(
                PropertyInfo(
                    "Teinte (tint)",
                    PropertyType.INT,
                    limits=(getattr(tc, "TOUPCAM_TINT_MIN", 200), getattr(tc, "TOUPCAM_TINT_MAX", 2500)),
                    category=col,
                    advanced=adv,
                ),
                lambda: temp_tint()[1],
                lambda v: h.put_TempTint(temp_tint()[0], int(v)),
            )
            optional(
                PropertyInfo(
                    "Saturation",
                    PropertyType.INT,
                    limits=(
                        getattr(tc, "TOUPCAM_SATURATION_MIN", 0),
                        getattr(tc, "TOUPCAM_SATURATION_MAX", 255),
                    ),
                    category=col,
                    advanced=adv,
                ),
                h.get_Saturation,
                h.put_Saturation,
            )

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        if self._info is not None:
            model = self._info.model
            out.update(
                model=self._info.displayname,
                id=self._info.id,
                flags=hex(model.flag),
                resolutions=[f"{r.width}x{r.height}" for r in getattr(model, "res", [])],
                software_trigger=self._has_trigger,
                max_speed=model.maxspeed,
            )
            out["resolution_scale"] = self.binning
            # les noms de ces méthodes ont changé d'une version du SDK à l'autre
            for key, names in (
                ("serial", ("SerialNumber", "get_SerialNumber")),
                ("firmware", ("FwVersion", "get_FwVersion")),
                ("hardware", ("HwVersion", "get_HwVersion")),
            ):
                for fn in names:
                    try:
                        out[key] = getattr(self._hcam, fn)()
                        break
                    except Exception:
                        continue
            try:
                out["sdk"] = self._tc.Toupcam.Version()
            except Exception:
                pass
        if self._last_error:
            out["last_error"] = self._last_error
        return out


def list_cameras(sdk_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Liste les caméras ToupTek branchées (pour ``fuji touptek info``)."""
    tc = load_toupcam(sdk_path)
    out = []
    for d in tc.Toupcam.EnumV2():
        m = d.model
        out.append(
            {
                "nom": d.displayname,
                "id": d.id,
                "flags": hex(m.flag),
                "monochrome": bool(m.flag & getattr(tc, "TOUPCAM_FLAG_MONO", 0x10)),
                "pixel_um": (m.xpixsz, m.ypixsz),
                "résolutions": [f"{r.width}x{r.height}" for r in m.res][: max(1, m.preview)],
                "vitesse_max": m.maxspeed,
            }
        )
    return out


@register_driver(
    "touptek.camera",
    title="Caméra ToupTek",
    category="Caméras",
    kind="camera",
    description="Caméras ToupTek et marques OEM (Amscope, Omegon...) via le SDK officiel toupcam.",
    options=[
        OptionSpec(
            "camera_id", "str", "Caméra", None, help="Nom ou identifiant ; vide = première caméra détectée"
        ),
        OptionSpec(
            "bit_depth",
            "choice",
            "Profondeur",
            12,
            choices=("8", "12"),
            help="12 bits : meilleure dynamique pour mesurer",
        ),
        OptionSpec(
            "resolution_index", "int", "Résolution (index)", 0, min=0, max=8, help="0 = pleine résolution"
        ),
        OptionSpec(
            "sdk_path", "dir", "Dossier du SDK", None, advanced=True, help="Par défaut : ~/.fuji/touptek"
        ),
    ],
)
def _touptek(spec: DeviceSpec, ctx: DriverContext) -> ToupTekCamera:
    return ToupTekCamera(spec.name, label=spec.label, **spec.options)
