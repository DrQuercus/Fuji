"""Faux module ``toupcam`` (imite le SDK ToupTek) pour tester le pilote sans caméra.

Il reproduit ce que le pilote utilise : énumération, options, exposition, gain,
déclenchement logiciel, mode vidéo, et surtout l'écriture des images dans un
tampon ``bytes`` avec des lignes alignées sur 4 octets (format DIB), comme le SDK.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field

import numpy as np

TOUPCAM_FLAG_MONO = 0x00000010
TOUPCAM_FLAG_RAW10 = 0x00000080
TOUPCAM_FLAG_RAW12 = 0x00000100
TOUPCAM_FLAG_TRIGGER_SOFTWARE = 0x00080000
TOUPCAM_FLAG_CG = 0x0000000200000000

TOUPCAM_OPTION_BITDEPTH = 0x06
TOUPCAM_OPTION_TRIGGER = 0x0B
TOUPCAM_OPTION_RGB = 0x0C
TOUPCAM_OPTION_BYTEORDER = 0x0D
TOUPCAM_OPTION_BLACKLEVEL = 0x15
TOUPCAM_OPTION_CG = 0x1A
TOUPCAM_OPTION_FLUSH = 0x3D

TOUPCAM_EVENT_IMAGE = 0x0004
TOUPCAM_EVENT_TRIGGERFAIL = 0x0007
TOUPCAM_EVENT_ERROR = 0x0080
TOUPCAM_EVENT_DISCONNECTED = 0x0081


class HRESULTException(Exception):
    def __init__(self, hr: int) -> None:
        super().__init__(hex(hr))
        self.hr = hr


@dataclass
class Resolution:
    width: int
    height: int


@dataclass
class ModelV2:
    name: str
    flag: int
    maxspeed: int = 2
    preview: int = 2
    still: int = 0
    maxfanspeed: int = 0
    ioctrol: int = 0
    xpixsz: float = 2.9
    ypixsz: float = 2.9
    res: list[Resolution] = field(default_factory=list)


@dataclass
class DeviceV2:
    displayname: str
    id: str
    model: ModelV2


# configuration modifiable par les tests
DEVICES: list[DeviceV2] = [
    DeviceV2(
        "E3ISPM08300KPD",
        "tp-usb-1",
        ModelV2(
            "E3ISPM08300KPD",
            TOUPCAM_FLAG_RAW12 | TOUPCAM_FLAG_TRIGGER_SOFTWARE | TOUPCAM_FLAG_CG,
            res=[Resolution(26, 10), Resolution(13, 5)],  # 13*3 octets : lignes à compléter
        ),
    )
]
OPENED: list[Toupcam] = []


class Toupcam:
    @staticmethod
    def EnumV2() -> list[DeviceV2]:
        return list(DEVICES)

    @staticmethod
    def Open(camid: str) -> Toupcam | None:
        dev = next((d for d in DEVICES if d.id == camid), None)
        if dev is None:
            return None
        cam = Toupcam(dev)
        OPENED.append(cam)
        return cam

    @staticmethod
    def Version() -> str:
        return "fake-57.0"

    def __init__(self, dev: DeviceV2) -> None:
        self.dev = dev
        self.options: dict[int, int] = {
            TOUPCAM_OPTION_TRIGGER: 0,
            TOUPCAM_OPTION_RGB: 0,
            TOUPCAM_OPTION_BITDEPTH: 0,
            TOUPCAM_OPTION_BLACKLEVEL: 0,
            TOUPCAM_OPTION_CG: 0,
        }
        self.esize = 0
        self.expo_us = 10_000
        self.gain = 100
        self.autoexpo = 1
        self.hflip = self.vflip = 0
        self.gamma, self.contrast, self.brightness, self.speed, self.sat = 100, 0, 0, 0, 128
        self.temp_tint = (6503, 1000)
        self.awb_calls = 0
        self.frame_no = 0
        self.closed = False
        #: Comportements à simuler dans les tests.
        self.stale_frames_on_trigger = 0  # images « en transit » livrées juste après un Trigger
        self.fail_next_trigger = False
        self.scale_to_16_bits = False  # valeurs étalées sur 16 bits au lieu de 12
        self.flushed = 0
        self.fail_option: int | None = None  # put_Option(cette option) lève une erreur
        self._cb = None
        self._ctx = None
        self._running = False
        self._pending = 0
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None

    # --- réglages ---
    def get_Size(self) -> tuple[int, int]:
        r = self.dev.model.res[self.esize]
        return r.width, r.height

    def put_eSize(self, i: int) -> None:
        self.esize = i

    def put_Option(self, opt: int, value: int) -> None:
        if opt == self.fail_option:
            raise HRESULTException(0x80004001)
        if opt == TOUPCAM_OPTION_FLUSH:
            self.flushed += 1
            return
        self.options[opt] = int(value)
        if opt == TOUPCAM_OPTION_TRIGGER:
            with self._cond:
                self._cond.notify_all()

    def get_Option(self, opt: int) -> int:
        return self.options.get(opt, 0)

    def put_AutoExpoEnable(self, v: int) -> None:
        self.autoexpo = int(v)

    def get_AutoExpoEnable(self) -> int:
        return self.autoexpo

    def put_ExpoTime(self, us: int) -> None:
        self.expo_us = int(us)

    def get_ExpoTime(self) -> int:
        return self.expo_us

    def get_ExpTimeRange(self) -> tuple[int, int, int]:
        return 100, 10_000_000, 10_000

    def put_ExpoAGain(self, g: int) -> None:
        self.gain = int(g)

    def get_ExpoAGain(self) -> int:
        return self.gain

    def get_ExpoAGainRange(self) -> tuple[int, int, int]:
        return 100, 5000, 100

    def MaxBitDepth(self) -> int:
        return 12

    def AwbOnce(self) -> None:
        self.awb_calls += 1

    def get_TempTint(self) -> tuple[int, int]:
        return self.temp_tint

    def put_TempTint(self, t: int, tint: int) -> None:
        self.temp_tint = (t, tint)

    def get_HFlip(self) -> int:
        return self.hflip

    def put_HFlip(self, v: int) -> None:
        self.hflip = v

    def get_VFlip(self) -> int:
        return self.vflip

    def put_VFlip(self, v: int) -> None:
        self.vflip = v

    def get_Gamma(self) -> int:
        return self.gamma

    def put_Gamma(self, v: int) -> None:
        self.gamma = v

    def get_Contrast(self) -> int:
        return self.contrast

    def put_Contrast(self, v: int) -> None:
        self.contrast = v

    def get_Brightness(self) -> int:
        return self.brightness

    def put_Brightness(self, v: int) -> None:
        self.brightness = v

    def get_Speed(self) -> int:
        return self.speed

    def put_Speed(self, v: int) -> None:
        self.speed = v

    def get_Saturation(self) -> int:
        return self.sat

    def put_Saturation(self, v: int) -> None:
        self.sat = v

    # --- flux ---
    def StartPullModeWithCallback(self, cb, ctx) -> None:
        self._cb, self._ctx = cb, ctx
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            with self._cond:
                video = self.options[TOUPCAM_OPTION_TRIGGER] == 0
                if not video and self._pending == 0:
                    self._cond.wait(0.05)
                    continue
                if not video:
                    self._pending -= 1
            time.sleep(self.expo_us / 1e6)  # une image n'arrive jamais avant la fin de la pose
            if self._running and self._cb is not None:
                self._emit_frame()

    def _emit_frame(self) -> None:
        self.frame_no += 1
        self._cb(TOUPCAM_EVENT_IMAGE, self._ctx)

    def emit_event(self, event: int) -> None:
        self._cb(event, self._ctx)

    def Trigger(self, n: int) -> None:
        if self.fail_next_trigger:
            self.fail_next_trigger = False
            threading.Timer(0.01, self.emit_event, (TOUPCAM_EVENT_TRIGGERFAIL,)).start()
            return
        for _ in range(self.stale_frames_on_trigger):  # image vidéo arrivée en retard
            threading.Timer(0.001, self._emit_frame).start()
        self.stale_frames_on_trigger = 0
        with self._cond:
            self._pending += n
            self._cond.notify_all()

    def Stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(1)

    def Close(self) -> None:
        self.Stop()
        self.closed = True

    def expected_frame(self, bits: int) -> np.ndarray:
        """Image que le pilote doit retrouver (sans le bourrage des lignes)."""
        w, h = self.get_Size()
        mono = bool(self.dev.model.flag & TOUPCAM_FLAG_MONO)
        n = 1 if mono else 3
        maxval = 4095 if bits in (16, 48) else 255
        base = (np.arange(h * w * n).reshape(h, w, n) * 7 + self.frame_no) % (maxval + 1)
        dtype = np.uint16 if bits in (16, 48) else np.uint8
        out = base.astype(dtype)
        if self.scale_to_16_bits and bits in (16, 48):
            out = out << 4
        return out[..., 0] if mono else out

    def PullImageV3(self, buf: bytes, still: int, bits: int, row_pitch: int, info: object) -> None:
        assert still == 0 and row_pitch == 0
        self.PullImageV2(buf, bits, info)

    def SerialNumber(self) -> str:
        return "SN-TEST"

    def PullImageV2(self, buf: bytes, bits: int, info: object) -> None:
        img = self.expected_frame(bits)
        w, h = self.get_Size()
        stride = ((w * bits + 31) // 32) * 4
        raw = np.zeros((h, stride), np.uint8)
        row_bytes = img.reshape(h, -1).view(np.uint8)
        raw[:, : row_bytes.shape[1]] = row_bytes
        raw[:, row_bytes.shape[1] :] = 0xAB  # bourrage visible si mal géré
        if len(buf) < raw.size:
            raise HRESULTException(0x80070057)
        addr = ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p).value
        ctypes.memmove(addr, raw.ctypes.data, raw.size)
