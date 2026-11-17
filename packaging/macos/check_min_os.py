"""Version minimale de macOS réellement exigée par une application empaquetée.

    python packaging/macos/check_min_os.py build/fuji/macos/app/Fuji.app [--strict]

Chaque bibliothèque compilée de l'application déclare la version de macOS à partir de
laquelle elle fonctionne. Ce script les lit toutes (commande ``otool`` de macOS) et compare
la plus exigeante à la version annoncée par l'application (``LSMinimumSystemVersion``).
Avec ``--strict``, une application qui annonce moins que ce qu'elle exige est refusée :
mieux vaut un message clair « nécessite macOS 13 » qu'un plantage au démarrage.
"""

from __future__ import annotations

import plistlib
import re
import subprocess
import sys
from pathlib import Path

MAGIC = {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xce\xfa\xed\xfe"}
PATTERN = re.compile(
    r"cmd (LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX)\n(?:.*\n)*?\s+(?:minos|version) (\d+(?:\.\d+)+)"
)


def version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def min_os(path: Path) -> str | None:
    """Plus petite version de macOS acceptée par un fichier Mach-O (None si ce n'en est pas un)."""
    try:
        with path.open("rb") as f:
            if f.read(4) not in MAGIC:
                return None
    except OSError:
        return None
    out = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True, check=False).stdout
    found = [m.group(2) for m in PATTERN.finditer(out)]
    return max(found, key=version) if found else None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    app = Path(args[0])
    with (app / "Contents" / "Info.plist").open("rb") as f:
        declared = str(plistlib.load(f).get("LSMinimumSystemVersion", "0"))
    needs: dict[str, list[Path]] = {}
    for path in app.rglob("*"):
        if path.is_file() and not path.is_symlink():
            found = min_os(path)
            if found:
                needs.setdefault(found, []).append(path)
    if not needs:
        print("Aucune bibliothèque compilée trouvée : mauvais dossier ?")
        return 2
    print(f"{sum(len(v) for v in needs.values())} bibliothèques compilées dans {app.name}")
    for wanted in sorted(needs, key=version, reverse=True)[:4]:
        names = sorted({p.relative_to(app).parts[-1] for p in needs[wanted]})
        shown = ", ".join(names[:6]) + ("…" if len(names) > 6 else "")
        print(f"  macOS {wanted:6s} : {len(needs[wanted]):4d} fichiers ({shown})")
    required = max(needs, key=version)
    print(f"Version exigée par les bibliothèques : macOS {required}")
    print(f"Version annoncée par l'application   : macOS {declared}")
    if version(required) > version(declared):
        print(
            "⚠ L'application annonce une version trop ancienne : "
            "augmenter min_os_version dans pyproject.toml."
        )
        return 1 if strict else 0
    print("✔ Cohérent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
