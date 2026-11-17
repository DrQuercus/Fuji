#!/bin/bash
# Fabrique Fuji.app et son .dmg sur ce Mac (pour le processeur de ce Mac).
#
#   ./packaging/macos/build_dmg.sh
#
# Besoin : macOS 13 ou plus récent, Python 3.12 (python.org ou Homebrew) et les outils de
# ligne de commande d'Apple (« xcode-select --install »). Compter 10 à 20 minutes et une
# connexion Internet : toutes les bibliothèques sont téléchargées puis copiées dans l'application.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    echo "Il faut Python 3.10 ou plus récent (trouvé : $("$PYTHON" --version 2>&1))." >&2
    exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Les outils de ligne de commande d'Apple manquent. Lance : xcode-select --install" >&2
    exit 1
fi

"$PYTHON" -m venv .venv-empaquetage
source .venv-empaquetage/bin/activate
python -m pip install --quiet --upgrade pip briefcase

rm -rf build/fuji/macos dist/Fuji-*.dmg
briefcase create macOS app --no-input
briefcase build macOS app --no-input

APP="build/fuji/macos/app/Fuji.app"
python packaging/macos/check_min_os.py "$APP"
echo "Essai de l'application (une fenêtre s'ouvre quelques secondes)…"
"$APP/Contents/MacOS/Fuji" selftest

briefcase package macOS app --adhoc-sign --no-input
ls -lh dist/*.dmg
open dist
