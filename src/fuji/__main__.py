"""``python -m fuji`` — et point d'entrée de l'application empaquetée (Fuji.app)."""

import sys

from fuji.cli import main

# Lancée depuis le Finder, une application peut recevoir un numéro de processus (-psn_…)
arguments = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
sys.exit(main(arguments))
