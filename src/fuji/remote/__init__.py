"""Commande à distance du microscope, pour un programme ou une IA externe.

- :class:`~fuji.remote.controller.MicroscopeController` : une façade simple et sûre du
  microscope (réglages, images, moteurs, acquisitions, analyse), utilisable depuis
  n'importe quel thread, avec des garde-fous configurables ;
- :mod:`fuji.remote.mcp_server` : expose ce contrôleur comme serveur MCP (Model Context
  Protocol), le standard par lequel un assistant IA appelle des outils.
"""

from .controller import ControlError, ManualActionRequired, MicroscopeController, NotAllowed, SafetyPolicy

__all__ = ["ControlError", "ManualActionRequired", "MicroscopeController", "NotAllowed", "SafetyPolicy"]
