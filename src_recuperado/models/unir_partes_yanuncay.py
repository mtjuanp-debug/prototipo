# -*- coding: utf-8 -*-
"""Envoltorio de compatibilidad: ``unir_partes.py --rio yanuncay``.

Uso (igual que antes; ``--destino`` es ahora opcional y por defecto ``Docs/resources/modelos_yanuncay``)::

    python src/models/unir_partes_yanuncay.py PARTE1 PARTE2 ... --destino Docs/resources/modelos_yanuncay
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unir_partes import main  # noqa: E402

if __name__ == "__main__":
    main(["--rio", "yanuncay", *sys.argv[1:]])
