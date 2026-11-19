"""Utilidades de texto para a interface."""
import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$])")


def md_escape(text: str) -> str:
    """Escapa texto vindo da base antes de ir para st.markdown (evita formatação e LaTeX acidentais)."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))
