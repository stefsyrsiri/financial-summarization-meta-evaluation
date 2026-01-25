from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str

LANGUAGES = {
    "English": Language("en"),
    "Greek": Language("el"),
    "Spanish": Language("es"),
}
