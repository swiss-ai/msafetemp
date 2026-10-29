"""Per-language cultural cards for post-hoc cultural localization (PH).

A cultural card lists the locally relevant, charged entities of one language / region
(demographic groups, organizations, charged concepts, sensitive locations, historical
events, public figures, institutional red lines). Each entity record carries
`entity`, `societal_context`, `grammatical_form`, `usable_as`, `not_usable_as` and
`example_usage`; the card also carries generator-facing `cultural_guidance`. The entity
list is the language's *entity pool* for post-hoc localization: the fit judge
(msafetemp.augmentation.posthoc) sees entity + context + form + usable_as / NOT usable_as,
the contextual generator receives grammatical_form + example_usage of the entities a recipe
uses.

Cards are stored as JSON under data/cultural_cards/<language>.json (override the directory
with MSAFETEMP_CARDS_DIR). English and Zulu have no card.
"""

from __future__ import annotations

import json
from pathlib import Path

from msafetemp.common.config import cards_dir

# Language key -> display name, script and region (the [LANGUAGE] / [AREA] values used to
# render the card generation prompt). English is the source language (no card).
LANGUAGE_REGISTRY: dict[str, dict] = {
    # English: source language (no translation, no cultural card)
    "english":               {"name": "English",                 "script": "Latin",                    "region": "the United States, the United Kingdom, and the Anglophone world"},
    # Initial languages
    "albanian":              {"name": "Albanian",                "script": "Latin",                    "region": "Albania and the Western Balkans"},
    "tagalog":               {"name": "Tagalog",                 "script": "Latin",                    "region": "Philippines"},
    "ukrainian":             {"name": "Ukrainian",               "script": "Cyrillic",                 "region": "Ukraine"},
    # European
    "russian":               {"name": "Russian",                 "script": "Cyrillic",                 "region": "Russia, CIS states, and Russian diaspora"},
    "spanish":               {"name": "Spanish",                 "script": "Latin",                    "region": "Latin America and Spain"},
    "french":                {"name": "French",                  "script": "Latin",                    "region": "West Africa and France"},
    "portuguese_brazilian":  {"name": "Portuguese (Brazilian)",  "script": "Latin",                    "region": "Brazil"},
    "polish":                {"name": "Polish",                  "script": "Latin",                    "region": "Poland and Eastern Europe"},
    "serbian":               {"name": "Serbian",                 "script": "Cyrillic",                 "region": "Serbia and the Western Balkans"},
    "german":                {"name": "German",                  "script": "Latin",                    "region": "Germany, Austria, and German-speaking Switzerland"},
    "italian":               {"name": "Italian",                 "script": "Latin",                    "region": "Italy"},
    "romanian":              {"name": "Romanian",                "script": "Latin",                    "region": "Romania and Moldova"},
    "greek":                 {"name": "Greek",                   "script": "Greek",                    "region": "Greece and Cyprus"},
    "dutch":                 {"name": "Dutch",                   "script": "Latin",                    "region": "Netherlands and Belgium (Flanders)"},
    "hungarian":             {"name": "Hungarian",               "script": "Latin",                    "region": "Hungary and Hungarian diaspora in Central Europe"},
    "czech":                 {"name": "Czech",                   "script": "Latin",                    "region": "Czech Republic (Czechia)"},
    "slovak":                {"name": "Slovak",                  "script": "Latin",                    "region": "Slovakia and Slovak diaspora in Central Europe"},
    "bulgarian":             {"name": "Bulgarian",               "script": "Cyrillic",                 "region": "Bulgaria and Bulgarian diaspora in Southeast Europe"},
    "croatian":              {"name": "Croatian",                "script": "Latin",                    "region": "Croatia and Croatian diaspora in the Western Balkans"},
    "slovenian":             {"name": "Slovenian",               "script": "Latin",                    "region": "Slovenia and Slovenian diaspora"},
    "macedonian":            {"name": "Macedonian",              "script": "Cyrillic",                 "region": "North Macedonia and Macedonian diaspora in the Balkans"},
    "swedish":               {"name": "Swedish",                 "script": "Latin",                    "region": "Sweden and Finnish Swedish-speaking minority"},
    "norwegian":             {"name": "Norwegian",               "script": "Latin",                    "region": "Norway (Bokmål)"},
    "danish":                {"name": "Danish",                  "script": "Latin",                    "region": "Denmark and Danish diaspora"},
    "finnish":               {"name": "Finnish",                 "script": "Latin",                    "region": "Finland and Finnish diaspora"},
    "lithuanian":            {"name": "Lithuanian",              "script": "Latin",                    "region": "Lithuania and Baltic states"},
    "latvian":               {"name": "Latvian",                 "script": "Latin",                    "region": "Latvia and Baltic diaspora"},
    "estonian":              {"name": "Estonian",                "script": "Latin",                    "region": "Estonia and Estonian diaspora"},
    "swiss_german":          {"name": "Swiss German (Schweizerdeutsch)", "script": "Latin",             "region": "Switzerland"},
    # European (extended coverage)
    "irish":                 {"name": "Irish (Gaeilge)",         "script": "Latin",                    "region": "Ireland"},
    "maltese":               {"name": "Maltese",                 "script": "Latin",                    "region": "Malta"},
    "icelandic":             {"name": "Icelandic",               "script": "Latin",                    "region": "Iceland"},
    "luxembourgish":         {"name": "Luxembourgish (Lëtzebuergesch)", "script": "Latin",             "region": "Luxembourg"},
    "catalan":               {"name": "Catalan",                 "script": "Latin",                    "region": "Catalonia, Valencia, and the Balearic Islands (Spain)"},
    "basque":                {"name": "Basque (Euskara)",        "script": "Latin",                    "region": "Basque Country (Spain and France)"},
    "galician":              {"name": "Galician",                "script": "Latin",                    "region": "Galicia (Spain)"},
    "welsh":                 {"name": "Welsh (Cymraeg)",         "script": "Latin",                    "region": "Wales (United Kingdom)"},
    "belarusian":            {"name": "Belarusian",              "script": "Cyrillic",                 "region": "Belarus and Belarusian diaspora"},
    "bosnian":               {"name": "Bosnian",                 "script": "Latin",                    "region": "Bosnia and Herzegovina"},
    "georgian":              {"name": "Georgian",                "script": "Georgian (Mkhedruli)",     "region": "Georgia"},
    "armenian":              {"name": "Armenian",                "script": "Armenian",                 "region": "Armenia and Armenian diaspora"},
    # Global
    "arabic":                {"name": "Arabic",                  "script": "Arabic (RTL)",             "region": "Middle East and North Africa (MENA)"},
    "hindi":                 {"name": "Hindi",                   "script": "Devanagari",               "region": "India and South Asia"},
    "chinese_simplified":    {"name": "Chinese (Simplified)",    "script": "Hanzi (Simplified)",       "region": "China and Chinese diaspora"},
    "indonesian":            {"name": "Indonesian",              "script": "Latin",                    "region": "Indonesia and Southeast Asia"},
    "turkish":               {"name": "Turkish",                 "script": "Latin",                    "region": "Turkey and Central Asia"},
    "swahili":               {"name": "Swahili",                 "script": "Latin",                    "region": "East Africa"},
    "bengali":               {"name": "Bengali",                 "script": "Bengali (Bangla)",         "region": "Bangladesh and West Bengal, India"},
    "korean":                {"name": "Korean",                  "script": "Hangul",                   "region": "South Korea and Korean diaspora"},
    "japanese":              {"name": "Japanese",                "script": "Kanji/Hiragana/Katakana",  "region": "Japan"},
    "persian":               {"name": "Persian",                 "script": "Arabic (RTL)",             "region": "Iran and Persian-speaking diaspora"},
    "yoruba":                {"name": "Yoruba",                  "script": "Latin",                    "region": "Nigeria and West Africa"},
    "amharic":               {"name": "Amharic",                 "script": "Ethiopic (Ge'ez)",         "region": "Ethiopia"},
    "thai":                  {"name": "Thai",                    "script": "Thai",                     "region": "Thailand"},
    "vietnamese":            {"name": "Vietnamese",              "script": "Latin (tonal diacritics)", "region": "Vietnam"},
    # Global (extended low-resource coverage)
    "urdu":                  {"name": "Urdu",                    "script": "Arabic (Nastaʿlīq, RTL)",  "region": "Pakistan and South Asian diaspora"},
    "hausa":                 {"name": "Hausa",                   "script": "Latin",                    "region": "Nigeria, Niger, and West Africa"},
    "zulu":                  {"name": "Zulu (isiZulu)",          "script": "Latin",                    "region": "South Africa"},
    "nepali":                {"name": "Nepali",                  "script": "Devanagari",               "region": "Nepal"},
}


SUPPORTED_LANGUAGES = list(LANGUAGE_REGISTRY.keys())

CARD_FIELDS = ("entity", "societal_context", "grammatical_form", "usable_as",
               "not_usable_as", "example_usage")


def card_path(language: str, directory: str | Path | None = None) -> Path:
    return Path(directory or cards_dir()) / f"{language}.json"


def has_card(language: str, directory: str | Path | None = None) -> bool:
    return card_path(language, directory).exists()


def available_cards(directory: str | Path | None = None) -> list[str]:
    d = Path(directory or cards_dir())
    return sorted(p.stem for p in d.glob("*.json"))


def load_card(language: str, directory: str | Path | None = None) -> dict:
    """{"taxonomy_supplements": {category: [entity records]}, "cultural_guidance": str, ...}"""
    path = card_path(language, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"no cultural card for '{language}' at {path}; generate one with "
            f"python -m msafetemp.cultural_cards.generate --language {language}")
    card = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(card.get("taxonomy_supplements"), dict) or \
            not isinstance(card.get("cultural_guidance"), str):
        raise ValueError(f"{path}: card must define taxonomy_supplements (dict) and "
                         f"cultural_guidance (str)")
    return card


def entity_pool(card: dict) -> list[str]:
    """All entity strings of a card (the language's entity pool)."""
    out = []
    for entities in card["taxonomy_supplements"].values():
        for e in entities:
            name = e.get("entity") if isinstance(e, dict) else e
            if name:
                out.append(str(name))
    return out
