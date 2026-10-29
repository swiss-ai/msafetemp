# msafetemp.cultural_cards

Per-language cultural cards for post-hoc localization (PH).

A card lists locally relevant, charged entities of one language and region in seven
categories (`demographic_groups`, `organizations`, `charged_concepts`, `sensitive_locations`,
`historical_events`, `public_figures`, `institutional_red_lines`). Each entity record carries
`entity` (native script, citation form), `societal_context` (meaning, connotation, dates),
`grammatical_form`, `usable_as`, `not_usable_as` (roles it must not fill, with reasons) and
`example_usage`. The card also carries `cultural_guidance`, generator-facing notes on register
and local harm architectures. The entity list is the language's entity pool.

How the cards are consumed (`msafetemp.augmentation.posthoc`): the fit judge sees entity,
context, form, usable_as and NOT usable_as; the contextual generator receives the
grammatical_form and example_usage of the entities a recipe uses.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | `LANGUAGE_REGISTRY` (name / script / region per language key), `load_card`, `entity_pool` |
| `CARD_GENERATION_PROMPT.md` | the card generation guideline (verbatim prompt from "1. Your role" on) |
| `render_prompts.py` | renders the guideline per language (`[LANGUAGE]`, `[AREA]` substituted) |
| `generate.py` | sends a rendered prompt to an OpenAI-compatible endpoint and writes a DRAFT card |
| `module_to_json.py` | converts a card returned as a Python module (the format the guideline asks for) to JSON, parsing with `ast` only |

The cards themselves are in `data/cultural_cards/<language>.json` (57 languages). Load them with
`load_card(language)`; set `MSAFETEMP_CARDS_DIR` to use another directory.

## How the released cards were made

1. The guideline was rendered per language and answered by a large model (Gemma 4 31B IT). The reply is one Python module per language.
2. Every card was checked manually for coherence and validity of the terms, then sanity-checked
   by an independent model (Gemini Flash 3.6), which corrected occasional typos and
   grammar errors.

Step 2 was a manual review plus an interactive model check; there is no automated verifier in
this package. `generate.py` reproduces step 1 only and its output must be reviewed before use.
Decoding settings of the original card generation were not recorded; `generate.py` defaults to
temperature 0.7.

## Coverage notes

* English (source language) and Zulu have no card. Zulu therefore has no PH arm.
* Maltese: the card file available at release time is truncated (it does not parse), so it is
  not shipped. The HarmBench Maltese PH variants were generated with an earlier version of this
  card.
* The cards of 14 shipped languages that appear only in the HarmBench track (armenian, basque,
  belarusian, bosnian, catalan, galician, georgian, hausa, icelandic, irish, luxembourgish,
  nepali, urdu, welsh) were re-authored after the HarmBench PH run. The shipped files are these
  later versions, so not every HarmBench PH recipe of these languages is contained in the
  shipped entity pool. The cards of the 43 WildGuardMix languages are exactly the ones used for
  the released data.
