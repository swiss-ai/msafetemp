# Cultural cards

One JSON file per language key (57 languages):

```json
{
  "language": "german", "name": "German", "script": "Latin",
  "region": "Germany, Austria, and German-speaking Switzerland",
  "taxonomy_supplements": {"<category>": [{"entity": "...", "societal_context": "...",
     "grammatical_form": "...", "usable_as": ["..."], "not_usable_as": ["..."],
     "example_usage": "..."}]},
  "cultural_guidance": "..."
}
```

Category order and entity order are significant: the fit judge sees the card rendered in file
order. Load with `msafetemp.cultural_cards.load_card(language)`. See
`msafetemp/cultural_cards/README.md` for how the cards were made and for coverage notes
(no card for English, Zulu and Maltese).

Content warning: the cards catalogue real slurs, extremist organizations, atrocities and
charged public figures. They exist to build refusal and evaluation data and are released for
that research purpose only.
