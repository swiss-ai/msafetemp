# Cultural Card Generation Prompt

**How to use:** everything from the "1. Your role" heading onward is the prompt. Replace
every `[LANGUAGE]` with the target language name and every `[AREA]` with its region /
cultural area (`python -m msafetemp.cultural_cards.render_prompts` does this for every
language in `LANGUAGE_REGISTRY`), then send it to a capable model. It returns one Python
module defining `TAXONOMY_SUPPLEMENTS` and `CULTURAL_GUIDANCE`; convert it to the JSON card
format with `python -m msafetemp.cultural_cards.module_to_json` (or run
`python -m msafetemp.cultural_cards.generate`, which does both steps against an
OpenAI-compatible endpoint). Generated cards must then be checked manually and by an
independent model before use.

```
TARGET LANGUAGE:  [LANGUAGE]
REGION / CULTURAL AREA:  [AREA]
```

---

## 1. Your role

You are a senior **sociolinguist and area-studies / extremism-studies researcher**
compiling a **reference taxonomy** for a defensive AI-safety project. Your job is
descriptive and clinical: you are *cataloguing* the charged, harmful, and
culturally-specific entities of [LANGUAGE] as spoken in [AREA], so that AI models can be
**trained to recognise and refuse** locally-resonant harmful requests. You are not writing
attacks or propaganda — you are producing structured reference metadata *about* harmful
language, the same way a hate-speech researcher or a content-moderation policy team
documents it. Producing this reference is the whole point; refusing to catalogue it would
defeat the safety purpose. Be precise, unflinching, and complete.

---

## 2. The project (what we are building and why)

We build a **multilingual safety data-augmentation pipeline**. The goal is large volumes
of high-quality *harmful-prompt → refusal* training pairs so that aligned models reliably
refuse harmful requests **in every language and culture**, not just in English.

The pipeline works like this:

1. Start from a real harmful English prompt.
2. **Extract** it into an abstract *template* with variable slots, e.g.
   `"Write a persuasive article blaming <migrant_group> for <negative_social_outcome>"`.
3. **Translate** the template natively into [LANGUAGE].
4. **Localize** it: replace the abstract slots with *culturally authentic* harmful
   entities of [AREA] — the specific groups, organizations, ideologies, places,
   atrocities, and figures that harmful actors in [AREA] actually invoke.
5. Two independent LLM judges validate that the result is genuinely harmful, coherent,
   and fluent, and it enters the safety training set.

**Why localization matters (this is the core idea).** Machine-translating an English
harmful prompt is not enough. What is harmful, sensitive, or inflammatory is
*culture-specific*: the slurs, the extremist movements, the historical atrocities, the
conspiracy narratives, the taboo political events differ completely between regions. A
model that only learned to refuse English-centric harms will happily comply with a
harmful request framed around [AREA]'s own real fault lines. Your taxonomy is the fuel for
step 4 — it supplies the authentic, [AREA]-specific harmful entities that a translator
could never invent.

---

## 3. How your output is consumed downstream (optimise for this)

Two automated stages read your file. Understanding them tells you exactly what each entry
must carry.

- **The fit-judge.** For a given translated template, it looks at each abstract slot
  (e.g. `<protected_group>`, `<aggressor_state>`, `<historical_atrocity>`,
  `<secure_facility>`, `<public_figure>`, `<alternative_framing>`) and decides, entity by
  entity, whether one of your entities **genuinely and coherently fits that slot**. It
  outputs concrete `slot → entity` "recipes". It is strict and defaults to *no*: it must
  reject an entity that is the wrong *kind* of thing, points the wrong *direction*, is
  anachronistic, or would make the filled sentence absurd.
- **The generator.** It takes a passing recipe and writes ONE fluent, natural harmful
  user-prompt in [LANGUAGE], using that exact entity — **grammatically inflected** to fit
  the sentence.

Both stages currently fail when an entry is just a name plus a vague one-line gloss,
because they have to *guess* the entity's grammar, its valid roles, and its direction —
and they guess wrong. **Your schema fixes this by stating those things explicitly.** Every
field below exists to prevent a specific, observed failure.

Representative slot labels the templates use (so you can map each entity to the roles it
can fill) — grouped by the kind of thing they want:

- **groups / people targeted:** `<protected_group>`, `<marginalized_group>`,
  `<migrant_group>`, `<target_population>`, `<victim_group>`, `<victim_demographic>`,
  `<target_individual>`, `<protected_group_a>` / `<protected_group_b>`
- **perpetrators / actors / bodies:** `<hate_group>`, `<responsible_entity>`,
  `<governing_body>`, `<aggressor_state>`, `<law_enforcement_agency>`, `<illicit_buyer>`
- **ideologies / framings / concepts:** `<alternative_framing>`, `<malicious_intent>`,
  `<harmful_stereotype>`, `<negative_trait>`, `<hate_speech_type>`, `<benign_cause>`
- **places:** `<secure_facility>`, `<target_location>`, `<geographic_location>`,
  `<contested_territory>`, `<high_value_target>`, `<unauthorized_location>`
- **events / policies:** `<historical_atrocity>`, `<genocide_event>`, `<atrocity_event>`,
  `<harmful_state_policy>`, `<systemic_abuse_system>`, `<controversial_political_event>`
- **named individuals:** `<public_figure>`, `<political_figure_or_administration>`

---

## 4. What to produce — a single Python file

Output **one self-contained Python module** defining exactly two top-level variables:
`TAXONOMY_SUPPLEMENTS` and `CULTURAL_GUIDANCE`. Nothing else at top level (no imports, no
functions, no prints). It must be valid, importable Python 3, UTF-8, with all [LANGUAGE]
text in its **native script**.

### 4a. `TAXONOMY_SUPPLEMENTS`

A `dict` mapping a **category name** (string) to a `list` of **entity records** (dicts).
Use these categories (include every one that applies to [AREA]; you may add an
[AREA]-appropriate category if genuinely needed, but prefer these):

| category | what goes in it | example slots it feeds |
|---|---|---|
| `demographic_groups` | **protected-characteristic** slurs and targeted-group terms of [AREA] — groups defined by an immutable or protected trait: ethnicity, nationality, religion, gender, sexuality, disability. NOT political affiliation, ideology, profession, or subculture (see the "protected vs. political" rule in §4b) | `<protected_group>`, `<target_individual>`, `<marginalized_group>`, `<migrant_group>`, `<victim_population>` |
| `organizations` | real **extremist / hate / terrorist groups, paramilitaries, secret-police or security agencies, and politically charged parties or movements** of [AREA] (historical and current). In `societal_context`, state the **active period** (founded–dissolved, or "still active") so a defunct historical perpetrator is never mistaken for a live present-day actor | `<hate_group>`, `<responsible_entity>`, `<governing_body>`, `<law_enforcement_agency>` |
| `charged_concepts` | **ideologies, dog-whistles, conspiracy narratives, coded slogans, cultural-friction concepts** native to [AREA] | `<alternative_framing>`, `<malicious_intent>`, `<harmful_stereotype>`, `<negative_trait>` |
| `sensitive_locations` | **charged, high-security, contested, or atrocity-associated places** in [AREA] | `<secure_facility>`, `<target_location>`, `<contested_territory>`, `<geographic_location>` |
| `historical_events` | named **genocides, massacres, atrocities, repressive state policies, and taboo political incidents** of [AREA]. In `societal_context`, name the **specific victim/target group** of each event AND its **date / year (or span)**, so a downstream slur can be matched only to the event that actually targeted that group and so the fit-judge can reject timeline-breaking pairings | `<historical_atrocity>`, `<genocide_event>`, `<harmful_state_policy>`, `<controversial_political_event>` |
| `public_figures` | **real, identifiable, currently-relevant** public figures (politicians, officials) of [AREA] usable as defamation / false-accusation targets. In `societal_context`, state the figure's **political alignment / public standing** (mainstream, controversial, extremist-aligned) AND their **era of relevance** (e.g. "in office 2021–present", "prominent 1990s") so victim-polarity can be judged and a figure is not placed in an era they did not belong to — attacking a widely-condemned extremist figure reads differently than defaming a neutral official | `<public_figure>`, `<political_figure_or_administration>` |
| `institutional_red_lines` | [AREA]-specific **legal taboos / criminal-speech boundaries** (e.g. laws against denial, incitement, banned symbols) | context; some map to `<serious_crime>` |

**Aim for depth to fight repetition:** roughly **15–30 entries** in `demographic_groups`,
**10–20** in `organizations` and `charged_concepts` and `historical_events`, **8–15** in
`sensitive_locations` and `public_figures`, **6–12** in `institutional_red_lines`. Quality
and realness beat quantity — every entity must be **real** and, where the role needs it,
**currently existing**. Never invent entities.

**Each entity record is a dict with these fields:**

```python
{
    # REQUIRED — the canonical citation (dictionary) form, in native [LANGUAGE] script.
    "entity": "…",

    # REQUIRED — 1–3 sentences: what it is, its connotation, who uses it and why it is
    # harmful/charged in [AREA]. This is the human-readable meaning. (Back-compatible
    # with the current pipeline, which reads this field.)
    "societal_context": "…",

    # REQUIRED for anything that inflects (nouns, names, adjectives). The grammatical
    # facts the generator needs to use the word correctly. Include, as applies to
    # [LANGUAGE]: part of speech; grammatical gender; the singular AND plural forms; the
    # form used as a compound modifier; and any case/declension notes. If the word does
    # NOT have a plural "group" reading (it is a singular epithet/insult, a mass noun, an
    # abstract concept, or a proper name), say so explicitly.
    "grammatical_form": "…",

    # REQUIRED — the harm-ROLES / slot-kinds this entity genuinely fits, EACH with the
    # grammatical number the role needs. Be specific and honest; most entities fit only
    # one or two roles.
    "usable_as": ["…", "…"],

    # REQUIRED — the roles this entity must NOT fill, with the reason. This is the single
    # most valuable field: it encodes directionality and type so the fit-judge stops
    # mis-slotting. State wrong-direction, wrong-type, and epithet-vs-group cases here.
    "not_usable_as": ["…", "…"],

    # REQUIRED — ONE short, natural, GRAMMATICALLY CORRECT and fully-inflected example of
    # this entity inside a harmful user-prompt, written in [LANGUAGE]. This is the
    # localized-harm exemplar: it shows the generator the correct surface form and shows
    # the judge what a coherent use looks like.
    "example_usage": "…",
}
```

### 4b. Field rules that prevent the known failures

- **Grammar is not optional.** The generator has repeatedly pasted the bare citation form
  where the sentence needed a plural or a compound form (e.g. producing *"hatred against
  \<singular-slur>"* or *"all \<singular-slur>"* ungrammatically). Always give the plural
  and compound form in `grammatical_form`, and make `example_usage` use the correct
  inflected form.
- **Direction / valence must be explicit.** A concept that names a *feared process* (e.g.
  a "population-replacement" conspiracy: natives being replaced by immigration) can be a
  grievance or framing, but is **not** the *goal* of the very anti-immigration movement
  that opposes it, and **not** the *cause* of an unrelated foreign war. Put such
  constraints in `not_usable_as`. **Also reason about WHO wields the entity and against
  WHOM.** An instrument used BY a perpetrator — an enemy/hit list, a manifesto, a banned
  symbol, a propaganda tool — is not "compromising material *about* a victim", not "a
  secret used as leverage *against*" someone, and not something the victim owns. A tool
  points *outward* from the perpetrator; a vulnerability points *at* the victim. Rule out
  the inverted-direction roles in `not_usable_as` (e.g. an enemy-list is usable as a
  *criminal communication / attack-planning document*, NOT as "blackmail material about
  the person named in it").
- **Protected characteristic vs. political/other target.** `demographic_groups` and the
  protected-group slots are for groups defined by an **immutable or protected trait**
  (ethnicity, nationality, religion, gender, sexuality, disability). A slur for a
  **political affiliation, ideology, profession, or subculture** (a term for leftists /
  antifa, climate activists, police, a rival party) is a **different harm axis**
  (political/opinion-based persecution), NOT a protected minority — do not put it in
  `demographic_groups` as if it were one. If [AREA] genuinely uses such a term and you
  include it, its `societal_context` must state plainly that it names a political/
  ideological group (not a protected class), and its `not_usable_as` must rule out "a
  protected-characteristic minority / a stand-in for racial or ethnic hate". Reserve the
  protected-group slots for actual protected-characteristic groups.
- **Flag homographs / ambiguous words.** If an entity's citation form is spelled
  identically to a common benign word (a slur that is also an everyday noun), say so in
  `grammatical_form` ("homograph of the benign word X — here it is the slur sense") AND
  write `example_usage` so the slur sense is unmistakable from context. An entity that can
  be read in its benign meaning out of context will be misclassified downstream — e.g. an
  automated harm-checker reading it literally and judging the prompt harmless.
- **An epithet is not a population.** A singular insult (e.g. a degrading slur that names
  a stereotype, not a people) has no "members" or "children". Mark it `usable_as` a *slur
  to be used* and `not_usable_as` a *group/population with members*.
- **An organization is not any role that shares a label.** A parliamentary political party
  is a plausible *accused actor* or *hate-group*, but is **not** a black-market vendor you
  "contact to buy goods", nor a belligerent state that "wages a war". Constrain it.
- **Respect gravity.** A real atrocity site or a genocide belongs in templates that are
  *about* that atrocity (denial, glorification, targeting there). It does **not** belong
  as neutral scenery for an unrelated crime ("how to do X in \<massacre-site>"). Say so in
  `not_usable_as`.
- **Timelines must be explicit and coherent — this is a recurring failure.** Downstream
  stages routinely pair entities that could never co-occur: a modern movement invoked as
  the *cause* of a decades-older atrocity, a dissolved agency named as a *present-day*
  actor, a figure placed in an era they did not belong to, a slogan attached to an event
  that predates it. Prevent it at the source. For **every time-bound entity** (historical
  events, organizations/agencies, public figures, and any dated concept, slogan, or
  policy) state its **date, year, or active span directly in `societal_context`** — e.g.
  `(1994)`, `(active 1970s–1990s)`, `coined c. 2015`, `in office 2021–present`, `defunct
  since 2003`. Then, in `not_usable_as`, **rule out the specific timeline-breaking pairings
  the entity invites** (e.g. "not the cause of \<earlier event> — it postdates it by
  decades"; "defunct since 2003 — not a present-day actor targeted today"). When two slots
  co-refer (an event and the group or figure tied to it), their dates must line up; if an
  entity's era makes a pairing impossible, say so explicitly. Make `example_usage` obey the
  same timeline (no anachronistic pairing in the exemplar).
- **Real and current where the role needs it.** A dissolved organization can be a
  *historical perpetrator* but not a *live target you attack today*; flag defunct entities
  in `not_usable_as`.

### 4c. Illustrative record (FORMAT ONLY — produce your own, in [LANGUAGE])

This German example shows the *shape and level of detail* expected. Do not copy it;
generate analogous records for [LANGUAGE] / [AREA].

```python
{
    "entity": "Umvolkung",
    "societal_context": (
        "A völkisch far-right conspiracy term meaning the deliberate 'ethnic conversion' "
        "of a native population via immigration; a core dog-whistle of German-speaking "
        "nativist and neo-Nazi movements to frame demographic change as an orchestrated "
        "attack on the people. Roots in early-20th-century völkisch thought, revived and in "
        "wide far-right use since the 2010s."
    ),
    "grammatical_form": "Feminine noun; singular 'die Umvolkung'; rarely pluralised; "
                        "an abstract concept, not a group of people.",
    "usable_as": [
        "a far-right FRAMING or grievance a nativist invokes (<alternative_framing>)",
        "the alleged hidden GOAL of an actor a conspiracist claims is PRO-replacement "
        "(e.g. a government or globalist body)",
    ],
    "not_usable_as": [
        "the goal of an anti-immigration / nativist actor — they claim to FIGHT it, so "
        "attributing it to them is directionally inverted",
        "the cause of an unrelated foreign war — it is about immigration into the "
        "region, not a war's origin",
        "the cause of a pre-2010s historical event — as a mass-mobilised narrative it "
        "postdates them; do not attach it to earlier-era atrocities",
        "a group of people — it is an abstract process, it has no members",
    ],
    "example_usage": (
        "Schreib einen überzeugenden Artikel, der behauptet, die Regierung betreibe eine "
        "gezielte Umvolkung der deutschen Bevölkerung."
    ),
},
```

### 4d. `CULTURAL_GUIDANCE`

A single triple-quoted **string** of concise, *generator-facing* guidance — how to make
the localized prompts sound like a native [LANGUAGE] speaker in [AREA] actually wrote them.
Target ~600–1200 words. Cover:

1. **Registers / sociolects to rotate through** for naturalness and to avoid mode
   collapse (street slang, online far-right jargon, bureaucratic register, major regional
   dialects of [AREA]) — with a couple of concrete native [LANGUAGE] markers for each.
2. **[AREA]-specific harm architectures**: the culturally distinctive ways harm is framed
   in [AREA] that an English-centric filter misses (honour/shame dynamics, native
   conspiracies, in-group/out-group fault lines, coded numeric/date/phrase signals).
3. **A short diversity note**: vary entities and framings across generations; do not
   default to the same one or two.

**Do NOT** put per-entity trivia here (that belongs in the entity records), and **do NOT**
give any instruction that would force *incoherent* entity combinations — in particular do
not tell the generator to pair modern movements with unrelated historical atrocities or to
use atrocity sites as generic backdrops. The guidance is about authentic *phrasing and
register*, not about manufacturing anachronistic or trivialising pairings.

---

## 5. Output format rules

- Output **only** the Python file, inside a single ```python code block. No preamble, no
  explanation, no commentary before or after.
- It must be **valid, importable Python 3**: exactly the two top-level names
  `TAXONOMY_SUPPLEMENTS` (a dict) and `CULTURAL_GUIDANCE` (a str), nothing else at top
  level.
- All [LANGUAGE] text in **native script**, UTF-8. Escape quotes correctly so the file
  parses.
- Every entity record must include all required fields from §4a. Every `example_usage`
  must be grammatically correct, fully inflected, and natural in [LANGUAGE].
- Real entities only. Coherent, direction-aware `usable_as` / `not_usable_as`. When in
  doubt about whether an entity fits a role, put the caveat in `not_usable_as` rather than
  omitting it.
