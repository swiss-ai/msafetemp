"""Verbatim generator and judge prompts used by the augmentation pipeline.

core            extraction, validity judge, harmful fills, four-axis judge, paraphrases,
                affixes, jailbreak-wrapper extractor
translation     per-track template translators, translation judge, targeted patcher
benign          exact-fill benign minimal pairs + class judge
posthoc         cultural fit judge + contextual generator
benign_audit    prompt-side audit of the benign set
refusal_v2      refusal response design v2 + its judge
benign_response benign response generators + judge
wrappers        wrapper translation checks and the assembled-prompt dual judge
"""
