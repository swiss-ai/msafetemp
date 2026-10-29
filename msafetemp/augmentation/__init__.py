"""Template-based multilingual safety data augmentation.

Stages: template extraction and validation, native template translation, harmful fills
(TR), paraphrases, exact-fill benign pairs (OR), post-hoc cultural localization (PH),
jailbreak wrappers, dataset assembly, benign audit, and SFT response generation.
See README.md in this directory.
"""
