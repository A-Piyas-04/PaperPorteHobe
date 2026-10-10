import numpy as np

from scholargrid.acronyms import expand, mine
from scholargrid.phrase import PhraseIndex, matched_words, normalize

TITLES = ["Self-admitted technical debt detection",
          "Technical debt in machine learning systems",
          "Date parsing for technical documents",
          "Diffusion models for images"]
ABSTRACTS = ["We detect self-admitted technical debt (SATD) in code comments.",
             "Hidden costs accumulate as systems grow.",
             "We parse date strings from technical manuals.",
             "Self-admitted technical debt (SATD) is not discussed here; we study images."]


def test_normalize_folds_case_punctuation_and_plurals():
    assert normalize("Self-Admitted Technical Debts!") == "self admitted technical debt"
    assert normalize("Naïve  Networks") == "naive network"
    assert normalize("Strategies") == normalize("strategy")
    assert normalize("analysis") == "analysis"


def test_phrase_levels_title_abstract_and_nearby_words():
    idx = PhraseIndex(TITLES, ABSTRACTS)
    levels = idx.match(["technical debt"], window=8)
    assert levels[1] == 1                 # phrase in the title
    assert levels[0] == 1
    assert levels[3] == 2                 # phrase only in the abstract
    assert 2 not in levels                # neither the phrase nor both words
    near = idx.match(["technical date"], window=8)
    assert near == {2: 3}                 # both words within the window, not adjacent
    assert idx.match(["technical date"], window=2) == {}


def test_phrase_mask_and_append():
    idx = PhraseIndex(TITLES, ABSTRACTS)
    mask = np.array([False, True, True, True])
    assert set(idx.match(["technical debt"], mask)) == {1, 3}
    idx.append(["Managing technical debt"], [""])
    assert 4 in idx.match(["technical debt"])


def test_matched_words_for_highlighting():
    words = matched_words(["technical debt"], TITLES[1], ABSTRACTS[1])
    assert {"technical", "debt"} <= {w.lower() for w in words}


def test_acronyms_mined_and_expanded_both_ways():
    acr = mine([f"{t}. {a}" for t, a in zip(TITLES, ABSTRACTS)])
    assert "satd" in acr
    exp = expand("SATD", acr)
    assert "self admitted technical debt" in exp.phrases
    assert {"self-admitted technical debt", "self admitted technical debt"} & set(exp.api_phrases[1:])
    back = expand("self-admitted technical debt", acr)
    assert "satd" in back.phrases          # defined twice, at least 3 letters


def test_short_or_rare_acronyms_are_never_added():
    acr = mine(["Technical debt (TD) is costly.", "We measure technical debt (TD) again.",
                "Graph neural network (GNNX) once."])
    assert "td" not in expand("technical debt", acr).phrases
    assert "technical debt" in expand("TD", acr).phrases
    assert "gnnx" not in expand("graph neural network", acr).phrases


def test_synonyms_expand_both_ways():
    syn = {"technical debt": ["tech debt"]}
    assert "tech debt" in expand("technical debt", None, syn).phrases
    assert "technical debt" in expand("tech debt", None, syn).phrases
    assert expand("diffusion", None, syn).added == []
