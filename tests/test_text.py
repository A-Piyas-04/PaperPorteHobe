from scholargrid.text import (arxiv_version, base_arxiv_id, has_category, is_english,
                              normalize_text, title_key)


def test_base_id_strips_version_and_url():
    assert base_arxiv_id("2501.01234v3") == "2501.01234"
    assert base_arxiv_id("http://arxiv.org/abs/cs/0101001v1") == "cs/0101001"
    assert arxiv_version("2501.01234v3") == 3
    assert arxiv_version("2501.01234") == 1


def test_normalize_strips_latex_and_whitespace():
    raw = "We study $\\mathcal{O}(n)$ methods with \\emph{sparse}\n  attention~layers."
    out = normalize_text(raw)
    assert "\\" not in out and "$" not in out
    assert "sparse attention layers" in out
    assert "  " not in out


def test_title_key_ignores_case_and_punctuation():
    assert title_key("Attention Is All You Need!") == title_key("attention is all you need")


def test_category_matching_is_per_token():
    assert has_category("cs.LG stat.ML")
    assert not has_category("physics.comp-ph math.NA")
    assert not has_category("econ.cs.like")  # substring would match, token prefix does not
    assert has_category("math.OC cs.SY", whitelist=["cs.SY"])
    assert not has_category("cs.LG", whitelist=["cs.CL"])


def test_language_filter():
    en = "We propose a new method for the analysis of graphs and show that it improves results on benchmarks."
    de = "Wir schlagen eine neue Methode zur Analyse von Graphen vor und zeigen Verbesserungen auf Benchmarks."
    zh = "我们提出了一种新的图分析方法，并在多个基准上取得了改进。"
    assert is_english(en)
    assert not is_english(de)
    assert not is_english(zh)
