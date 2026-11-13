from src.verification import normalize, verify_quote, verify_edges, split_sentences

SOURCE = (
    "Riverbank stabilization is widespread. "
    "Artificial riprap crevices significantly increased juvenile round goby survival and local density by 340% "
    "compared to natural gravel-sand banks (p < 0.001). "
    "In high-flow riffles (> 0.65 m/s) native Cottus gobio maintained competitive dominance."
)


def test_exact_quote_is_verified():
    status, quote, sim = verify_quote(
        "Artificial riprap crevices significantly increased juvenile round goby survival and local density by 340%", SOURCE
    )
    assert status == "exact" and sim == 1.0


def test_whitespace_and_glyph_differences_still_exact():
    q = "In high-flow  riffles (> 0.65\u00a0m/s)\nnative Cottus gobio maintained competitive dominance"
    assert verify_quote(q, SOURCE)[0] == "exact"


def test_paraphrase_snaps_to_real_sentence():
    status, quote, sim = verify_quote(
        "Riprap crevices significantly increased juvenile round goby survival and density by 340% versus natural gravel banks",
        SOURCE,
    )
    assert status == "snapped"
    assert quote in normalize(SOURCE)  # replaced by genuine source text


def test_invented_quote_is_unverified():
    status, quote, _ = verify_quote("Warming caused crayfish plague to collapse across all catchments.", SOURCE)
    assert status == "unverified"
    assert quote.startswith("Warming")


def test_faithfulness_is_computed_from_exact_matches_only():
    edges = [
        {"edge_id": "E1", "provenance_quote": "Riverbank stabilization is widespread."},
        {"edge_id": "E2", "provenance_quote": "Riprap crevices significantly increased juvenile round goby survival and density by 340% versus natural gravel banks"},
        {"edge_id": "E3", "provenance_quote": "Completely made up sentence about pH."},
    ]
    out, stats = verify_edges(edges, SOURCE)
    assert [e["provenance_status"] for e in out] == ["exact", "snapped", "unverified"]
    assert stats["faithfulness"] == round(1 / 3, 4)
    assert stats["total"] == 3


def test_no_edges_gives_zero_not_error():
    assert verify_edges([], SOURCE)[1]["faithfulness"] == 0.0


def test_sentence_splitter_keeps_species_abbreviations_together():
    text = "P. leniusculus increased predation (p = 0.004). Furthermore, A. astacus declined."
    assert len(split_sentences(text)) == 2


def test_capitalised_first_word_still_counts_as_exact():
    assert verify_quote("in high-flow riffles (> 0.65 m/s) native Cottus gobio maintained competitive dominance", SOURCE)[0] == "exact"
    assert verify_quote("IN HIGH-FLOW RIFFLES (> 0.65 M/S) NATIVE COTTUS GOBIO MAINTAINED COMPETITIVE DOMINANCE", SOURCE)[0] == "exact"
