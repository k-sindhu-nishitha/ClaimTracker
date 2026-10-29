from claimtrackr.chunking import split_text
from claimtrackr.rules import find_exclusion, names_match, parse_amount


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))
    chunks = split_text(text, size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # overlap: the end of one chunk reappears at the start of the next
    assert chunks[0].split()[-1] in chunks[1]


def test_short_text_is_single_chunk_and_empty_is_empty():
    assert split_text("hello world", 1000, 200) == ["hello world"]
    assert split_text("   ", 1000, 200) == []


def test_excluded_conditions_are_flagged():
    for text in ["HIV", "Alzheimer\u2019s", "Parkinson's Disease Treatment", "pregnancy check", "Substance abuse"]:
        assert find_exclusion(text) is not None, text


def test_normal_conditions_are_not_flagged():
    for text in ["Bodyache with fever, cold and could", "Heart disease", "Fractured wrist", "Back pain with existing ..."]:
        assert find_exclusion(text) is None, text


def test_generic_word_disease_no_longer_causes_false_positive():
    # original code: cosine("Heart disease", "Parkinson's disease") = 0.5 > 0.4 -> wrongly rejected
    assert find_exclusion("Heart disease") is None


def test_names_match_is_fuzzy_but_not_loose():
    assert names_match("Surbhit", "Surbhit")
    assert names_match("Surbhit Kumar", "Surbhit")
    assert names_match("Surbhit", "Surbhit K")
    assert not names_match("Rahul Sharma", "Surbhit")


def test_parse_amount_handles_messy_values():
    assert parse_amount("Rs. 3,150.00") == 3150.0
    assert parse_amount(3150) == 3150.0
    assert parse_amount("n/a") is None and parse_amount(None) is None
