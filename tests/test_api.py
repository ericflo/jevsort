import subprocess
import sys

import pytest

import pairsort
from pairsort import Dimension, as_dimensions, as_items


def longer(question, a, b):  # a toy judge: longer text wins
    return len(a) > len(b)


def test_one_liner_returns_the_callers_objects():
    items = ["hi", "hello there", "hey", "good morning to you"]
    r = pairsort.sort(items, "Which greeting is warmer?", judge=longer)
    assert r.sorted == ["good morning to you", "hello there", "hey", "hi"]
    assert r.best == "good morning to you" and r.top(2) == r.sorted[:2] and r[0] == r.best
    assert list(r) == r.sorted and len(r) == 4
    assert list(r.scores) == r.ids and abs(sum(r.scores.values()) - 1) < 1e-9
    assert "greeting_warmer" in repr(r)


def test_dicts_keep_identity_and_fields():
    items = [{"id": "x", "title": "Short", "abstract": "a"}, {"id": "y", "title": "A much longer title", "abstract": "b"}]
    r = pairsort.sort(items, "Which is longer?", judge=longer)
    assert r.ids == ["y", "x"] and r.sorted[0] is items[1]
    r2 = pairsort.sort({"a": "x", "b": "xxx", "c": "xx"}, "Which is longer?", judge=longer)
    assert r2.ids == ["b", "c", "a"] and r2.sorted == ["xxx", "xx", "x"]


@pytest.mark.parametrize("by, names", [
    ("Which idea would have more impact?", ["idea_impact"]),
    (["Which is clearer?", "Which is clearer?"], ["clearer", "clearer_2"]),
    ({"impact": "Which matters more?", "fun": {"question": "Which is funnier?", "guidance": "jokes"}}, ["impact", "fun"]),
    ([{"question": "Which is better?"}], ["quality"]),
    ([("speed", "Which is faster?")], ["speed"]),
    (Dimension("x", "Which is x?"), ["x"]),
    ("papers", ["evidence", "relevance", "contribution"]),
])
def test_questions_in_any_shape(by, names):
    assert [d.name for d in as_dimensions(by)] == names


def test_missing_question_is_a_clear_error():
    with pytest.raises(ValueError, match="say what to sort by"):
        pairsort.sort(["a", "b"], judge=longer)


def test_items_from_a_file_use_its_objective_and_preset(tmp_path):
    p = tmp_path / "items.txt"
    p.write_text("one\nthree\ntwo2\n")
    r = pairsort.sort(str(p), "Which is longer?", judge=longer)
    assert r.sorted[0] == "three"
    its, originals, extra = as_items("examples/data/papers.json")
    assert len(its) == 16 and extra["preset"] == "papers"


def test_builder_reaches_every_option():
    r = (pairsort.sorter().by("Which is longer?", brevity="Which is shorter?").judge(longer)
         .objective("strings").budget(4).strategy("active").adaptive(False).sort(["a", "bbb", "cc", "dddd", "eeeee"]))
    assert r.usage["pairs"] == 4 and r.dims == ["longer", "brevity"] and r.config["objective"] == "strings"
    assert r.config["pair_strategy"] == "active" and r.config["adaptive"] is False


def test_power_keywords_pass_through():
    r = pairsort.sort(["a", "bb", "ccc", "dddd"], "Which is longer?", judge=longer, coupling="bt", both_orders=False, seed=3)
    assert r.config["both_orders"] is False and r.per_dim["longer"].method == "bt"


def test_compare_and_function_judge_forms():
    assert pairsort.compare("long text", "short", "Which is longer?", judge=longer) > 0.99
    assert pairsort.compare("a", "bb", judge=lambda q, a, b: "B" if len(b) > len(a) else "A") < 0.01
    seen = []

    def with_context(question, a, b, context=None):
        seen.append(context)
        return 0.7

    pairsort.sort(["x", "y"], "Which?", judge=with_context, objective="be kind")
    assert seen[0] == "be kind"


def test_pairsorter_accepts_the_easy_forms_too():
    r = pairsort.PairSorter(longer, "Which is longer?").sort(["a", "ccc", "bb"])
    assert r.sorted == ["ccc", "bb", "a"]


def test_hard_votes_use_bradley_terry_not_saturated_eq7():
    r = pairsort.sort(["a", "bb", "ccc", "dddd"], "Which is longer?", judge=longer)
    assert r.per_dim["longer"].method == "bt"
    assert sorted(r.scores.values())[-2] > 0.05  # the runner-up keeps a meaningful probability


def test_cli_sort_stdin_and_errors(tmp_path):
    exe = [sys.executable, "-m", "pairsort"]
    r = subprocess.run(exe + ["sort", "-"], input="a\nb\n", capture_output=True, text=True)
    assert r.returncode != 0 and "say what to sort by" in r.stderr
    r = subprocess.run(exe + ["sort", "nope.txt", "Which?"], capture_output=True, text=True)
    assert r.returncode != 0 and "no such file" in r.stderr
    r = subprocess.run(exe + ["sort", "--help"], capture_output=True, text=True)
    assert "QUESTION" in r.stdout and "--budget" in r.stdout and "--top" in r.stdout

