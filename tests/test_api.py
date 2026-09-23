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



def test_questions_as_keywords():
    r = pairsort.sort(["a", "bbb", "cc"], useful="Which is longer?", easy="Which is shorter?", judge=longer)
    assert r.dims == ["useful", "easy"]
    r = pairsort.sort(["a", "bbb", "cc"], "Which is longer?", extra="Which is weirder?", judge=longer, seed=2)
    assert r.dims == ["longer", "extra"] and r.config["objective"] == ""


def test_keyword_collisions_and_typos():
    # a question named like an option goes through the dict form
    assert pairsort.sort(["a", "bb"], by={"judge": "Which would a judge prefer?"}, judge=longer).dims == ["judge"]
    # options are never mistaken for questions, even string-valued ones
    r = pairsort.sort(["a", "bb", "ccc"], useful="Which is longer?", coupling="bt", judge=longer)
    assert r.dims == ["useful"] and r.per_dim["useful"].method == "bt"
    with pytest.raises(TypeError, match="max_pairs"):
        pairsort.sort(["a", "b"], "Q?", judge=longer, max_pair=5)
    with pytest.raises(TypeError, match="misspelled option 'fusion'"):
        pairsort.sort(["a", "b"], "Q?", judge=longer, fusoin="linear+meta")
    assert {"judge", "max_pairs", "budget", "objective", "fusion", "coupling"} <= pairsort.reserved_names()


def test_blended_questions_show_per_question_ranks_and_weights():
    ideas = ["a", "bbbb", "cc", "ddd"]
    r = pairsort.sort(ideas, long="Which is longer?", short="Which is shorter?", judge=lambda q, a, b: (len(a) > len(b)) == ("longer" in q))
    t = r.table()
    assert "P(best)" in t and "long" in t and "short" in t and "#1" in t and "item" not in t.split("\n")[2]
    assert abs(sum(r.shares().values()) - 1) < 1e-9 and abs(r.shares()["long"] - 0.5) < 1e-9
    assert r.rows()[0]["ranks"]["long"] in (1, 4)


def test_question_weights_python_and_cli():
    judge = lambda q, a, b: (len(a) > len(b)) == ("longer" in q)  # noqa: E731
    ideas = ["a", "bbbb", "cc", "ddd"]
    r = pairsort.sort(ideas, long=("Which is longer?", 3), short="Which is shorter?", judge=judge)
    assert r.best == "bbbb" and abs(r.shares()["long"] - 0.75) < 1e-9
    r = pairsort.sort(ideas, {"long": "Which is longer?", "short": {"question": "Which is shorter?", "weight": 3}}, judge=judge)
    assert r.best == "a" and abs(r.shares()["short"] - 0.75) < 1e-9
    from argparse import Namespace

    from pairsort.cli import _dims
    d = _dims(Namespace(questions=['useful:2=Which is more useful?', 'easy=Which is easier?', "Which is cheaper?"], dim=None))
    assert [(x.name, x.weight) for x in d] == [("useful", 2.0), ("easy", 1.0), ("cheaper", 1.0)]


def test_cli_rejects_shell_split_questions():
    r = subprocess.run([sys.executable, "-m", "pairsort", "sort", "-", "useful=Which", "is", "more", "useful?"],
                       input="a\nb\n", capture_output=True, text=True)
    assert r.returncode != 0 and "split your quotes" in r.stderr
