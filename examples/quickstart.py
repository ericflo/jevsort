"""The 60-second quickstart, as a script. Every snippet in docs/quickstart.md is here, verbatim.

    export OPENROUTER_API_KEY=sk-or-...
    python examples/quickstart.py
"""

import pairsort

ideas = [
    "A browser extension that summarizes long email threads",
    "A CLI that turns any CSV into a chart in one command",
    "A to-do app with a blockchain backend",
    "A tool that ranks pull requests by review urgency",
    "A smart fridge magnet that tweets your grocery list",
]

# 1. One question, one line.
result = pairsort.sort(ideas, "Which side project would developers find most useful?")
print(result.best)            # the winner (your own string back)
print(result.top(3))          # the three best
print(result.scores)          # {id: probability of being the best}

# 2. Several questions, blended: name each one as a keyword.
result = pairsort.sort(ideas, useful="Which would developers find more useful?",
                              easy="Which is easier to build in a weekend?")
# (Same thing as a dict: pairsort.sort(ideas, {"useful": "...", "easy": "..."}). Use the dict when a question's
#  name would clash with an option such as judge= or budget=.)
# To make one question count double: useful=("Which would developers find more useful?", 2)
print(result)                 # a table: overall + each question

# 3. One comparison.
p = pairsort.compare("Short, clear sentences.",
                    "Sentences that, owing to a proliferation of subordinate clauses, meander.",
                    "Which is easier to read?")
print(f"P(first is easier to read) = {p:.2f}")

# 4. Your own judge: any function (question, a, b) -> P(a is better). No API key needed.
by_length = pairsort.sort(ideas, "Which is shorter?", judge=lambda q, a, b: len(a) < len(b))
print(by_length.best)

# 5. Everything else is still there: builder, budgets, meta-judge, calibration, any backend.
result = (pairsort.sorter()
          .by("Which would developers find more useful?")
          .objective("We are picking one weekend hackathon project.")
          .judge("llm:deepseek/deepseek-v4.1-flash")   # any OpenRouter model, Jev, or a local model
          .budget(8)                                    # at most 8 comparisons
          .meta()                                       # let the judge re-rank the top items overall
          .sort(ideas))
print(result.ids, result.usage["pairs"], "pairs")
