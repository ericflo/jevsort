"""Cross-lingual eval: the same summaries in English, Spanish, German and Japanese. Same truth; does the judge agree
with itself across languages?

    python examples/crosslingual_eval.py generate   # fictional documents + summaries, rendered from hand-written templates
    python examples/crosslingual_eval.py verify     # recount every error / fact in every language from the text alone
    python examples/crosslingual_eval.py judge
    python examples/crosslingual_eval.py plots

Ground truth
------------
Exactly as in the verifiable eval (exact counts of planted false statements and of facts mentioned, fixed by
construction), but every document and summary exists in four languages. The four versions are rendered from
hand-written per-language sentence templates filled with the *same* invented names, numbers and planted errors, so the
truth is identical in every language by construction (no machine translation that could drift). Questions and
instructions are asked in the document's own language.

Signals: (1) accuracy vs the truth in each language; (2) cross-lingual consistency: for the same pair of summaries,
how often does a judge give the same answer in two languages? A judge whose answers depend on the language is less
trustworthy even when each language looks fine on its own.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

DATA = HERE / "data" / "crosslingual.json"
RESULT = HERE / "results" / "crosslingual_eval.json"
SEED = 20260925
N_DOCS, K, N_FACTS = 3, 12, 16
LANGS = ["en", "es", "de", "ja"]
LANG_NAME = {"en": "English", "es": "Spanish", "de": "German", "ja": "Japanese"}
SYL = ["ka", "lor", "ven", "mi", "tas", "dru", "el", "sor", "qui", "ban", "the", "zu", "rom", "ail", "fen", "ost", "ny", "gar"]

KEYS = ["founded", "city", "founder", "staff", "labs", "budget", "director", "director_year", "satellite",
        "satellite_year", "patents", "share", "journal", "partner", "award", "students"]
T = {
    "en": {"org": "{A} {B} Institute", "title": "Report on the {org}.",
           "founded": "The {org} was founded in {v}.", "city": "The {org} is headquartered in {v}.",
           "founder": "It was founded by {v}.", "staff": "It employs {v} researchers.", "labs": "It operates {v} laboratories.",
           "budget": "Its annual budget is {v} million crowns.", "director": "Its current director is {v}.",
           "director_year": "The current director took office in {v}.", "satellite": "It opened a satellite campus in {v}.",
           "satellite_year": "The satellite campus opened in {v}.", "patents": "It holds {v} patents.",
           "share": "{v} of its funding comes from private donors.",
           "journal": "Its journal, the {j} Review, publishes {v} issues a year.",
           "partner": "Its main partner university is in {v}.", "award": "It has won the {a} Prize {v} times.",
           "students": "It trains {v} doctoral students each year.",
           "filler": ["The organisation publishes an annual report describing its activities.",
                      "Its research spans several scientific disciplines.",
                      "Visitors can tour parts of the main building by appointment."],
           "q_acc": "Which summary contains fewer statements that contradict the source document?",
           "g_acc": "Compare every number, name, year and place in the summary with the source document in the state.",
           "q_comp": "Which summary mentions more of the facts in the source document?",
           "g_comp": "Count facts mentioned, whether or not they are stated correctly.",
           "objective": "You are comparing summaries of a source document.", "source": "SOURCE DOCUMENT"},
    "es": {"org": "Instituto {A} {B}", "title": "Informe sobre el {org}.",
           "founded": "El {org} fue fundado en {v}.", "city": "El {org} tiene su sede en {v}.",
           "founder": "Fue fundado por {v}.", "staff": "Emplea a {v} investigadores.", "labs": "Opera {v} laboratorios.",
           "budget": "Su presupuesto anual es de {v} millones de coronas.", "director": "Su director actual es {v}.",
           "director_year": "El director actual asumió el cargo en {v}.", "satellite": "Abrió un campus satélite en {v}.",
           "satellite_year": "El campus satélite se inauguró en {v}.", "patents": "Posee {v} patentes.",
           "share": "El {v} de su financiación procede de donantes privados.",
           "journal": "Su revista, la {j} Review, publica {v} números al año.",
           "partner": "Su principal universidad asociada está en {v}.", "award": "Ha ganado el Premio {a} {v} veces.",
           "students": "Forma a {v} doctorandos cada año.",
           "filler": ["La organización publica un informe anual sobre sus actividades.",
                      "Su investigación abarca varias disciplinas científicas.",
                      "Los visitantes pueden recorrer parte del edificio principal con cita previa."],
           "q_acc": "¿Qué resumen contiene menos afirmaciones que contradicen el documento fuente?",
           "g_acc": "Compara cada número, nombre, año y lugar del resumen con el documento fuente.",
           "q_comp": "¿Qué resumen menciona más datos del documento fuente?",
           "g_comp": "Cuenta los datos mencionados, estén o no expresados correctamente.",
           "objective": "Estás comparando resúmenes de un documento fuente.", "source": "DOCUMENTO FUENTE"},
    "de": {"org": "{A}-{B}-Institut", "title": "Bericht über das {org}.",
           "founded": "Das {org} wurde {v} gegründet.", "city": "Das {org} hat seinen Sitz in {v}.",
           "founder": "Es wurde von {v} gegründet.", "staff": "Es beschäftigt {v} Forschende.", "labs": "Es betreibt {v} Labore.",
           "budget": "Sein Jahresbudget beträgt {v} Millionen Kronen.", "director": "Derzeitiger Direktor ist {v}.",
           "director_year": "Der derzeitige Direktor trat sein Amt {v} an.", "satellite": "Es eröffnete einen Außencampus in {v}.",
           "satellite_year": "Der Außencampus wurde {v} eröffnet.", "patents": "Es hält {v} Patente.",
           "share": "{v} seiner Finanzierung stammen von privaten Spendern.",
           "journal": "Seine Zeitschrift, die {j} Review, erscheint {v}-mal im Jahr.",
           "partner": "Seine wichtigste Partneruniversität befindet sich in {v}.", "award": "Es hat den {a}-Preis {v}-mal gewonnen.",
           "students": "Es bildet jedes Jahr {v} Doktoranden aus.",
           "filler": ["Die Organisation veröffentlicht jährlich einen Tätigkeitsbericht.",
                      "Seine Forschung umfasst mehrere wissenschaftliche Disziplinen.",
                      "Besucher können Teile des Hauptgebäudes nach Vereinbarung besichtigen."],
           "q_acc": "Welche Zusammenfassung enthält weniger Aussagen, die dem Quelldokument widersprechen?",
           "g_acc": "Vergleiche jede Zahl, jeden Namen, jedes Jahr und jeden Ort mit dem Quelldokument.",
           "q_comp": "Welche Zusammenfassung erwähnt mehr Fakten aus dem Quelldokument?",
           "g_comp": "Zähle die erwähnten Fakten, unabhängig davon, ob sie korrekt wiedergegeben sind.",
           "objective": "Du vergleichst Zusammenfassungen eines Quelldokuments.", "source": "QUELLDOKUMENT"},
    "ja": {"org": "{A} {B}研究所", "title": "{org}に関する報告書。",
           "founded": "{org}は{v}年に設立された。", "city": "{org}の本部は{v}にある。",
           "founder": "創設者は{v}である。", "staff": "研究者{v}人を雇用している。", "labs": "{v}の研究室を運営している。",
           "budget": "年間予算は{v}百万クローネである。", "director": "現在の所長は{v}である。",
           "director_year": "現所長は{v}年に就任した。", "satellite": "{v}にサテライトキャンパスを開設した。",
           "satellite_year": "サテライトキャンパスは{v}年に開設された。", "patents": "{v}件の特許を保有している。",
           "share": "資金の{v}は民間の寄付者によるものである。",
           "journal": "機関誌『{j} Review』は年に{v}号発行される。",
           "partner": "主な提携大学は{v}にある。", "award": "{a}賞を{v}回受賞している。",
           "students": "毎年{v}人の博士課程学生を育成している。",
           "filler": ["同組織は活動に関する年次報告書を発行している。", "研究は複数の科学分野にまたがっている。",
                      "見学者は予約制で本館の一部を見学できる。"],
           "q_acc": "原文と矛盾する記述が少ない要約はどちらか？",
           "g_acc": "要約中のすべての数字・名前・年・場所を、状態にある原文と照合すること。",
           "q_comp": "原文の事実をより多く挙げている要約はどちらか？",
           "g_comp": "正しく述べられているかどうかにかかわらず、言及された事実を数えること。",
           "objective": "あなたは原文の要約を比較している。", "source": "原文"},
}


def _name(rng, n=2):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def _values(rng):
    def num(lo, hi):
        v = rng.randint(lo, hi)
        return str(v), str(max(1, v + rng.choice([-1, 1]) * rng.randint(max(2, v // 3), max(3, v // 2 + 2))))

    def year(v):
        return str(v), str(v + rng.choice([-1, 1]) * rng.randint(4, 15))

    def pick(pool):
        t = pool[0]
        return t, rng.choice(pool[1:])

    cities = [_name(rng, 3) for _ in range(7)]
    people = [f"{_name(rng)} {_name(rng, 3)}" for _ in range(6)]
    pct = rng.randint(12, 88)
    vals = {"founded": year(rng.randint(1931, 1989)), "city": pick(cities[:1] + cities[3:]),
            "founder": pick(people[:1] + people[2:]), "staff": num(40, 900), "labs": num(3, 40), "budget": num(12, 480),
            "director": pick(people[1:2] + people[2:]), "director_year": year(rng.randint(2001, 2021)),
            "satellite": pick(cities[1:2] + cities[3:]), "satellite_year": year(rng.randint(1990, 2015)),
            "patents": num(15, 700), "share": (f"{pct}%", f"{min(99, max(1, pct + rng.choice([-1, 1]) * rng.randint(9, 25)))}%"),
            "journal": num(2, 24), "partner": pick(cities[2:3] + cities[3:]), "award": num(2, 19), "students": num(8, 160)}
    names = {"A": _name(rng), "B": _name(rng), "j": _name(rng), "a": _name(rng)}
    return vals, names


def _tpl(lang, key, names):
    org = T[lang]["org"].format(A=names["A"], B=names["B"])
    return T[lang][key].replace("{org}", org).replace("{j}", names["j"]).replace("{a}", names["a"]), org


def _join(lang, sents):
    return "".join(sents) if lang == "ja" else " ".join(sents)


def _split(lang, text):
    if lang == "ja":
        return [s + "。" for s in text.split("。") if s]
    return [s for s in re.split(r"(?<=\.)\s+", text.strip()) if s]


def generate():
    rng = random.Random(SEED)
    docs = []
    for d in range(N_DOCS):
        vals, names = _values(rng)
        order = list(KEYS)
        filler_pos = sorted(rng.sample(range(len(KEYS) + 1), 3))
        cs = list(range(N_FACTS - K + 1, N_FACTS + 1))
        es = [n // 2 for n in range(K)]
        rng.shuffle(cs)
        rng.shuffle(es)
        plan = []
        for i, (c, e) in enumerate(zip(cs, es)):
            mentioned = rng.sample(range(N_FACTS), c)
            wrong = set(rng.sample(mentioned, e))
            seq = mentioned[:]
            rng.shuffle(seq)
            plan.append({"id": f"X{d + 1}S{i + 1:02d}", "seq": seq, "wrong": sorted(wrong), "n_mentioned": c, "n_wrong": e})
        rng.shuffle(plan)
        doc = {"doc": f"X{d + 1}", "values": vals, "names": names, "plan": plan, "lang": {}}
        for lang in LANGS:
            body = []
            for n, key in enumerate(order):
                if n in filler_pos:
                    body.append(T[lang]["filler"][filler_pos.index(n)])
                t, org = _tpl(lang, key, names)
                body.append(t.replace("{v}", vals[key][0]))
            title = T[lang]["title"].format(org=org)
            source = title + ("\n\n" if lang != "ja" else "\n\n") + _join(lang, body)
            summaries = []
            for p in plan:
                sents = [_tpl(lang, KEYS[k], names)[0].replace("{v}", vals[KEYS[k]][1 if k in p["wrong"] else 0]) for k in p["seq"]]
                summaries.append({"id": p["id"], "text": _join(lang, sents)})
            doc["lang"][lang] = {"source": source, "summaries": summaries}
        docs.append(doc)
    return {"seed": SEED, "languages": LANGS, "k": K, "docs": docs,
            "truth": {"accuracy": "-(number of statements whose value contradicts the source), identical in every language",
                      "completeness": "number of source facts mentioned, identical in every language"}}


def verify(data) -> bool:
    ok = True
    for doc in data["docs"]:
        truth = {p["id"]: (p["n_mentioned"], p["n_wrong"]) for p in doc["plan"]}
        for lang in LANGS:
            tpls = {k: _tpl(lang, k, doc["names"])[0] for k in KEYS}
            for s in doc["lang"][lang]["summaries"]:
                m = w = 0
                for sent in _split(lang, s["text"]):
                    hits = []
                    for k, t in tpls.items():
                        pre, post = t.split("{v}")
                        if sent.startswith(pre) and sent.endswith(post):
                            hits.append((k, sent[len(pre): len(sent) - len(post)]))
                    if len(hits) != 1:
                        print(f"  {lang} {s['id']}: unmapped sentence {sent!r} ({len(hits)} matches)")
                        ok = False
                        continue
                    k, v = hits[0]
                    m += 1
                    w += v != doc["values"][k][0]
                if (m, w) != truth[s["id"]]:
                    print(f"  {lang} {s['id']}: recount {(m, w)} != {truth[s['id']]}")
                    ok = False
    return ok


def cmd_judge(args):
    from jevsort import Dimension, Item, JevSorter
    from summary_showdown import _judge_backend

    data = json.loads(DATA.read_text())
    assert verify(data), "data does not verify"
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {"judges": {}}
    for spec in [s for s in args.judges.split(",") if s]:
        b = _judge_backend(spec)
        out = {}
        for lang in LANGS:
            L = T[lang]
            for doc in data["docs"]:
                D = doc["lang"][lang]
                items = [Item(s["id"], s["text"]) for s in D["summaries"]]
                ctx = f"{L['source']}:\n{D['source']}"
                dims = [Dimension("accuracy", L["q_acc"], L["g_acc"], context=ctx),
                        Dimension("completeness", L["q_comp"], L["g_comp"], context=ctx)]
                r = JevSorter(b, dims, L["objective"], pair_strategy="round_robin", coupling="bt", state_mode="pair",
                              delta=0.0).sort(items)
                out[f"{lang}|{doc['doc']}"] = {
                    "log_strength": {d: {it.id: float(r.per_dim[d].log_strength[i]) for i, it in enumerate(items)} for d in r.dims},
                    "pairs": [{k: a[k] for k in ("dim", "a", "b", "p_sym")} for a in r.audit if a["stage"] == "pair"]}
        res["judges"][spec] = {"runs": out, "usage": b.usage.as_dict()}
        print(f"  {spec}: ${b.usage.cost_usd:.3f}")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    score()


def score():
    from jevsort.metrics import kendall_tau

    data = json.loads(DATA.read_text())
    R = json.loads(RESULT.read_text())
    truth = {p["id"]: {"accuracy": -p["n_wrong"], "completeness": p["n_mentioned"]} for d in data["docs"] for p in d["plan"]}
    for spec, j in R["judges"].items():
        per_lang = {}
        picks = {}  # (doc, dim, a, b) -> {lang: pick}
        for key, run in j["runs"].items():
            lang, doc = key.split("|")
            for d in ("accuracy", "completeness"):
                ids = list(run["log_strength"][d])
                tau = kendall_tau([run["log_strength"][d][i] for i in ids], [truth[i][d] for i in ids])
                per_lang.setdefault(lang, {}).setdefault(d, []).append(tau)
            for p in run["pairs"]:
                a, b = sorted((p["a"], p["b"]))
                pa = p["p_sym"] if p["a"] == a else 1 - p["p_sym"]
                picks.setdefault((doc, p["dim"], a, b), {})[lang] = pa > 0.5
        agree = {}
        for l1, l2 in itertools.combinations(LANGS, 2):
            v = [x[l1] == x[l2] for x in picks.values() if l1 in x and l2 in x]
            agree[f"{l1}-{l2}"] = float(np.mean(v)) if v else None
        all4 = [len(set(x.values())) == 1 for x in picks.values() if len(x) == len(LANGS)]
        j["score"] = {"tau": {lang: {d: float(np.mean(v)) for d, v in dd.items()} for lang, dd in per_lang.items()},
                      "pair_agreement": agree, "same_answer_in_all_languages": float(np.mean(all4)) if all4 else None}
        s = j["score"]
        print(f"  {spec:<34} " + "  ".join(f"{lang}: {np.mean(list(s['tau'][lang].values())):.2f}" for lang in LANGS)
              + f"   same answer in all 4 languages: {s['same_answer_in_all_languages']:.0%}")
    RESULT.write_text(json.dumps(R, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("verify")
    j = sub.add_parser("judge")
    j.add_argument("--judges", default="typesafe/jev-1.13,deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning")
    sub.add_parser("score")
    sub.add_parser("plots")
    a = ap.parse_args()
    if a.cmd == "generate":
        d = generate()
        DATA.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {DATA.relative_to(HERE.parent)}: {N_DOCS} docs x {K} summaries x {len(LANGS)} languages; verifies: {verify(d)}")
    elif a.cmd == "verify":
        ok = verify(json.loads(DATA.read_text()))
        print("VERIFIED in all languages" if ok else "VERIFY FAILED")
        sys.exit(0 if ok else 1)
    elif a.cmd == "judge":
        cmd_judge(a)
    elif a.cmd == "score":
        score()
    else:
        import eval_figs

        eval_figs.crosslingual()


if __name__ == "__main__":
    main()
