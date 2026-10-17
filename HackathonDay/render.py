"""Build a self-contained HTML report from report.json.

Run: python3 HackathonDay/render.py
Check: python3 HackathonDay/render.py --check
"""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "report.json"
OUTPUT = ROOT / "index.html"


def e(value: object) -> str:
    return escape(str(value), quote=True)


def link(path: str, label: str = "Evidence ↗") -> str:
    return f'<a class="evidence" href="{e(path)}">{e(label)}</a>'


def profile_rows(profiles: list[dict]) -> str:
    rows = []
    for rank, item in enumerate(profiles, 1):
        wins, games = int(item["wins"]), int(item["games"])
        pct = 100 * wins / games
        rows.append(
            f'<tr><td class="rank">{rank:02d}</td>'
            f'<td class="profile-name">{e(item["name"])}</td>'
            f'<td class="number">{wins:,}<span class="denom"> / {games:,}</span></td>'
            f'<td class="rate"><div class="rate-line"><strong>{pct:.1f}%</strong>'
            f'<span class="bar"><span style="width:{pct:.2f}%"></span></span></div></td>'
            f'<td class="seat">{int(item["seat0_wins"]):,} / {int(item["seat1_wins"]):,}</td>'
            f'<td>{link(item["source"], "Game log ↗")}</td></tr>'
        )
    return "\n".join(rows)


STATUS_LABEL = {
    "leader": "Current leader",
    "gain": "Measured gain",
    "partial": "Partial repair",
    "mixed": "Mixed result",
    "no-gain": "No gain",
    "failed": "Failed gate",
    "diagnostic": "Diagnostic",
    "foundation": "Foundation",
}


def experiment_cards(experiments: list[dict]) -> str:
    cards = []
    for item in experiments:
        searchable = " ".join(str(item[key]) for key in (
            "title", "kind", "track", "status", "metric", "comparison", "finding", "decision"
        )).lower()
        cards.append(f"""
        <article class="experiment" data-track="{e(item['track'])}"
                 data-status="{e(item['status'])}" data-search="{e(searchable)}">
          <div class="experiment-top">
            <span class="track">{e(item['track'])}</span>
            <span class="kind">{e(item['kind'])}</span>
            <span class="status status-{e(item['status'])}">{e(STATUS_LABEL[item['status']])}</span>
          </div>
          <h3>{e(item['title'])}</h3>
          <div class="metric">{e(item['metric'])}</div>
          <p class="comparison">{e(item['comparison'])}</p>
          <p>{e(item['finding'])}</p>
          <div class="decision"><span>Decision</span>{e(item['decision'])}</div>
          {link(item['source'])}
        </article>""")
    return "\n".join(cards)


def render(data: dict) -> str:
    if data.get("schema") != 1:
        raise ValueError("Unknown report schema")
    date.fromisoformat(data["as_of"])
    profiles, experiments = data["profiles"], data["experiments"]
    if len({x["id"] for x in profiles}) != len(profiles):
        raise ValueError("Duplicate profile ID")
    if len({x["id"] for x in experiments}) != len(experiments):
        raise ValueError("Duplicate experiment ID")
    if any(x["status"] not in STATUS_LABEL for x in experiments):
        raise ValueError("Unknown experiment status")
    if any(int(x["wins"]) > int(x["games"]) for x in profiles):
        raise ValueError("Profile wins exceed games")
    if not profiles or not experiments:
        raise ValueError("Report needs both profiles and experiments")

    open_questions = "\n".join(
        f'<li><span>{i:02d}</span>{e(question)}</li>'
        for i, question in enumerate(data["open_questions"], 1)
    )
    status_options = "\n".join(
        f'<option value="{e(key)}">{e(label)}</option>'
        for key, label in STATUS_LABEL.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>{e(data['edition'])} · {e(data['title'])}</title>
  <style>
    :root {{
      --ink:#101b1a; --deep:#0a1212; --panel:#172321; --panel2:#1d2c29;
      --line:#39504a; --text:#eef3e8; --muted:#afc0b4; --mint:#b7e5cd;
      --lime:#dbf29c; --gold:#e4c580; --red:#efada5;
    }}
    *{{box-sizing:border-box}}
    html{{scroll-behavior:smooth}}
    body{{margin:0;background:var(--deep);color:var(--text);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    a{{color:var(--lime);text-decoration-thickness:1px;text-underline-offset:3px}}
    a:hover{{color:#fff}}
    .shell{{max-width:1260px;margin:auto;padding:0 30px}}
    .topbar{{border-bottom:1px solid var(--line);background:rgba(10,18,18,.95);position:sticky;top:0;z-index:10;backdrop-filter:blur(15px)}}
    .topbar-inner{{display:flex;justify-content:space-between;align-items:center;gap:20px;min-height:62px}}
    .brand{{display:flex;gap:13px;align-items:center;text-decoration:none;color:var(--text);font-weight:800;letter-spacing:.13em;text-transform:uppercase;font-size:12px}}
    .brand-mark{{display:grid;place-items:center;width:30px;height:30px;border:1px solid var(--lime);color:var(--lime);font-size:17px}}
    nav{{display:flex;gap:22px;flex-wrap:wrap}} nav a{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;text-decoration:none}} nav a:hover{{color:var(--lime)}}
    .hero{{position:relative;overflow:hidden;padding:79px 0 65px;background:radial-gradient(circle at 78% 28%,#2b4939 0,transparent 36%),linear-gradient(135deg,#172721,#0a1212 72%)}}
    .hero:after{{content:"";position:absolute;right:-180px;top:-180px;width:650px;height:650px;border:1px solid rgba(219,242,156,.13);border-radius:50%;box-shadow:0 0 0 100px rgba(219,242,156,.025),0 0 0 200px rgba(219,242,156,.018);pointer-events:none}}
    .eyebrow{{display:flex;align-items:center;gap:12px;color:var(--lime);font-size:12px;font-weight:750;letter-spacing:.22em;text-transform:uppercase}}
    .eyebrow:before{{content:"";width:33px;height:1px;background:var(--lime)}}
    h1{{font-size:clamp(47px,7vw,102px);line-height:.95;letter-spacing:-.07em;max-width:920px;margin:26px 0 25px;font-weight:820}}
    h1 span{{color:var(--lime)}}
    .lead{{font-size:clamp(17px,2vw,22px);line-height:1.47;max-width:760px;color:#d7e8d9;margin:0}}
    .hero-meta{{display:flex;gap:25px;flex-wrap:wrap;margin-top:38px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.11em}}
    .hero-meta strong{{color:var(--text);font-weight:700}}
    section{{padding:68px 0;border-top:1px solid var(--line)}}
    .section-head{{display:grid;grid-template-columns:180px 1fr;gap:28px;align-items:start;margin-bottom:30px}}
    .section-number{{color:var(--lime);font-size:12px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;padding-top:8px}}
    h2{{font-size:clamp(28px,3vw,44px);line-height:1.06;letter-spacing:-.045em;margin:0 0 12px}}
    .section-head p{{max-width:740px;color:var(--muted);margin:0}}
    .verdict{{display:grid;grid-template-columns:1.3fr .7fr;gap:18px}}
    .feature{{background:var(--panel);border:1px solid var(--line);padding:28px;min-height:230px}}
    .feature.primary{{background:linear-gradient(145deg,#243b31,#152520);border-color:#66937a}}
    .label{{color:var(--lime);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.18em}}
    .feature h3{{font-size:26px;line-height:1.12;letter-spacing:-.035em;margin:18px 0 12px}}
    .feature p{{color:var(--muted);margin:0;max-width:640px}}
    .big-number{{font-size:clamp(35px,4.4vw,62px);font-weight:850;line-height:1;letter-spacing:-.07em;color:var(--lime);margin:24px 0 12px}}
    .fact-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}}
    .fact{{padding:19px 20px;border:1px solid var(--line);background:#14211e}}
    .fact strong{{display:block;color:var(--text);font-size:18px;margin-bottom:3px}}
    .fact span{{font-size:12px;color:var(--muted)}}
    .protocol{{border-left:3px solid var(--gold);background:#25291f;padding:21px 24px;display:grid;gap:8px;margin-top:18px}}
    .protocol strong{{color:var(--gold);font-size:12px;letter-spacing:.16em;text-transform:uppercase}}
    .protocol p{{margin:0;color:#dfdfcf}}
    .table-wrap{{overflow-x:auto;border:1px solid var(--line);background:var(--panel)}}
    table{{width:100%;border-collapse:collapse;min-width:800px}}
    thead{{background:#20332c;color:var(--mint)}}
    th{{font-size:11px;text-transform:uppercase;letter-spacing:.13em;text-align:left;padding:15px 15px;white-space:nowrap}}
    td{{border-top:1px solid #31453f;padding:12px 15px;font-size:13px}}
    tbody tr:hover{{background:#21352e}} tbody tr:first-child{{background:#294433}}
    .rank,.seat,.denom{{color:var(--muted)}} .number{{white-space:nowrap;font-variant-numeric:tabular-nums}}
    .profile-name{{font-weight:650;min-width:200px}}
    .rate{{min-width:210px}} .rate-line{{display:flex;gap:12px;align-items:center}}
    .rate-line strong{{width:50px;font-variant-numeric:tabular-nums}}
    .bar{{height:7px;background:#3b4b40;border-radius:9px;flex:1;overflow:hidden}}
    .bar span{{display:block;background:linear-gradient(90deg,#719e76,var(--lime));height:100%;border-radius:9px}}
    .evidence{{font-size:12px;white-space:nowrap}}
    .toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
    .toolbar label{{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.12em;font-weight:700}}
    .toolbar input,.toolbar select{{height:42px;border:1px solid #52665b;background:#1d3029;color:var(--text);padding:0 13px;font:14px system-ui;min-width:160px}}
    .toolbar input{{min-width:min(340px,80vw)}}
    .filter-count{{margin-left:auto;align-self:end;color:var(--muted);font-size:12px;padding-bottom:9px}}
    .cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
    .experiment{{border:1px solid var(--line);background:var(--panel);padding:23px 23px 21px;display:flex;flex-direction:column;min-height:345px}}
    .experiment:hover{{border-color:#84ad8c;background:#1c2d27}}
    .experiment[hidden]{{display:none}}
    .experiment-top{{display:flex;flex-wrap:wrap;gap:7px;align-items:center}}
    .track,.kind,.status{{padding:4px 8px;font-size:10px;text-transform:uppercase;letter-spacing:.11em;font-weight:800}}
    .track{{background:#35473c;color:var(--lime)}} .kind{{background:#293934;color:#bdd0c1}}
    .status{{margin-left:auto;border:1px solid #5b7064;color:#d1e3d2}}
    .status-leader,.status-gain{{border-color:#87ab7a;color:var(--lime)}}
    .status-failed,.status-no-gain{{border-color:#a77169;color:#f0b9ae}}
    .status-mixed,.status-partial{{border-color:#a69066;color:#e7d196}}
    .experiment h3{{font-size:21px;letter-spacing:-.027em;line-height:1.15;margin:17px 0 8px}}
    .metric{{font-size:15px;font-weight:800;color:var(--lime);margin:0 0 4px}}
    .comparison{{font-size:12px;color:var(--muted);margin:0 0 12px}}
    .experiment p:not(.comparison){{margin:0 0 15px;color:#d9e3d7}}
    .decision{{margin-top:auto;border-top:1px solid var(--line);padding-top:12px;color:#dbe5d8;font-size:13px}}
    .decision span{{color:var(--gold);font-size:10px;letter-spacing:.14em;text-transform:uppercase;font-weight:800;margin-right:9px}}
    .experiment .evidence{{margin-top:13px}}
    .questions{{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}
    .questions li{{border:1px solid var(--line);background:var(--panel);padding:19px;display:flex;gap:15px}}
    .questions span{{color:var(--lime);font-weight:800;font-variant-numeric:tabular-nums}}
    footer{{border-top:1px solid var(--line);padding:30px 0 45px;color:var(--muted);font-size:12px}}
    footer .shell{{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}}
    @media(max-width:850px){{.section-head{{grid-template-columns:1fr;gap:7px}}.verdict,.cards,.questions{{grid-template-columns:1fr}}.fact-grid{{grid-template-columns:1fr 1fr}}}}
    @media(max-width:600px){{.shell{{padding:0 17px}}.topbar-inner{{align-items:start;padding-top:12px;padding-bottom:12px;flex-direction:column}}nav{{gap:13px}}.hero{{padding:58px 0}}section{{padding:49px 0}}.fact-grid{{grid-template-columns:1fr}}.filter-count{{width:100%}}.experiment{{min-height:0}}}}
    @media print{{:root{{--deep:#fff;--panel:#fff;--text:#17231c;--muted:#42544b;--line:#becbc1;--lime:#1b6a3f}}body{{background:#fff;color:#17231c}}.topbar,.toolbar{{display:none}}.hero{{background:#fff;padding:20px 0}}.feature,.fact,.experiment,.table-wrap,.questions li{{background:#fff;color:#17231c;break-inside:avoid}}.cards{{grid-template-columns:1fr 1fr}}a{{color:#235e41}}section{{padding:24px 0}}}}
  </style>
</head>
<body>
  <header class="topbar"><div class="shell topbar-inner">
    <a class="brand" href="#top"><span class="brand-mark">⬡</span> Hackathon Day / Catan Lab</a>
    <nav aria-label="Report sections"><a href="#verdict">Verdict</a><a href="#grid">20 profiles</a><a href="#ledger">Experiment ledger</a><a href="#questions">Next gates</a></nav>
  </div></header>
  <main id="top">
    <div class="hero"><div class="shell">
      <div class="eyebrow">{e(data['edition'])} / live strategy report</div>
      <h1>Build to <span>win.</span><br>Measure everything.</h1>
      <p class="lead">{e(data['headline'])}</p>
      <div class="hero-meta"><div>Report as of <strong>{e(data['as_of'])}</strong></div><div><strong>{len(profiles)}</strong> measured 1v1 profiles</div><div><strong>{len(experiments)}</strong> documented trials</div></div>
    </div></div>
    <section id="verdict"><div class="shell">
      <div class="section-head"><div class="section-number">01 / Current position</div><div><h2>The measured incumbent</h2><p>One decision for play, with every alternative kept visible as evidence.</p></div></div>
      <div class="verdict">
        <div class="feature primary"><div class="label">Agent in play</div><h3>{e(data['current_agent'])}</h3><p>{e(data['headline'])}</p></div>
        <div class="feature"><div class="label">Broad 1v1 diagnostic</div><div class="big-number">71.7%</div><p>{e(data['current_result'])}</p></div>
      </div>
      <div class="fact-grid">
        <div class="fact"><strong>291 / 412</strong><span>Outcome-Transformer hybrid; plain OneVOne 290 / 412 on the same held-out games</span></div>
        <div class="fact"><strong>50 / 50</strong><span>OneVOne versus the transferred short-trained Benjamin CPU checkpoint</span></div>
        <div class="fact"><strong>67 / 128</strong><span>Fixed-horizon trade/build planner; OneVOne 93 / 128 on the same screen</span></div>
      </div>
      <div class="protocol"><strong>Interpret the score correctly</strong><p>{e(data['rules'])}</p><p>{e(data['evaluation_note'])}</p></div>
    </div></section>
    <section id="grid"><div class="shell">
      <div class="section-head"><div class="section-number">02 / Common grid</div><div><h2>Every 1v1 profile</h2><p>Each profile played 1,028 board seeds from both seats against the same pinned Catanatron web bot. This grid compares agents on a common schedule; it does not establish an independent held-out rate.</p></div></div>
      <div class="table-wrap"><table><thead><tr><th>Rank</th><th>Profile</th><th>Wins</th><th>Win rate</th><th>Seat 0 / 1</th><th>Raw result</th></tr></thead><tbody>{profile_rows(profiles)}</tbody></table></div>
    </div></section>
    <section id="ledger"><div class="shell">
      <div class="section-head"><div class="section-number">03 / Trial ledger</div><div><h2>What worked. What failed.</h2><p>Experiments retain their own opponents, rule sets and sample sizes. Compare each candidate with its paired control inside its card.</p></div></div>
      <div class="toolbar" role="group" aria-label="Filter experiments">
        <label>Search<input id="search" type="search" placeholder="Model, move, result…" autocomplete="off"></label>
        <label>Arena<select id="track"><option value="all">All arenas</option><option value="1v1">1v1</option><option value="4-player">4-player</option></select></label>
        <label>Outcome<select id="status"><option value="all">All outcomes</option>{status_options}</select></label>
        <div class="filter-count" id="filter-count" aria-live="polite">{len(experiments)} trials shown</div>
      </div>
      <div class="cards" id="cards">{experiment_cards(experiments)}</div>
    </div></section>
    <section id="questions"><div class="shell">
      <div class="section-head"><div class="section-number">04 / Next gates</div><div><h2>Questions that change the agent</h2><p>{e(data['next_gate'])}</p></div></div>
      <ol class="questions">{open_questions}</ol>
    </div></section>
  </main>
  <footer><div class="shell"><div>Hackathon Day / Catan Agent Lab · compiled {e(data['as_of'])}</div><div>Update <code>report.json</code>, then run <code>python3 render.py</code>.</div></div></footer>
  <script>
    const cards = [...document.querySelectorAll('.experiment')];
    const controls = ['search','track','status'].map(id => document.getElementById(id));
    function filterCards() {{
      const query = controls[0].value.trim().toLowerCase();
      const track = controls[1].value;
      const status = controls[2].value;
      let visible = 0;
      for (const card of cards) {{
        const show = (!query || card.dataset.search.includes(query)) &&
          (track === 'all' || card.dataset.track === track) &&
          (status === 'all' || card.dataset.status === status);
        card.hidden = !show;
        visible += show;
      }}
      document.getElementById('filter-count').textContent = `${{visible}} of ${{cards.length}} trials shown`;
    }}
    controls.forEach(control => control.addEventListener('input', filterCards));
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify index.html matches report.json")
    args = parser.parse_args()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    output = render(data)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != output:
            raise SystemExit("index.html is stale; run python3 HackathonDay/render.py")
        print(f"Up to date: {OUTPUT}")
    else:
        OUTPUT.write_text(output, encoding="utf-8")
        print(f"Built {OUTPUT} from {len(data['profiles'])} profiles and {len(data['experiments'])} trials")


if __name__ == "__main__":
    main()
