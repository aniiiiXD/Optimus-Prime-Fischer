"""Render the fresh Hackathon Day experiment log as a standalone HTML page."""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "report.json"
OUTPUT = ROOT / "index.html"
STATUSES = {
    "documented": "Documented",
    "measured": "Measured baseline",
    "planned": "Planned",
    "running": "Running",
    "improved": "Improved",
    "no-gain": "No gain",
    "failed": "Failed",
    "inconclusive": "Inconclusive",
}
REQUIRED = ("id", "date", "title", "status", "objective", "setup", "result", "learning", "next_step")


def safe(value: object) -> str:
    return escape(str(value), quote=True)


def validate(data: dict) -> None:
    if data.get("schema") != 2:
        raise ValueError("Expected report schema 2")
    date.fromisoformat(data["started"])
    entries = data["entries"]
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    ids = set()
    for item in entries:
        for field in REQUIRED:
            if field not in item:
                raise ValueError(f"Entry missing {field}")
        if item["id"] in ids:
            raise ValueError(f"Duplicate entry id: {item['id']}")
        ids.add(item["id"])
        date.fromisoformat(item["date"])
        if item["status"] not in STATUSES:
            raise ValueError(f"Unknown status: {item['status']}")


def render(data: dict) -> str:
    validate(data)
    entries = data["entries"]
    cards = []
    for item in reversed(entries):
        evidence = item.get("evidence")
        evidence_html = (
            f'<a class="evidence" href="{safe(evidence)}">Open evidence ↗</a>'
            if evidence else ""
        )
        searchable = " ".join(str(item[k]) for k in REQUIRED).lower()
        cards.append(f"""
        <article class="card" data-status="{safe(item['status'])}" data-search="{safe(searchable)}">
          <div class="card-top"><time>{safe(item['date'])}</time><span class="badge {safe(item['status'])}">{safe(STATUSES[item['status']])}</span></div>
          <h3>{safe(item['title'])}</h3>
          <dl>
            <dt>Question</dt><dd>{safe(item['objective'])}</dd>
            <dt>Setup</dt><dd>{safe(item['setup'])}</dd>
            <dt>Result</dt><dd>{safe(item['result'])}</dd>
            <dt>Learning</dt><dd>{safe(item['learning'])}</dd>
            <dt>Next</dt><dd>{safe(item['next_step'])}</dd>
          </dl>
          {evidence_html}
        </article>""")
    cards_html = "\n".join(cards)
    empty = (
        '<div class="empty">No experiments logged yet. The next run becomes entry 01.</div>'
        if not entries else ""
    )
    options = "".join(
        f'<option value="{safe(k)}">{safe(v)}</option>' for k, v in STATUSES.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>{safe(data['title'])} · Catan Lab</title>
  <style>
    :root {{ --bg:#0b1412; --panel:#172622; --line:#3a5046; --text:#eef4eb; --muted:#b6c8bd; --accent:#d8f49c; --gold:#f0cf8c; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
    .wrap {{ max-width:1100px; margin:auto; padding:0 26px }}
    header {{ border-bottom:1px solid var(--line); padding:20px 0; color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.18em; text-transform:uppercase }}
    .hero {{ padding:86px 0 76px; background:radial-gradient(circle at 85% 20%,#2a4936,transparent 35%) }}
    .eyebrow {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.2em; text-transform:uppercase }}
    h1 {{ font-size:clamp(54px,9vw,110px); line-height:.94; letter-spacing:-.075em; margin:22px 0 24px; max-width:850px }}
    h1 span {{ color:var(--accent) }}
    .lead {{ max-width:680px; font-size:21px; color:#d8e6d8; margin:0 }}
    .meta {{ display:flex; gap:30px; flex-wrap:wrap; margin-top:35px; font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted) }}
    .meta strong {{ color:var(--text) }}
    section {{ border-top:1px solid var(--line); padding:65px 0 }}
    .section-head {{ display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:24px }}
    h2 {{ font-size:clamp(30px,4vw,47px); letter-spacing:-.05em; line-height:1.05; margin:0 }}
    .count {{ color:var(--accent); font-weight:750; font-size:14px; white-space:nowrap }}
    .focus {{ background:linear-gradient(140deg,#253f30,#172622); border:1px solid #658571; padding:30px; max-width:800px }}
    .focus label {{ color:var(--accent); font-size:11px; font-weight:800; letter-spacing:.15em; text-transform:uppercase }}
    .focus p {{ font-size:23px; line-height:1.3; margin:12px 0 0 }}
    .toolbar {{ display:flex; gap:12px; flex-wrap:wrap; margin:0 0 20px }}
    input,select {{ background:#1b3026; color:var(--text); border:1px solid var(--line); padding:11px 13px; font:inherit }}
    input {{ min-width:min(340px,100%) }}
    .cards {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px }}
    .card {{ background:var(--panel); border:1px solid var(--line); padding:26px; display:flex; flex-direction:column }}
    .card[hidden] {{ display:none }}
    .card-top {{ display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:12px; letter-spacing:.08em }}
    .badge {{ color:var(--gold); text-transform:uppercase; font-weight:800 }}
    .badge.improved {{ color:var(--accent) }}
    .badge.failed,.badge.no-gain {{ color:#eeb0a5 }}
    h3 {{ font-size:24px; line-height:1.12; letter-spacing:-.035em; margin:24px 0 18px }}
    dl {{ margin:0 }}
    dt {{ color:var(--gold); font-size:11px; letter-spacing:.15em; text-transform:uppercase; font-weight:800; margin-top:17px }}
    dd {{ margin:3px 0 0; color:#e0e9dc }}
    .evidence {{ color:var(--accent); margin-top:26px }}
    .empty {{ grid-column:1/-1; padding:60px 30px; border:1px dashed #678071; color:var(--muted); text-align:center; background:#14221c }}
    footer {{ border-top:1px solid var(--line); color:var(--muted); padding:30px 0; font-size:13px }}
    @media(max-width:700px) {{ .wrap{{padding:0 18px}} .hero{{padding:64px 0}} section{{padding:48px 0}} .cards{{grid-template-columns:1fr}} .section-head{{align-items:start;flex-direction:column}} }}
  </style>
</head>
<body>
  <header><div class="wrap">⬡ &nbsp; Hackathon Day / Catan Lab</div></header>
  <main>
    <div class="hero"><div class="wrap">
      <div class="eyebrow">Fresh experiment log</div>
      <h1>Build to <span>win.</span><br>Log the evidence.</h1>
      <p class="lead">{safe(data['objective'])}</p>
      <div class="meta"><div>Started <strong>{safe(data['started'])}</strong></div><div><strong>{len(entries)}</strong> log entries</div></div>
    </div></div>
    <section><div class="wrap">
      <div class="section-head"><h2>Current focus</h2></div>
      <div class="focus"><label>What we are doing now</label><p>{safe(data['current_focus'])}</p></div>
    </div></section>
    <section><div class="wrap">
      <div class="section-head"><h2>Experiment log</h2><div class="count" id="count">{len(entries)} entries</div></div>
      <div class="toolbar"><input id="search" type="search" placeholder="Search experiments" aria-label="Search experiments"><select id="status" aria-label="Filter by status"><option value="all">All statuses</option>{options}</select></div>
      <div class="cards" id="cards">{cards_html}{empty}</div>
    </div></section>
  </main>
  <footer><div class="wrap">Update report.json, then run python3 HackathonDay/render.py.</div></footer>
  <script>
    const cards = [...document.querySelectorAll('.card')];
    const search = document.getElementById('search');
    const status = document.getElementById('status');
    function filter() {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const show = (!query || card.dataset.search.includes(query)) && (status.value === 'all' || card.dataset.status === status.value);
        card.hidden = !show;
        visible += show;
      }}
      document.getElementById('count').textContent = `${{visible}} of ${{cards.length}} entries`;
    }}
    search.addEventListener('input', filter);
    status.addEventListener('change', filter);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify index.html matches report.json")
    args = parser.parse_args()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    html = render(data)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != html:
            raise SystemExit("index.html is stale; run python3 HackathonDay/render.py")
        print(f"Up to date: {OUTPUT}")
    else:
        OUTPUT.write_text(html, encoding="utf-8")
        print(f"Built {OUTPUT} from {len(data['entries'])} entries")


if __name__ == "__main__":
    main()
