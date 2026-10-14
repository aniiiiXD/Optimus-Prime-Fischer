#!/usr/bin/env python3
"""Build the compact MAV Transformer takeaway PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "papers" / "MAV_transformer_hackathon_takeaways.pdf"
PAGE_W, PAGE_H = A4

INK = colors.HexColor("#152338")
MUTED = colors.HexColor("#53657A")
BLUE = colors.HexColor("#276EF1")
CYAN = colors.HexColor("#13B8A6")
AMBER = colors.HexColor("#F2A93B")
RED = colors.HexColor("#D95C59")
PAPER = colors.HexColor("#F6F8FB")
WHITE = colors.white
LINE = colors.HexColor("#D9E1EB")
PALE_BLUE = colors.HexColor("#EAF1FF")
PALE_CYAN = colors.HexColor("#E7F8F5")
PALE_AMBER = colors.HexColor("#FFF5DF")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("TakeawaySans", regular))
            pdfmetrics.registerFont(TTFont("TakeawaySansBold", bold))
            return "TakeawaySans", "TakeawaySansBold", "Helvetica-Bold"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Bold"


FONT, FONT_BOLD, FONT_DISPLAY = register_fonts()


def stylesheet():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName=FONT_DISPLAY, fontSize=28,
            leading=31, textColor=WHITE, alignment=TA_LEFT, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName=FONT, fontSize=11.5,
            leading=16, textColor=colors.HexColor("#D9E7FF"),
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=19,
            leading=23, textColor=INK, spaceAfter=6,
        ),
        "kicker": ParagraphStyle(
            "Kicker", parent=base["Normal"], fontName=FONT_BOLD, fontSize=8.5,
            leading=10, textColor=BLUE, tracking=1.2, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=FONT, fontSize=9.2,
            leading=13.2, textColor=INK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName=FONT, fontSize=7.7,
            leading=10.5, textColor=MUTED,
        ),
        "card_title": ParagraphStyle(
            "CardTitle", parent=base["Normal"], fontName=FONT_BOLD, fontSize=10,
            leading=12, textColor=INK, spaceAfter=3,
        ),
        "card_body": ParagraphStyle(
            "CardBody", parent=base["Normal"], fontName=FONT, fontSize=8.2,
            leading=11.2, textColor=MUTED,
        ),
        "equation": ParagraphStyle(
            "Equation", parent=base["Normal"], fontName="Courier-Bold", fontSize=8.3,
            leading=12, textColor=INK, alignment=TA_CENTER,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["Normal"], fontName=FONT_BOLD, fontSize=7.8,
            leading=9.5, textColor=WHITE,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["Normal"], fontName=FONT, fontSize=7.5,
            leading=9.3, textColor=INK,
        ),
        "number": ParagraphStyle(
            "Number", parent=base["Normal"], fontName=FONT_BOLD, fontSize=17,
            leading=18, textColor=BLUE, alignment=TA_CENTER,
        ),
    }
    return styles


S = stylesheet()


class ArchitectureFlow(Flowable):
    def __init__(self, width=170 * mm, height=57 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        boxes = [
            (0, "STATE", "board/history"),
            (1, "TRANSFORMER", "shared representation"),
            (2, "MULTI-ACTION", "Q(s,a) for candidates"),
            (3, "DECISION", "policy / search"),
        ]
        gap = 6 * mm
        box_w = (self.width - 3 * gap) / 4
        y = 15 * mm
        for i, title, subtitle in boxes:
            x = i * (box_w + gap)
            fill = [PALE_BLUE, PALE_CYAN, PALE_AMBER, colors.HexColor("#F0ECFF")][i]
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, box_w, 27 * mm, 4 * mm, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont(FONT_BOLD, 8.6)
            c.drawCentredString(x + box_w / 2, y + 17 * mm, title)
            c.setFillColor(MUTED)
            c.setFont(FONT, 7.2)
            c.drawCentredString(x + box_w / 2, y + 10 * mm, subtitle)
            if i < 3:
                ax = x + box_w + 1.2 * mm
                ay = y + 13.5 * mm
                c.setStrokeColor(BLUE)
                c.setLineWidth(1.3)
                c.line(ax, ay, ax + gap - 2.4 * mm, ay)
                c.line(ax + gap - 4 * mm, ay + 1.6 * mm, ax + gap - 2.4 * mm, ay)
                c.line(ax + gap - 4 * mm, ay - 1.6 * mm, ax + gap - 2.4 * mm, ay)
        c.setFont(FONT_BOLD, 8)
        c.setFillColor(BLUE)
        c.drawString(0, 48 * mm, "THE PAPER'S CORE INTERFACE")
        c.setFont(FONT, 7.4)
        c.setFillColor(MUTED)
        c.drawRightString(self.width, 48 * mm, "one request can return state + legal moves + values + best move")


class SearchFlow(Flowable):
    def __init__(self, width=170 * mm, height=53 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        columns = [
            ("DIRECT", "1 forward pass", "fastest", BLUE),
            ("EXTERNAL SEARCH", "MCTS calls model", "strongest / costly", CYAN),
            ("INTERNAL SEARCH", "generate search trace", "one long call", AMBER),
        ]
        gap = 6 * mm
        bw = (self.width - 2 * gap) / 3
        for i, (title, mid, low, color) in enumerate(columns):
            x = i * (bw + gap)
            c.setFillColor(WHITE)
            c.setStrokeColor(color)
            c.setLineWidth(1.5)
            c.roundRect(x, 9 * mm, bw, 33 * mm, 4 * mm, fill=1, stroke=1)
            c.setFillColor(color)
            c.rect(x, 34 * mm, bw, 8 * mm, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont(FONT_BOLD, 8.2)
            c.drawCentredString(x + bw / 2, 36.5 * mm, title)
            c.setFillColor(INK)
            c.setFont(FONT_BOLD, 9.5)
            c.drawCentredString(x + bw / 2, 24 * mm, mid)
            c.setFillColor(MUTED)
            c.setFont(FONT, 7.7)
            c.drawCentredString(x + bw / 2, 16 * mm, low)
        c.setFont(FONT_BOLD, 8)
        c.setFillColor(BLUE)
        c.drawString(0, 47 * mm, "THREE INFERENCE MODES")


def P(text, style="body"):
    return Paragraph(text, S[style])


def card(title, body, color=PALE_BLUE, width=82 * mm):
    table = Table([[P(title, "card_title")], [P(body, "card_body")]], colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
    ]))
    return table


def bullets(items, size="body"):
    return [P(f"<b>{i + 1}.</b> {item}", size) for i, item in enumerate(items)]


def page_header(kicker, title, intro=None):
    flow = [P(kicker.upper(), "kicker"), P(title, "h1")]
    if intro:
        flow.append(P(intro, "body"))
    flow.append(Spacer(1, 3 * mm))
    return flow


def draw_page(canvas, doc):
    page = canvas.getPageNumber()
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(INK)
    canvas.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, fill=1, stroke=0)
    canvas.setFont(FONT_BOLD, 7.5)
    canvas.setFillColor(WHITE)
    canvas.drawString(18 * mm, PAGE_H - 8.5 * mm, "MAV TRANSFORMER / HACKATHON TAKEAWAY")
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7.3)
    canvas.drawString(18 * mm, 8 * mm, "Source: Schultz et al., arXiv:2412.12119v3 (2025)")
    canvas.drawRightString(PAGE_W - 18 * mm, 8 * mm, f"{page} / 5")
    canvas.restoreState()


def build_story():
    story = []

    # Page 1
    story.extend(page_header("01 / Thesis", "A Transformer can be policy, value function, and world model", "The paper's useful idea is not its 2.7B-parameter scale. It is the <b>multi-action interface</b>: evaluate several legal actions and the resulting state in one structured model call."))
    story.append(ArchitectureFlow())
    story.append(Spacer(1, 3 * mm))
    row = Table([
        [card("WORLD MODEL", "Track the board, apply an action, detect terminal states.", PALE_BLUE), card("ACTION VALUES", "Predict Q(s,a) for several candidate moves in one response.", PALE_CYAN)],
        [card("POLICY", "Select the best action and learn tie-breaking, not only legality.", PALE_AMBER), card("PLANNING", "Use values inside MCTS or distill a search trace into the model.", colors.HexColor("#F0ECFF"))],
    ], colWidths=[84 * mm, 84 * mm], rowHeights=[32 * mm, 32 * mm], hAlign="LEFT")
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    story.append(row)
    story.append(Spacer(1, 3 * mm))
    story.append(card("OUR TRANSLATION", "10 Tic-Tac-Toe state tokens -> shared Transformer -> candidate-move representations -> policy score per candidate + value. The legal-action mask stays in the trusted game engine.", colors.HexColor("#EAF1FF"), 170 * mm))
    story.append(PageBreak())

    # Page 2
    story.extend(page_header("02 / Learning targets", "Five capabilities, one shared representation", "For each state s and legal action a, the model should learn both what happens next and how good the choice is."))
    capability_data = [
        [P("CAPABILITY", "table_head"), P("TARGET", "table_head"), P("WHY IT MATTERS", "table_head")],
        [P("Transition", "table"), P("T(s,a) -> s'", "table"), P("Supports rollouts and catches state-tracking errors.", "table")],
        [P("Legality", "table"), P("A<sub>legal</sub>(s)", "table"), P("Separates knowing rules from choosing strong moves.", "table")],
        [P("Action value", "table"), P("Q(s,a)", "table"), P("Ranks alternatives by eventual outcome, not imitation frequency.", "table")],
        [P("Best action", "table"), P("argmax over candidates", "table"), P("Breaks ties and prevents aimless play in won positions.", "table")],
        [P("Terminal result", "table"), P("win / draw / loss", "table"), P("Stops search correctly and anchors value labels.", "table")],
    ]
    table = Table(capability_data, colWidths=[36 * mm, 43 * mm, 88 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F1F4F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 5 * mm))
    story.append(P("PAPER-SPECIFIC TRICKS", "kicker"))
    tricks = Table([
        [card("64 VALUE BUCKETS", "Predict a distribution over discrete win-probability buckets; decode with its expectation. The paper reports mean scoring beating mode/max scoring.", PALE_CYAN),
         card("RANDOM CANDIDATE ORDER", "Shuffle move order and vary requested top-k so the model cannot depend on a fixed slot or earlier emitted candidates.", PALE_AMBER)],
    ], colWidths=[84 * mm, 84 * mm])
    tricks.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(tricks)
    story.append(Spacer(1, 5 * mm))
    story.append(P("Our compact supervised objective", "card_title"))
    story.append(card("LOSS", "L = 1.0 CE(policy) + 1.0 CE(Q-buckets) + 0.25 MSE(value) + 0.25 BCE(legality)<br/><br/>Mask input/prompt tokens from the generative loss. If using a scalar Q head, replace bucket CE with Huber loss.", colors.HexColor("#EEF2F7"), 170 * mm))
    story.append(PageBreak())

    # Page 3
    story.extend(page_header("03 / Dataset", "Use a solver as a teacher; spend data on decisions", "MAV is supervised distillation at enormous scale. Our CPU project needs the same label structure with a much smaller, deliberately diverse state set."))
    numbers = Table([
        [P("3.1B", "number"), P("1.2B", "number"), P("21.8M", "number"), P("125.6M", "number")],
        [P("Chess positions", "small"), P("Chess960 positions", "small"), P("Connect Four positions", "small"), P("Hex positions", "small")],
    ], colWidths=[42 * mm] * 4)
    numbers.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WHITE), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8)]))
    story.append(numbers)
    story.append(Spacer(1, 3 * mm))
    story.append(P("Paper scale, not our requirement. Connect Four was generated from 2,458,293 epsilon-greedy games (epsilon = 0.4), with best actions labeled by the Fhourstones minimax solver. Chess used Stockfish; Hex used neurobenzene.", "small"))
    story.append(Spacer(1, 5 * mm))
    story.append(P("OUR DATA FACTORY", "kicker"))
    data_steps = [
        "Sample legal states from random, heuristic, solver, and adversarial trajectories.",
        "Enumerate every legal candidate and compute a solver/minimax/MCTS value.",
        "Store best action, per-action value, final outcome, depth/budget, and teacher version.",
        "Canonicalize and cache states so repeated runs never recompute the same labels.",
        "Apply valid board symmetries; transform state and action together.",
        "Replay high-regret mistakes, tactical traps, rare endings, and underrepresented openings.",
    ]
    for item in bullets(data_steps):
        story.append(item)
    story.append(Spacer(1, 3 * mm))
    schema = [[P(k, "table_head") for k in ["state", "candidates", "Q targets", "policy", "result", "metadata"]],
              [P(v, "table") for v in ["canonical board", "legal actions", "value/action", "best or visits", "-1 / 0 / +1", "teacher, depth, seed"]]]
    schema_table = Table(schema, colWidths=[28 * mm] * 6)
    schema_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 1), (-1, 1), WHITE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(schema_table)
    story.append(Spacer(1, 5 * mm))
    story.append(card("MOST IMPORTANT BIAS TEST", "Train with shuffled candidate order, then evaluate every position under several permutations. A strategy is real only if the chosen move and values remain stable.", PALE_AMBER, 170 * mm))
    story.append(PageBreak())

    # Page 4
    story.extend(page_header("04 / Planning", "Direct policy first; search only when it buys strength", "The paper studies both external search and search distilled into the model. Their compute profiles are very different."))
    story.append(SearchFlow())
    story.append(Spacer(1, 3 * mm))
    story.append(P("External MCTS prior used in the paper", "card_title"))
    story.append(card("TOP-K + EXPLORATION", "P(a|s) = (1 - epsilon) softmax(Q(s,a) / temperature) over top-k moves + epsilon / |A<sub>legal</sub>|<br/><br/>Reported tuned values: temperature = 0.1, k = 5, epsilon = 0.05; asynchronous leaf batch = 16.", PALE_CYAN, 170 * mm))
    story.append(Spacer(1, 4 * mm))
    findings = Table([
        [P("EVIDENCE", "table_head"), P("IMPLEMENTATION LESSON", "table_head")],
        [P("100 MCTS simulations improved chess by 68 internal Elo over searchless MAV.", "table"), P("Even shallow search can matter; sweep budget rather than assuming more is always affordable.", "table")],
        [P("Connect Four agents gained at least 244 Elo with MCTS.", "table"), P("Use Connect Four to test whether learned values actually guide planning.", "table")],
        [P("Tree reuse saved 5-10% of model calls.", "table"), P("Retain the subtree after each played move; cache canonical states and batched leaf evaluations.", "table")],
        [P("Internal-search strength rose with generated token budget.", "table"), P("Internal search is controllable but long autoregressive traces are unattractive for our CPU-first baseline.", "table")],
    ], colWidths=[70 * mm, 98 * mm])
    findings.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), 0.5, LINE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F1F4F8")]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(findings)
    story.append(Spacer(1, 4 * mm))
    story.append(card("CPU RULE", "Compare direct policy, alpha-beta, and MCTS under the same wall-clock budget. The winning metric is strength per second, not simulations alone.", colors.HexColor("#F0ECFF"), 170 * mm))
    story.append(PageBreak())

    # Page 5
    story.extend(page_header("05 / Build plan", "Five experiments that turn the paper into our engine", "Each stage must improve a measurable quantity before we add the next layer of complexity."))
    plan = [
        ("1", "POLICY BASELINE", "Candidate tokens + legal mask. Beat random and heuristic players; inspect top-3 moves."),
        ("2", "ADD ACTION VALUES", "Train Q buckets or scalar Q. Measure regret and calibration, not only exact move match."),
        ("3", "FIX BIAS", "Shuffle candidates, augment symmetries, add hard-example replay, and run trap suites."),
        ("4", "ADD SEARCH", "Shallow alpha-beta/MCTS with batched leaves, state cache, and tree reuse."),
        ("5", "CPU OPTIMIZE", "Profile batch 1, threads, precision, quantization, compile/ONNX, and cache hit rate."),
    ]
    rows = []
    for n, title, body in plan:
        rows.append([P(n, "number"), P(f"<b>{title}</b><br/>{body}", "body")])
    pt = Table(rows, colWidths=[18 * mm, 148 * mm])
    pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WHITE), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(pt)
    story.append(Spacer(1, 5 * mm))
    story.append(P("ABLATION SCOREBOARD", "kicker"))
    ab = Table([
        [P("RUN", "table_head"), P("CHANGE", "table_head"), P("STRENGTH", "table_head"), P("SYSTEMS", "table_head")],
        [P("A", "table"), P("policy only", "table"), P("win rate, top-3, regret", "table"), P("p50/p95 latency, RSS", "table")],
        [P("B", "table"), P("+ Q/value targets", "table"), P("calibration, win-rate delta", "table"), P("head overhead", "table")],
        [P("C", "table"), P("+ symmetry + hard replay", "table"), P("trap-suite delta", "table"), P("training time", "table")],
        [P("D", "table"), P("+ fixed-budget search", "table"), P("Elo / strength per second", "table"), P("calls, cache hits, move time", "table")],
        [P("E", "table"), P("+ INT8 / optimized runtime", "table"), P("policy/value parity", "table"), P("speedup and memory delta", "table")],
    ], colWidths=[15 * mm, 49 * mm, 57 * mm, 47 * mm])
    ab.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), 0.5, LINE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F1F4F8")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(ab)
    story.append(Spacer(1, 4 * mm))
    story.append(card("THE ONE-SENTENCE OBJECTIVE", "Learn values for every legal candidate in one pass, then prove that each extra millisecond of search or optimization buys measurable playing strength.", PALE_BLUE, 170 * mm))

    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=14 * mm,
        title="MAV Transformer Hackathon Takeaways",
        author="Prepared from Schultz et al. (2025)",
        subject="Strategies and objectives for a CPU-first game decision Transformer",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="takeaway", frames=[frame], onPage=draw_page)])
    doc.build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
