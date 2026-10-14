"""Serve a local browser UI for playing against a saved Connect Four model."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

from connect_four_engine import PLAYER_ONE, ConnectFour, IllegalMoveError

from .agent import TransformerAgent


PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"
DEFAULT_CHECKPOINT = PACKAGE_DIR / "artifacts" / "checkpoints" / "candidate_tactical_v5_5min.pt"


class GameSession:
    def __init__(self, agent: TransformerAgent, human: str) -> None:
        self.agent = agent
        self.lock = threading.Lock()
        self.reset(human)

    def reset(self, human: str) -> None:
        if human not in {"X", "O"}:
            raise ValueError("human must be X or O")
        self.human_piece = human
        self.human_player = PLAYER_ONE if human == "X" else -PLAYER_ONE
        self.game = ConnectFour()
        self.last_decision = None
        self.last_move = None
        if self.game.current_player != self.human_player:
            self._bot_turn()

    def _bot_turn(self) -> None:
        decision = self.agent.inspect(self.game)
        action = int(decision["selected_action"])
        self.game.step(action)
        self.last_move = {"column": action, "player": "X" if -self.human_player == PLAYER_ONE else "O"}
        self.last_decision = decision

    def play_human(self, column: int) -> None:
        if self.game.is_terminal:
            raise ValueError("the game is already over; start a new game")
        if self.game.current_player != self.human_player:
            raise ValueError("wait for the bot to finish its move")
        result = self.game.step(column)
        self.last_move = {"column": column, "player": self.human_piece}
        self.last_decision = None
        if not result.is_terminal:
            self._bot_turn()

    def payload(self) -> dict:
        player_symbol = lambda player: None if player is None else ("X" if player == PLAYER_ONE else "O")
        if self.game.is_terminal:
            status = "draw" if self.game.is_draw else "win"
            message = "Draw game" if self.game.is_draw else f"{player_symbol(self.game.winner)} wins"
        elif self.game.current_player == self.human_player:
            status, message = "human", "Your turn"
        else:
            status, message = "bot", "Bot is thinking…"
        return {
            "board": [self.game.board[row * 7:(row + 1) * 7] for row in range(6)],
            "current_player": player_symbol(self.game.current_player) if not self.game.is_terminal else None,
            "human_piece": self.human_piece,
            "bot_piece": "O" if self.human_piece == "X" else "X",
            "human_turn": not self.game.is_terminal and self.game.current_player == self.human_player,
            "legal_columns": list(self.game.legal_actions()),
            "move_count": self.game.move_count,
            "status": status,
            "message": message,
            "winner": player_symbol(self.game.winner),
            "draw": self.game.is_draw,
            "winning_line": [list(cell) for cell in self.game.winning_line],
            "last_move": self.last_move,
            "decision": self.last_decision,
        }


def make_handler(session: GameSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ConnectFourLocal/1.0"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: dict) -> None:
            self._send(status, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in {"/api/state", "/api/health"}:
                with session.lock:
                    value = {"ok": True} if path.endswith("health") else session.payload()
                self._json(200, value)
                return
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/styles.css": ("styles.css", "text/css; charset=utf-8")}
            asset = assets.get(path)
            if asset is None:
                self._json(404, {"error": "not found"})
                return
            filename, content_type = asset
            self._send(200, (WEB_DIR / filename).read_bytes(), content_type)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("request is too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                with session.lock:
                    if path == "/api/new":
                        session.reset(body.get("human_piece", "X"))
                    elif path == "/api/move":
                        column = body.get("column")
                        if isinstance(column, bool) or not isinstance(column, int):
                            raise ValueError("column must be an integer from 1 to 7")
                        if column not in range(1, 8):
                            raise ValueError("column must be from 1 to 7")
                        session.play_human(column - 1)
                    else:
                        self._json(404, {"error": "not found"})
                        return
                    payload = session.payload()
                self._json(200, payload)
            except (ValueError, json.JSONDecodeError, IllegalMoveError) as error:
                self._json(400, {"error": str(error)})
            except Exception as error:  # Keep the local server alive and return a useful failure.
                self._json(500, {"error": f"bot error: {error}"})

        def log_message(self, format: str, *args) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1", help="bind locally by default")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--human", choices=("X", "O"), default="X")
    parser.add_argument("--no-lookahead", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    agent = TransformerAgent(args.checkpoint, tactical_guard=not args.no_lookahead)
    session = GameSession(agent, args.human)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Human: {args.human} | Bot: {'O' if args.human == 'X' else 'X'}")
    print(f"Open http://{args.host}:{args.port} in Chrome (Ctrl-C to stop).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local Connect Four server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
