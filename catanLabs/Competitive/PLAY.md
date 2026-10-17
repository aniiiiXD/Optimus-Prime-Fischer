# Play against the benchmark incumbent

`HYDE_FAIRVALUE` exposes the measured FairValue policy in Catanatron's human
play UI. It loads the promoted `fair_army_finish.json` configuration and the
same pinned Catanatron v5 rules/protocol used by the lab's benchmarks.

## Start the game

One-time setup, from the repository root:

```bash
uv pip install --python start/.venv/bin/python -e 'start/catan_lab/vendor/catanatron[web]'
(cd start/catan_lab/vendor/catanatron/ui && npm ci --engine-strict=false)
```

Then start both servers:

```bash
./catanLabs/Competitive/play.sh
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000). Choose **Human** for one
seat and **Hyde FairValue (measured incumbent)** for a bot seat; the built-in
Random, Weighted Random, Victory Point, and Catanatron bots are available too.
To match the
current benchmark, set the map to **BASE**, points to win to **10**, discard
limit to **7**, and Friendly Robber to **off**. Start the game, then follow the
board prompts to place your first settlement and road.

Use Ctrl+C in the terminal to stop both servers. The upstream Catanatron
checkout remains unchanged; the bot registration and launch commands live in
this lab.
