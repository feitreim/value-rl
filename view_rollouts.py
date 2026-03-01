"""
view_rollouts.py - TUI viewer for saved GRPO rollouts
"""

import json
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Static


def find_latest_rollout(directory: str = "rollouts") -> str:
    files = sorted(Path(directory).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    assert files, f"No .jsonl files found in {directory}/"
    return str(files[-1])


def load_rollouts(path: str) -> list[dict]:
    p = Path(path)
    assert p.exists(), f"File not found: {path}"
    rollouts = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def render_completions(group: dict) -> str:
    parts = []
    n_c = len(group["completions"])
    for i, c in enumerate(group["completions"]):
        r = c["reward"]
        reward_color = "green" if r > 0 else "red" if r < 0 else "white"
        scores_str = "   ".join(f"[dim]{name}[/dim] [{reward_color}]{val:+.2f}[/{reward_color}]" for name, val in c["scores"].items())
        parts.append(
            f"[bold]{i + 1}/{n_c}[/bold]"
            f"   reward=[{reward_color}]{r:+.3f}[/{reward_color}]"
            f"   adv={c['advantage']:+.3f}\n"
            f"[dim]{scores_str}[/dim]\n\n"
            f"{c['text']}"
        )
    return "\n\n\n".join(parts)


class RolloutViewer(App):
    TITLE = "RL-Values Rollout Viewer"

    CSS = """
    Screen { layout: horizontal; }

    #steps {
        width: 40;
        height: 1fr;
        border: solid $primary-darken-2;
    }

    #right {
        width: 1fr;
        layout: vertical;
    }

    #step-bar {
        height: 1;
        background: $primary-darken-2;
        padding: 0 1;
    }

    #prompt-nav {
        height: 3;
        background: $panel;
        layout: horizontal;
    }

    #btn-prev {
        width: 14;
        min-width: 0;
    }

    #btn-next {
        width: 14;
        min-width: 0;
    }

    #prompt-counter {
        width: 1fr;
        content-align: center middle;
        background: $panel;
    }

    #info-bar {
        height: auto;
        min-height: 3;
        max-height: 8;
        background: $panel;
        padding: 0 1 1 1;
        border-bottom: solid $primary-darken-2;
    }

    #completions {
        height: 1fr;
        overflow-y: scroll;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "prev_prompt", "Prev prompt"),
        Binding("k", "next_prompt", "Next prompt"),
    ]

    def __init__(self, rollouts: list[dict], live: bool = False) -> None:
        super().__init__()
        self.rollouts = rollouts
        self.live = live
        self.step_idx = 0
        self.prompt_idx = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="steps")
            with Vertical(id="right"):
                yield Static("", id="step-bar", markup=True)
                with Horizontal(id="prompt-nav"):
                    yield Button("< Prev", id="btn-prev")
                    yield Static("", id="prompt-counter", markup=True)
                    yield Button("Next >", id="btn-next")
                yield Static("", id="info-bar", markup=True)
                yield ScrollableContainer(
                    Static("", id="completions-content", markup=True),
                    id="completions",
                )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("Step", "Loss", "Reward", "Time")
        self._populate_table()
        self._refresh_detail()
        if self.live:
            self.set_interval(2.0, self._poll_new_steps)

    def _populate_table(self) -> None:
        table = self.query_one(DataTable)
        for r in self.rollouts[table.row_count :]:
            t_total = r.get("times", {}).get("total", 0)
            table.add_row(
                str(r["step"]),
                f"{r['loss']:.4f}",
                f"{r['mean_reward']:+.4f}",
                f"{t_total:5.1f}s",
            )

    def _poll_new_steps(self) -> None:
        path = getattr(self, "_rollout_path", None)
        if path is None:
            return
        fresh = load_rollouts(path)
        if len(fresh) > len(self.rollouts):
            self.rollouts = fresh
            self._populate_table()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row is not None:
            self.step_idx = event.cursor_row
            self.prompt_idx = 0
            self._refresh_detail()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-prev":
            self.action_prev_prompt()
        elif event.button.id == "btn-next":
            self.action_next_prompt()

    def action_prev_prompt(self) -> None:
        n = len(self.rollouts[self.step_idx]["groups"])
        self.prompt_idx = (self.prompt_idx - 1) % n
        self._refresh_detail()

    def action_next_prompt(self) -> None:
        n = len(self.rollouts[self.step_idx]["groups"])
        self.prompt_idx = (self.prompt_idx + 1) % n
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        if not self.rollouts:
            return
        r = self.rollouts[self.step_idx]
        n_p = len(r["groups"])
        group = r["groups"][self.prompt_idx]

        ts = r.get("timestamp", "")[:19].replace("T", " ")
        t = r.get("times", {})
        timing_str = ""
        if t:
            timing_str = f" [dim]rollout:{t.get('rollout', 0):.1f}s score:{t.get('score', 0):.1f}s grad:{t.get('grad_step', 0):.1f}s[/dim]"

        self.query_one("#step-bar", Static).update(
            f"step {r['step']}   loss={r['loss']:.4f}   reward={r['mean_reward']:+.4f}{timing_str}   [dim]{ts}[/dim]"
        )

        self.query_one("#prompt-counter", Static).update(f"[bold]Prompt {self.prompt_idx + 1} / {n_p}[/bold]")

        category = group.get("category", "")
        notes = group.get("notes", "")
        lines = []
        if category or notes:
            meta = "  ".join(
                filter(
                    None,
                    [
                        f"[bold]{category}[/bold]" if category else "",
                        f"[dim]{notes}[/dim]" if notes else "",
                    ],
                )
            )
            lines.append(meta)
            lines.append("")
        lines.append(group["prompt"])
        self.query_one("#info-bar", Static).update("\n".join(lines))

        self.query_one("#completions-content", Static).update(render_completions(group))
        self.query_one("#completions", ScrollableContainer).scroll_home(animate=False)


def main() -> None:
    args = sys.argv[1:]
    live = "--live" in args
    args = [a for a in args if a != "--live"]
    path = args[0] if args else find_latest_rollout()

    rollouts = load_rollouts(path)
    print(f"Loading: {path}")
    assert rollouts, f"No rollouts found in {path}"

    app = RolloutViewer(rollouts, live=live)
    app._rollout_path = path
    app.run()


if __name__ == "__main__":
    main()
