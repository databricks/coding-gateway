"""Rich/questionary presentation primitives. No project knowledge."""

from __future__ import annotations

import itertools
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal

import questionary
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.dimension import Dimension
from questionary.prompts.common import InquirerControl
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

# Past this many options the choice list is pinned to a fixed-height scrolling viewport (see
# `_cap_choice_viewport`) rather than growing to fill the terminal, and the pickers append a
# "↑/↓ scroll" note to their instruction line. The value is both the boundary and the number of
# rows shown at once; matches mcp.py's MCP_PICKER_VISIBLE_ROWS so both picker families agree.
_SCROLL_HINT_THRESHOLD = 10


def _with_scroll_hint(instruction: str, option_count: int) -> str:
    """Append a scroll affordance when the option list is too long to fully show at once.

    The hint is discoverability only — the viewport already scrolls; users just can't tell there is
    more below the fold. Short lists are left alone so the instruction stays terse.
    """
    if option_count > _SCROLL_HINT_THRESHOLD:
        return f"{instruction[:-1]}, ↑/↓ scroll)" if instruction.endswith(")") else instruction
    return instruction


def _cap_choice_viewport(question: questionary.Question, option_count: int) -> None:
    """Pin a long choice list to a fixed ``_SCROLL_HINT_THRESHOLD``-row scrolling window.

    questionary sizes the choice window to its content, bounded only by the terminal, so a tall
    terminal shows every option and a short one scrolls — the visible count depends on the window
    size. Capping the window height makes the picker show at most ``_SCROLL_HINT_THRESHOLD`` rows and
    scroll through the rest, keeping the highlighted row in view (``InquirerControl`` emits a cursor
    token at the pointer, which prompt_toolkit's ``Window`` tracks). Short lists are left untouched so
    they render at their natural height with no empty rows.

    Best-effort: it mutates questionary's internal layout, so any shape change (missing application,
    no ``InquirerControl`` window) leaves the picker as-is rather than raising — the list still works,
    it just falls back to terminal-fit sizing.
    """
    if option_count <= _SCROLL_HINT_THRESHOLD:
        return
    application = getattr(question, "application", None)
    if application is None:
        return
    visible = Dimension(preferred=_SCROLL_HINT_THRESHOLD, max=_SCROLL_HINT_THRESHOLD)
    for window in application.layout.find_all_windows():
        if isinstance(window, Window) and isinstance(window.content, InquirerControl):
            window.height = visible
            return


# Output verbosity. "normal" (default) renders decorative panels; "low" trades
# them for terse single-line output. Set once at CLI entry via set_verbosity.
_verbosity = "normal"


def set_verbosity(value: str) -> None:
    global _verbosity
    _verbosity = value or "normal"


def get_verbosity() -> str:
    return _verbosity


def is_low_verbosity() -> bool:
    return _verbosity == "low"


def print_section(title: str) -> None:
    console.print()
    console.print(Panel(title, style="bold blue", expand=False))


def print_heading(text: str) -> None:
    console.print()
    console.print(f"[bold]{text}[/bold]")


def print_kv(key: str, val: str) -> None:
    console.print(f"  [bold]{key}:[/bold] [cyan]{val}[/cyan]")


def kv_line(key: str, val: str) -> str:
    """A `print_kv`-styled line, returned instead of printed, for collecting into a panel.

    The value is markup-escaped. Rich reads bracketed text as a style tag and renders nothing for
    it, so a policy name of ``[prod] tiered routing`` displayed as ``tiered routing`` in the config
    summary — the one block an admin reads to confirm what they are about to publish workspace-wide.
    Values here include admin-typed free text (policy name, skills locations, tracing table).
    """
    return f"[bold]{escape(key)}:[/bold] [cyan]{escape(val)}[/cyan]"


def print_panel(title: str, lines: list[str]) -> None:
    """Render `lines` inside a titled box.

    Unlike :func:`print_section`, which boxes a bare title, this boxes the body — so a block that
    should be read as one unit (a config summary an admin is about to publish) reads as one, rather
    than as loose lines that blend into whatever the flow printed before it.
    """
    console.print()
    console.print(Panel("\n".join(lines), title=title, style="blue", expand=False))


def print_warning_panel(message: str, *, title: str = "Warning") -> None:
    """Render a warning as a boxed panel, for a blocker that should stand out from inline notes.

    A `!` marker keeps it visually of a kind with :func:`print_warning`, but the box gives a
    dead-end message (e.g. "no budgets exist, nothing to do") the weight to be read before the flow
    exits, rather than scrolling past as one more line.
    """
    console.print()
    console.print(
        Panel(f"[bold yellow]![/bold yellow] {message}", title=title, style="yellow", expand=False)
    )


def print_note(text: str) -> None:
    console.print(f"[dim]•[/dim] {text}")


def print_success(message: str) -> None:
    console.print(f"[bold green]✔[/bold green] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_warning_err(message: str) -> None:
    """``print_warning`` on stderr, for when stdout is a machine-read stream."""
    err_console.print(f"[bold yellow]![/bold yellow] {message}")


def print_err(message: str) -> None:
    err_console.print(f"[bold red]ERROR[/bold red] {message}")


def heading(text: str) -> str:
    return f"[bold blue]{text}[/bold blue]"


def label(text: str) -> str:
    return f"[bold]{text}[/bold]"


def value(text: str) -> str:
    return f"[cyan]{text}[/cyan]"


def muted(text: str) -> str:
    return f"[dim]{text}[/dim]"


def status_badge(text: str, kind: str) -> str:
    color = {"ok": "green", "warn": "yellow", "error": "red", "info": "blue"}.get(kind, "bold")
    return f"[bold {color}]{text}[/bold {color}]"


@contextmanager
def spinner(message: str | Callable[[], str]):
    """Show a spinner while the block runs. `message` may be a callable, which
    is re-evaluated on every frame so callers can render live progress (e.g. a
    running count) during a long operation."""
    if not sys.stdout.isatty():
        yield
        return

    if isinstance(message, str):
        static_message = message

        def current_message() -> str:
            return static_message
    else:
        current_message = message

    stop_event = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            # `\033[K` erases to end of line so a shrinking dynamic message
            # doesn't leave stale characters behind.
            sys.stdout.write(f"\r\033[2m{frame}\033[0m {current_message()}\033[K")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


@contextmanager
def progress_bar(description: str, total: int) -> Iterator[Callable[[], None]]:
    """Yield an ``advance()`` callback that drives a ``k/n`` progress bar.

    Falls back to no live bar off a tty (e.g. CI), so logs stay single-line.
    """
    if total <= 0 or not sys.stdout.isatty():
        yield lambda: None
        return

    with Progress(
        TextColumn("[dim]{task.description}[/dim]"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.advance(task)


def render_box_table(
    headers: list[str],
    rows: list[list[str]],
    max_widths: list[int] | None = None,
) -> str:
    wrapped_rows: list[list[list[str]]] = []
    widths = [len(header) for header in headers]

    for row in rows:
        wrapped_row: list[list[str]] = []
        for index, cell in enumerate(row):
            raw_cell = cell if cell else "-"
            width_limit = max_widths[index] if max_widths and index < len(max_widths) else None
            if width_limit:
                cell_lines = textwrap.wrap(raw_cell, width=width_limit) or ["-"]
            else:
                cell_lines = raw_cell.splitlines() or ["-"]
            wrapped_row.append(cell_lines)
            widths[index] = max(widths[index], max(len(line) for line in cell_lines))
        wrapped_rows.append(wrapped_row)

    top = "┏" + "┳".join("━" * (w + 2) for w in widths) + "┓"
    header = "┃ " + " ┃ ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " ┃"
    middle = "┡" + "╇".join("━" * (w + 2) for w in widths) + "┩"
    bottom = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    body_lines: list[str] = []
    for wrapped_row in wrapped_rows:
        row_height = max(len(cell_lines) for cell_lines in wrapped_row)
        for line_index in range(row_height):
            body_lines.append(
                "│ "
                + " │ ".join(
                    (
                        wrapped_row[col][line_index] if line_index < len(wrapped_row[col]) else ""
                    ).ljust(widths[col])
                    for col in range(len(headers))
                )
                + " │"
            )

    return "\n".join([top, header, middle, *body_lines, bottom])


def format_token_count(token_count: int) -> str:
    value_float = float(token_count)
    if token_count >= 1_000_000_000:
        return f"{value_float / 1_000_000_000:.1f}B"
    if token_count >= 1_000_000:
        return f"{value_float / 1_000_000:.1f}M"
    if token_count >= 1_000:
        return f"{value_float / 1_000:.1f}K"
    return str(token_count)


def format_usd(amount: Decimal) -> str:
    return f"${amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def format_cost_usd(amount: Decimal) -> str:
    """Like `format_usd`, but keeps more precision for sub-cent per-model costs.

    A model that cost a fraction of a cent would round to ``$0.00`` at two
    decimals and read as free, so amounts under a cent show four decimals.
    """
    if Decimal(0) < amount < Decimal("0.01"):
        return f"${amount.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)}"
    return format_usd(amount)


def format_meter(fraction: float, width: int = 30) -> str:
    """Text meter for `fraction` of a whole, clamped to [0, 1]."""
    clamped = min(max(fraction, 0.0), 1.0)
    filled = int(clamped * width)
    # A small-but-real fraction shouldn't read as empty.
    if clamped > 0:
        filled = max(filled, 1)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_duration(duration_value: timedelta | None) -> str:
    if not duration_value or duration_value.total_seconds() <= 0:
        return "-"
    total_minutes = duration_value.total_seconds() / 60
    if total_minutes < 60:
        return f"{int(round(total_minutes))}m"
    total_hours = total_minutes / 60
    if total_hours < 10:
        return f"{total_hours:.1f}h"
    if total_hours < 24:
        return f"{round(total_hours):.0f}h"
    return f"{total_hours / 24:.1f}d"


def normalize_workspace_url(workspace: str) -> str:
    workspace = workspace.strip()
    if not workspace:
        raise ValueError("Workspace URL cannot be empty.")
    if not workspace.startswith(("http://", "https://")):
        workspace = f"https://{workspace}"
    return workspace.rstrip("/")


def prompt_for_workspace(
    description: str,
    profiles: list[tuple[str, str]] | None = None,
) -> tuple[str, str | None]:
    """Ask the user for a workspace URL, offering profiles as quick-select.

    `profiles` is a list of (host_url, profile_name) tuples. Caller fetches
    them — `ui.py` stays Databricks-agnostic. Duplicate hosts (multiple
    profiles pointing at the same workspace) are shown separately; the picker
    returns the exact (host, profile_name) the user selected. Returns
    ``(url, profile_name)``; profile_name is ``None`` when the user typed a
    URL manually.
    """
    console.print()

    if profiles:
        name_header = "Profile Name"
        url_header = "Workspace URL"
        # Clamp so a single very long profile name can't push the URL column
        # off-screen on an 80-col terminal — questionary doesn't wrap row
        # titles cleanly, and a wrapped row breaks the picker visually.
        max_name_width = 40
        name_width = min(
            max_name_width,
            max(len(name_header), *(len(name) for _, name in profiles)),
        )
        # Match the 2-char cursor gutter so the header line aligns with rows.
        header_title = f"  {name_header.ljust(name_width)}  {url_header}"
        choices: list[questionary.Choice | questionary.Separator] = [
            questionary.Separator(header_title)
        ]
        for host, profile_name in profiles:
            display_name = (
                profile_name
                if len(profile_name) <= name_width
                else profile_name[: name_width - 1] + "…"
            )
            row_title = f"{display_name.ljust(name_width)}  {host}"
            # Value carries the full untruncated profile name so downstream
            # `--profile` calls always use the real name, not the display form.
            choices.append(questionary.Choice(title=row_title, value=(host, profile_name)))
        choices.append(questionary.Choice(title="Enter a different URL", value=None))
        style = questionary.Style(
            [
                ("highlighted", "fg:cyan bold"),
                ("pointer", "fg:cyan bold"),
                ("answer", "fg:cyan"),
                ("separator", "fg:white bold"),
            ]
        )
        choice = questionary.select(
            "Select workspace:", choices=choices, style=style, pointer="›", qmark=""
        ).ask()
        if isinstance(choice, tuple):
            host, profile_name = choice
            return normalize_workspace_url(host), profile_name

    while True:
        raw_value = console.input(f"  [bold]Workspace URL[/bold] {muted('›')} ").strip()
        try:
            return normalize_workspace_url(raw_value), None
        except ValueError as exc:
            print_err(str(exc))


def prompt_for_tools(
    available: list[tuple[str, str]],
    preselected: list[str] | set[str] | None = None,
    prompt: str = "Select coding agents to configure:",
) -> list[str]:
    """Multi-select picker for coding agents.

    `available` is [(tool_id, display_name), ...]. Returns the chosen tool_ids.
    When ``preselected`` is None every option is checked by default, so hitting
    Enter selects everything; pass a subset to pre-check only those (e.g. the
    agents an existing managed config already enables). Returns [] if the user
    submits an empty selection.
    """
    style = questionary.Style(
        [
            # Theme-agnostic picker: every row renders in the terminal's
            # default foreground colour (`noinherit` strips the
            # prompt_toolkit defaults that would otherwise re-colour the
            # cursor row or every checked row).
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    preselected_set = {str(item) for item in preselected} if preselected is not None else None
    choices = [
        questionary.Choice(
            title=display,
            value=tool_id,
            checked=(preselected_set is None or tool_id in preselected_set),
        )
        for tool_id, display in available
    ]
    answer = questionary.checkbox(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction="(space to toggle, enter to confirm)",
    ).ask()
    return list(answer) if answer else []


def prompt_for_multi_selection(
    prompt: str,
    options: list[tuple[str, str]],
    preselected: list[str] | set[str] | None = None,
    *,
    searchable: bool = False,
) -> list[str] | None:
    """Multi-select picker over arbitrary `(value, label)` options.

    Distinct from :func:`prompt_for_tools`, which is agent-specific and defaults to
    everything checked: here nothing is checked unless ``preselected`` says so, since
    an admin picking models wants an explicit choice rather than "all of them".
    Returns the chosen values, [] on an empty submission, or None if cancelled
    (Ctrl-C) so callers can distinguish "chose nothing" from "aborted".

    ``searchable`` lets the user narrow a long list by typing; see
    :func:`prompt_for_selection` for why it trades away j/k navigation.
    """
    style = questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    preselected_set = {str(item) for item in preselected} if preselected is not None else set()
    choices = [
        questionary.Choice(title=option_label, value=value, checked=value in preselected_set)
        for value, option_label in options
    ]
    instruction = "(space to toggle, enter to confirm)"
    if searchable:
        instruction = "(type to filter, space to toggle, enter to confirm)"
    instruction = _with_scroll_hint(instruction, len(options))
    question = questionary.checkbox(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction=instruction,
        use_search_filter=searchable,
        use_jk_keys=not searchable,
    )
    _cap_choice_viewport(question, len(options))
    answer = question.ask()
    return None if answer is None else list(answer)


def prompt_for_text(
    prompt: str, *, default: str | None = None, required: bool = False
) -> str | None:
    """Free-text prompt, used when model discovery found nothing to pick from.

    Returns the trimmed input, ``default`` on an empty answer, or None when there is no
    default and the user submits nothing (or closes stdin).

    ``required=True`` raises ``KeyboardInterrupt`` on closed stdin instead of returning None, for
    callers that loop until they get a value: returning None to such a caller spins forever on a
    piped or exhausted stdin. Matches :func:`prompt_for_percentage`, which has no default and does
    the same.

    A default is shown as ``[value] (enter to accept)`` rather than the bare ``[value]``: bracketed
    text alone reads as a format example as easily as a value that will be used, so it invited
    retyping what pressing enter would already pick.

    The whole bracketed hint is markup-escaped, brackets included. Rich reads
    ``[coding-agents-tiered-routing]`` as a style tag and prints nothing for it, so an unescaped
    word-like default vanished from the prompt entirely — numeric ones like ``[80]`` are not valid
    tags and survived, which is why this looked fine wherever it was checked.
    """
    hint = f" {escape(f'[{default}]')} (enter to accept)" if default else ""
    while True:
        try:
            raw_value = console.input(f"{label(prompt)}{muted(hint)} {muted('›')} ").strip()
        except EOFError as exc:
            if required:
                raise KeyboardInterrupt from exc
            return default
        if raw_value:
            return raw_value
        if default is not None:
            return default
        print_err("Please enter a value.")


def prompt_for_percentage(prompt: str, *, default: float | None = None) -> float:
    """Prompt for a percentage (0-100) and return it as a fraction in [0, 1].

    Budget tiers are fractions in the API (the server validates 0..1), but admins think in
    percent — and the spec's own prose says "80%". Prompting in percent and converting here
    keeps that mismatch in one place instead of at every call site.

    No caller passes ``default`` today, and tier thresholds deliberately have none: a threshold
    decides when developers get downgraded, so it should be typed rather than accepted by accident.
    The hint is still formatted (and escaped) the same way :func:`prompt_for_text` formats its own,
    so the two cannot drift if a default is ever introduced.

    Raises ``KeyboardInterrupt`` on closed stdin when there is no default — see the handler below.
    """
    hint = f" {escape(f'[{default * 100:g}]')} (enter to accept)" if default is not None else ""
    while True:
        try:
            raw_value = console.input(f"{label(prompt)}{muted(hint)} {muted('› ')}").strip()
        except EOFError as exc:
            if default is not None:
                return default
            # Closed stdin with no default to fall back on is the admin abandoning the prompt, which
            # is what Ctrl-C means here too. Raised as KeyboardInterrupt so the CLI's existing
            # handler prints "Interrupted." and exits 130; a bare EOFError has no handler anywhere
            # above this and reached the admin as a traceback.
            raise KeyboardInterrupt from exc
        if not raw_value and default is not None:
            return default
        try:
            percent = float(raw_value.rstrip("%"))
        except ValueError:
            print_err("Please enter a number between 0 and 100.")
            continue
        if 0 <= percent <= 100:
            return percent / 100
        print_err("Please enter a number between 0 and 100.")


def prompt_for_selection(
    prompt: str, options: list[tuple[str, str]], *, searchable: bool = False
) -> str | None:
    """Single-select arrow-key picker. `options` is [(value, label), ...].

    The prompt renders above the choices (questionary convention). Returns the
    chosen value, or None if the user cancels (Ctrl-C / empty).

    ``searchable`` lets the user narrow a long list by typing. It costs j/k navigation — questionary
    rejects both at once, since j and k are also search characters — so it is opt-in for the pickers
    that are actually long (model and budget lists), leaving short ones on plain arrow keys.
    """
    style = questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    choices = [questionary.Choice(title=label, value=value) for value, label in options]
    instruction = "(type to filter, arrow keys to move)" if searchable else "(use arrow keys)"
    instruction = _with_scroll_hint(instruction, len(options))
    question = questionary.select(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction=instruction,
        use_search_filter=searchable,
        use_jk_keys=not searchable,
    )
    _cap_choice_viewport(question, len(options))
    answer = question.ask()
    return answer


def prompt_yes_no(prompt: str) -> bool:
    while True:
        response = console.input(f"{label(prompt)} {muted('(y/n)')} {muted('›')} ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_err("Please answer yes or no.")


def prompt_yes_no_default(prompt: str, *, default: bool) -> bool:
    """Empty answer or closed stdin (EOF) takes ``default`` (no abort on piped runs)."""
    hint = "(Y/n)" if default else "(y/N)"
    while True:
        try:
            response = console.input(f"{label(prompt)} {muted(hint)} {muted('›')} ").strip().lower()
        except EOFError:
            return default
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_err("Please answer yes or no.")


def prompt_for_choice(prompt: str, options: list[tuple[str, str]]) -> str:
    console.print()
    for index, (_, option_label) in enumerate(options, start=1):
        console.print(f"  [bold]{index}.[/bold] [cyan]{option_label}[/cyan]")

    while True:
        raw_value = console.input(f"{label(prompt)} {muted('›')} ").strip()
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(options):
                return options[selected_index - 1][0]
        print_err("Please enter a valid option number.")


def prompt_for_client_id() -> str:
    while True:
        client_id = console.input(f"{label('OAuth client ID')} {muted('›')} ").strip()
        if client_id:
            return client_id
        print_err("Client ID cannot be empty.")


def prompt_for_client_secret() -> str:
    while True:
        client_secret = console.input(f"{label('OAuth client secret')} {muted('›')} ").strip()
        if client_secret:
            return client_secret
        print_err("Client secret cannot be empty.")
