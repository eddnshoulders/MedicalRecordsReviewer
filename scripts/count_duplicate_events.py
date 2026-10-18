#!/usr/bin/env python3
"""Count repeated event objects in an extraction JSON file.

Accepts a plain {"events": [...]} object, a top-level event list, or an
LM Studio chat-completions response whose message content contains the JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class EventList(list):
    def __init__(self, values=(), *, recovered=False, incomplete=False, finish_reason=None,
                 output_tokens=None, content_empty=False, reasoning_length=0):
        super().__init__(values)
        self.recovered = recovered
        self.incomplete = incomplete
        self.finish_reason = finish_reason
        self.output_tokens = output_tokens
        self.content_empty = content_empty
        self.reasoning_length = reasoning_length


def recover_complete_events(text: str) -> EventList:
    """Keep complete event objects from a response cut off mid-JSON."""
    match = re.search(r'"events"\s*:\s*\[', text)
    if not match:
        return EventList()

    events = []
    event_start = None
    object_depth = 0
    in_string = False
    escaped = False
    array_closed = False

    for index in range(match.end(), len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif event_start is None:
            if char == ']':
                array_closed = True
                break
            elif char == '{':
                event_start = index
                object_depth = 1
        elif char == '{':
            object_depth += 1
        elif char == '}':
            object_depth -= 1
            if object_depth == 0:
                try:
                    event = json.loads(text[event_start:index + 1])
                    if isinstance(event, dict):
                        events.append(event)
                except json.JSONDecodeError:
                    pass
                event_start = None

    return EventList(events, recovered=bool(events), incomplete=event_start is not None or not array_closed)


def parse_json_text(text: str, *, _depth: int = 0) -> Any:
    """Parse JSON directly, from a Markdown fence, or embedded in model text."""
    if _depth > 5:
        raise ValueError("The JSON output is nested too many times.")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(stripped)
        # Some export paths JSON-encode the complete model output as a string.
        if isinstance(value, str):
            return parse_json_text(value, _depth=_depth + 1)
        return value
    except json.JSONDecodeError:
        # Some chat exports place explanatory text around an otherwise valid
        # JSON response. Prefer a complete response object if one is present.
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char in "[{":
                try:
                    value, _ = decoder.raw_decode(stripped[index:])
                    find_events(value)
                    return value
                except json.JSONDecodeError:
                    continue
                except ValueError:
                    continue
        # If the response itself is truncated, retain complete event objects
        # that precede the cutoff and let the CLI label the counts as partial.
        recovered = recover_complete_events(stripped)
        if recovered:
            return recovered
        raise ValueError("No valid JSON object or array found in the file.")


def find_events(value: Any) -> list[Any]:
    """Extract the events array from common model-response wrappers."""
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return value if isinstance(value, EventList) else EventList(value)
        raise ValueError("The top-level JSON array must contain event objects.")
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object or an array of event objects.")

    events = value.get("events")
    if isinstance(events, list):
        return events
    if isinstance(events, str):
        return find_events(parse_json_text(events))

    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        for choice in choices:
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                reasoning = message.get("reasoning_content", "") if isinstance(message, dict) else ""
                usage = value.get("usage") or {}
                details = {
                    "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                    "output_tokens": usage.get("completion_tokens"),
                    "reasoning_length": len(reasoning) if isinstance(reasoning, str) else 0,
                }
                if not content.strip():
                    return EventList(content_empty=True, **details)
                events = find_events(parse_json_text(content))
                events.finish_reason = details["finish_reason"]
                events.output_tokens = details["output_tokens"]
                events.reasoning_length = details["reasoning_length"]
                return events
            if isinstance(content, list):
                # Some APIs encode text as content blocks.
                text = "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
                if text:
                    return find_events(parse_json_text(text))

    for key in ("content", "output", "response", "result", "text", "generated_text"):
        nested = value.get(key)
        if isinstance(nested, str):
            return find_events(parse_json_text(nested))
        if isinstance(nested, (dict, list)):
            try:
                return find_events(nested)
            except ValueError:
                pass

    raise ValueError("Couldn't find an 'events' array in the JSON file.")


def canonical(value: Any) -> str:
    """Build a stable exact-event key, ignoring the derived quote_match flag."""
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "quote_match"}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def duplicate_groups(events: list[Any]) -> list[list[int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events, start=1):
        positions[canonical(event)].append(index)
    return sorted((indices for indices in positions.values() if len(indices) > 1), key=lambda indices: indices[0])


def describe_event(event: Any) -> str:
    if not isinstance(event, dict):
        return json.dumps(event, ensure_ascii=False)
    parts = []
    for key in ("event_type", "treatment", "date_as_written", "source_quote"):
        value = event.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    description = " | ".join(parts) or json.dumps(event, ensure_ascii=False, sort_keys=True)
    return description if len(description) <= 240 else description[:237] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Count exact duplicate events in a model JSON response.")
    parser.add_argument("json_file", type=Path, help="Path to the JSON output file")
    parser.add_argument("--show", type=int, default=20, help="Maximum duplicate groups to list (default: 20; use 0 for all)")
    args = parser.parse_args()

    try:
        events = find_events(parse_json_text(args.json_file.read_text(encoding="utf-8-sig")))
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    groups = duplicate_groups(events)
    duplicated_entries = sum(len(indices) for indices in groups)
    extra_copies = sum(len(indices) - 1 for indices in groups)

    print(f"Events: {len(events)}")
    print(f"Duplicate groups: {len(groups)}")
    print(f"Event entries in duplicate groups: {duplicated_entries}")
    print(f"Extra duplicate copies: {extra_copies}")
    if getattr(events, "finish_reason", None):
        print(f"LM Studio finish reason: {events.finish_reason}")
    if getattr(events, "output_tokens", None) is not None:
        print(f"LM Studio completion tokens: {events.output_tokens}")
    if getattr(events, "content_empty", False):
        print("No structured events were returned: message.content is empty.")
        if getattr(events, "reasoning_length", 0):
            print(f"reasoning_content contains {events.reasoning_length} characters; it is not counted as event output.")
    if getattr(events, "recovered", False):
        print("Warning: the events JSON was incomplete. Counts include only complete event objects before the cutoff.")
        if getattr(events, "incomplete", False):
            print("The final event or events array may be cut off; incomplete event objects were excluded.")
    if not groups:
        return 0

    print("\nDuplicate groups (event numbers are 1-based):")
    shown = groups if args.show <= 0 else groups[:args.show]
    for number, indices in enumerate(shown, start=1):
        print(f"  Group {number}: events {', '.join(map(str, indices))} ({len(indices)} copies)")
        print(f"    {describe_event(events[indices[0] - 1])}")
    if len(shown) < len(groups):
        print(f"  ... {len(groups) - len(shown)} more group(s); use --show 0 to list all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
