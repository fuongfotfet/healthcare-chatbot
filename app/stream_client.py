import argparse
import json
import sys
import urllib.error
import urllib.request


def sse_events(response):
    """Yield parsed SSE events as (event_name, data)."""
    event_name = None
    data_lines = []

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")

        if line.endswith("\r"):
            line = line[:-1]

        # Empty line marks the end of one SSE event.
        if line == "":
            if event_name is not None or data_lines:
                yield event_name or "message", "\n".join(data_lines)
            event_name = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            # Per SSE, a single optional space may appear after "data:".
            # Remove only that one space and keep the rest to preserve streamed spacing.
            payload = line[len("data:") :]
            if payload.startswith(" "):
                payload = payload[1:]
            data_lines.append(payload)

    # Flush trailing buffered event if server closes without a final blank line.
    if event_name is not None or data_lines:
        yield event_name or "message", "\n".join(data_lines)


def stream_chat(base_url, query, timeout, role=""):
    payload = json.dumps({"query": query, "role": role or ""}).encode("utf-8")
    endpoint = base_url.rstrip("/") + "/api/chat/stream"

    request = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for event, data in sse_events(response):
                if event == "error":
                    print(f"\n[Lỗi] {data}", file=sys.stderr)
                    return 1
                if event == "trace":
                    if data:
                        print(data, flush=True)
                    continue
                if event == "done":
                    print("\n\n[Hoàn tất luồng phản hồi]")
                    return 0
                if data:
                    print(data, end="", flush=True)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        print(f"Lỗi HTTP {exc.code}: {details}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Lỗi kết nối: {exc}", file=sys.stderr)
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="SSE streaming client for Medical RAG API")
    parser.add_argument("--url", default="http://127.0.0.1:3636", help="Base URL of API server")
    parser.add_argument("--query", required=True, help="User query to send")
    parser.add_argument("--role", default="", help="Optional role for shortcut flow (e.g. bac_si_tramyte)")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    raise SystemExit(stream_chat(args.url, args.query, args.timeout, args.role))


if __name__ == "__main__":
    main()
