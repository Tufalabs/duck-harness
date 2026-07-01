from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests

from inference.utils.openai_compat import build_chat_payload, build_headers, normalize_provider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local analyzer model.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--provider", default=os.environ.get("CHAT_PROVIDER", os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--system", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def _resolve_model(
    base_url: str,
    model: str,
    timeout: float,
    *,
    provider: str = "vllm",
    api_key: str = "",
    site_url: str = "",
    app_name: str = "",
) -> str:
    requested = model.strip()
    if requested and requested.lower() != "auto":
        return requested

    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers=build_headers(
            provider=provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        ),
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    models = payload.get("data", [])
    if not models:
        raise requests.RequestException("server returned no models")
    resolved = str(models[0]["id"])
    print(f"using model: {resolved}", flush=True)
    return resolved


def _request_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    timeout: float,
    provider: str = "vllm",
    api_key: str = "",
    site_url: str = "",
    app_name: str = "",
) -> str:
    payload = build_chat_payload(
        provider=provider,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        thinking=thinking,
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=build_headers(
            provider=provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        ),
        json=payload,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        message = f"{exc}"
        if detail:
            message += f" | response: {detail}"
        raise requests.RequestException(message) from exc
    payload: dict[str, Any] = response.json()
    message = payload["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def main() -> int:
    args = _parse_args()
    provider = normalize_provider(args.provider)
    api_key = (
        os.environ.get("CHAT_API_KEY", "").strip()
        or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    site_url = os.environ.get("CHAT_SITE_URL", os.environ.get("LOCAL_ANALYZER_SITE_URL", "")).strip()
    app_name = os.environ.get("CHAT_APP_NAME", os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness")).strip()
    try:
        model = _resolve_model(
            args.base_url,
            args.model,
            args.timeout,
            provider=provider,
            api_key=api_key,
            site_url=site_url,
            app_name=app_name,
        )
    except requests.RequestException as exc:
        print(f"failed to resolve model: {exc}", file=sys.stderr)
        return 1

    messages: list[dict[str, str]] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    if args.prompt.strip():
        messages.append({"role": "user", "content": args.prompt.strip()})
        try:
            print(
                _request_chat(
                    base_url=args.base_url,
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    thinking=args.thinking,
                    timeout=args.timeout,
                    provider=provider,
                    api_key=api_key,
                    site_url=site_url,
                    app_name=app_name,
                )
            )
            return 0
        except requests.RequestException as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            return 1

    print("Interactive chat. Ctrl-D or /exit to quit. /reset clears history.", flush=True)
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/reset":
            messages = [msg for msg in messages if msg["role"] == "system"]
            print("history cleared", flush=True)
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            reply = _request_chat(
                base_url=args.base_url,
                model=model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                thinking=args.thinking,
                timeout=args.timeout,
                provider=provider,
                api_key=api_key,
                site_url=site_url,
                app_name=app_name,
            )
        except requests.RequestException as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"model> {reply}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
