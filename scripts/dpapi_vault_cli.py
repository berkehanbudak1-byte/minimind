#!/usr/bin/env python3
"""Zero-dependency Windows DPAPI vault CLI for MiniMind secrets.

Inspects whether MiniMind secret keys (`MINIMIND_API_KEY`, `OPENAI_API_KEY`,
`XAI_API_KEY`, `GROK_API_KEY`, and any `*_API_KEY`, `*_TOKEN`, `*_SECRET`, or
`*_PASSWORD` variables) in the process environment or a `.env` file are sealed
(`dpapi:v1:`), plaintext, or unset, and atomically seals plaintext keys
in-place inside `.env` with `0o600` file permissions.

Reuses the stdlib DPAPI helpers in `scripts/api_security.py`. Never prints or
logs raw secret values, sealed ciphertexts, local directory paths, or raw
exception tracebacks.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, NoReturn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.api_security import (
    DPAPI_ALLOWED_SCOPES,
    DPAPI_PREFIX,
    DPAPIError,
    redact_sensitive_text,
    seal_dpapi_secret,
)

DEFAULT_SECRET_ENV_KEYS: tuple[str, ...] = (
    "MINIMIND_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
)

SECRET_KEY_SUFFIXES: tuple[str, ...] = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
)

_ENV_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_EXPLICIT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGN_RE = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")
_INLINE_COMMENT_RE = re.compile(r"(\s+#.*)$")
_WIN_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"`,;()]+")
_POSIX_PATH_RE = re.compile(r"(?:(?<=^)|(?<=[\s'\"=(]))/(?:[^\s'\"`,;()]+)(?:/[^\s'\"`,;()]*)*")


def is_dpapi_available() -> bool:
    """Return True when running on Windows where DPAPI is supported."""
    return sys.platform == "win32"


def is_secret_env_key(key: str) -> bool:
    """Return True if an environment variable name matches MiniMind secret key rules."""
    name = str(key or "").strip().upper()
    if not name or not _ENV_KEY_RE.match(name):
        return False
    if name in DEFAULT_SECRET_ENV_KEYS:
        return True
    return any(name.endswith(suffix) and len(name) > len(suffix) for suffix in SECRET_KEY_SUFFIXES)


def _normalize_explicit_keys(keys: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_key in keys:
        key = str(raw_key or "").strip()
        if not key or not _EXPLICIT_KEY_RE.match(key):
            raise ValueError("Invalid environment variable key name.")
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one valid environment variable key name is required.")
    return tuple(normalized)


def _safe_display_name(env_file: str | Path) -> str:
    """Return only the file basename without leaking local directory paths."""
    raw = str(env_file or "").strip()
    if not raw:
        return ".env"
    normalized = raw.replace("\\", "/").rstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    if not name or name in (".", "..") or ":" in name:
        return ".env"
    return name


def _sanitize_output_text(text: str) -> str:
    """Redact secrets and local filesystem paths from diagnostic messages."""
    cleaned = redact_sensitive_text(str(text or ""))
    cleaned = _WIN_PATH_RE.sub("[PATH]", cleaned)
    cleaned = _POSIX_PATH_RE.sub("[PATH]", cleaned)
    return cleaned


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _parse_rhs_value(rhs: str) -> tuple[str, str]:
    """Parse a `.env` value expression into `(value, trailing_suffix)`."""
    stripped = rhs.strip()
    if not stripped:
        return "", ""
    if stripped[0] in ('"', "'"):
        quote = stripped[0]
        chars: list[str] = []
        idx = 1
        escaped = False
        closed = False
        while idx < len(stripped):
            ch = stripped[idx]
            if quote == '"' and escaped:
                chars.append(ch)
                escaped = False
                idx += 1
                continue
            if quote == '"' and ch == "\\":
                escaped = True
                idx += 1
                continue
            if ch == quote:
                closed = True
                idx += 1
                break
            chars.append(ch)
            idx += 1
        if closed:
            return "".join(chars), stripped[idx:]
        return stripped, ""
    if stripped.startswith("#"):
        return "", rhs
    match = _INLINE_COMMENT_RE.search(rhs)
    if match:
        return rhs[: match.start()].strip(), match.group(1)
    return stripped, ""


def classify_secret_value(value: str | None) -> str:
    """Classify a secret value as `'sealed'`, `'plaintext'`, or `'unset'`."""
    if value is None:
        return "unset"
    raw = str(value).strip()
    if not raw:
        return "unset"
    if raw.startswith(DPAPI_PREFIX):
        return "sealed" if raw[len(DPAPI_PREFIX) :].strip() else "unset"
    return "plaintext"


def parse_env_file_values(env_file: str | Path) -> dict[str, str]:
    """Read key-value assignments from a `.env` file using stdlib only."""
    target = Path(env_file)
    if target.is_symlink():
        raise ValueError("Target .env path must not be a symlink.")
    if not target.exists():
        return {}
    if not target.is_file():
        raise ValueError("Target .env path must be a regular file.")
    values: dict[str, str] = {}
    for line in target.read_bytes().decode("utf-8-sig").splitlines():
        body, _ = _split_line_ending(line)
        match = _ASSIGN_RE.match(body)
        if not match:
            continue
        _, key, _, rhs = match.groups()
        val, _ = _parse_rhs_value(rhs)
        values[key] = val
    return values


def _resolve_status_keys(
    keys: Sequence[str] | None,
    dotenv_values: Mapping[str, str],
    env_source: Mapping[str, str],
) -> tuple[str, ...]:
    if keys is not None:
        return _normalize_explicit_keys(keys)
    ordered: list[str] = list(DEFAULT_SECRET_ENV_KEYS)
    known = set(ordered)
    extras = sorted(
        {
            k
            for k in (*dotenv_values.keys(), *env_source.keys())
            if k not in known and is_secret_env_key(k)
        }
    )
    ordered.extend(extras)
    return tuple(ordered)


def collect_vault_status(
    env_file: str | Path = ".env",
    *,
    keys: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect secret key protection state across process env and `.env`."""
    target = Path(env_file)
    if target.is_symlink():
        raise ValueError("Target .env path must not be a symlink.")
    env_file_exists = target.exists()
    if env_file_exists and not target.is_file():
        raise ValueError("Target .env path must be a regular file.")

    env_source = os.environ if environ is None else environ
    dotenv_values = parse_env_file_values(target) if env_file_exists else {}
    tracked_keys = _resolve_status_keys(keys, dotenv_values, env_source)

    key_statuses: dict[str, dict[str, str]] = {}
    has_plaintext = False
    summary = {"sealed": 0, "plaintext": 0, "unset": 0}

    for key in tracked_keys:
        env_state = classify_secret_value(env_source.get(key))
        dotenv_state = classify_secret_value(dotenv_values.get(key))
        if env_state != "unset":
            effective = env_state
            source = "env"
        elif dotenv_state != "unset":
            effective = dotenv_state
            source = "dotenv"
        else:
            effective = "unset"
            source = "none"

        if env_state == "plaintext" or dotenv_state == "plaintext":
            has_plaintext = True

        summary[effective] += 1
        key_statuses[key] = {
            "env": env_state,
            "dotenv": dotenv_state,
            "effective": effective,
            "source": source,
        }

    return {
        "ok": True,
        "dpapi_available": bool(is_dpapi_available()),
        "env_file": _safe_display_name(target),
        "env_file_exists": env_file_exists,
        "has_plaintext": has_plaintext,
        "summary": summary,
        "keys": key_statuses,
    }


def _validate_sealed_output(key: str, plaintext: str, sealed_val: Any) -> str:
    if not isinstance(sealed_val, str) or not sealed_val.startswith(DPAPI_PREFIX):
        raise DPAPIError(f"DPAPI sealing did not produce a valid sealed value for {key}.")
    payload = sealed_val[len(DPAPI_PREFIX) :].strip()
    if not payload:
        raise DPAPIError(f"DPAPI sealing did not produce a valid sealed value for {key}.")
    try:
        raw_cipher = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise DPAPIError(f"DPAPI sealing produced invalid Base64 output for {key}.") from None
    if (
        not raw_cipher
        or sealed_val == plaintext
        or raw_cipher == plaintext.encode("utf-8")
        or (len(plaintext) >= 4 and plaintext in payload)
    ):
        raise DPAPIError(f"DPAPI sealing did not produce a valid sealed value for {key}.")
    return f"{DPAPI_PREFIX}{payload}"


def seal_env_file(
    env_file: str | Path,
    *,
    keys: Sequence[str] | None = None,
    dry_run: bool = False,
    scope: str = "CurrentUser",
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Seal plaintext managed secrets in `.env` atomically with `0o600` permissions."""
    if not isinstance(scope, str) or scope not in DPAPI_ALLOWED_SCOPES:
        raise ValueError(f"Unsupported DPAPI scope; expected one of {sorted(DPAPI_ALLOWED_SCOPES)}.")

    explicit_keys = _normalize_explicit_keys(keys) if keys is not None else None
    explicit_set = set(explicit_keys) if explicit_keys is not None else None

    target = Path(env_file)
    if target.is_symlink():
        raise ValueError("Target .env path must not be a symlink.")
    if not target.exists():
        raise FileNotFoundError("Target .env file does not exist.")
    if not target.is_file():
        raise ValueError("Target .env path must be a regular file.")

    raw_bytes = target.read_bytes()
    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    raw_text = raw_bytes.decode("utf-8-sig")

    sealed_set: set[str] = set()
    would_seal_set: set[str] = set()
    already_sealed_set: set[str] = set()
    discovered_extra_keys: set[str] = set()
    new_lines: list[str] = []
    changed = False

    for line in raw_text.splitlines(keepends=True):
        body, eol = _split_line_ending(line)
        match = _ASSIGN_RE.match(body)
        if not match:
            new_lines.append(line)
            continue

        prefix, key, eq_part, rhs = match.groups()
        is_managed = (key in explicit_set) if explicit_set is not None else is_secret_env_key(key)
        if not is_managed:
            new_lines.append(line)
            continue

        if explicit_set is None and key not in DEFAULT_SECRET_ENV_KEYS:
            discovered_extra_keys.add(key)

        val, suffix = _parse_rhs_value(rhs)
        state = classify_secret_value(val)
        if state == "unset":
            new_lines.append(line)
            continue
        if state == "sealed":
            already_sealed_set.add(key)
            new_lines.append(line)
            continue

        if dry_run:
            would_seal_set.add(key)
            new_lines.append(line)
            continue

        if runner is None and not is_dpapi_available():
            raise DPAPIError(f"Windows DPAPI is unavailable; cannot seal {key}.")

        try:
            raw_sealed = seal_dpapi_secret(val, scope=scope, runner=runner)
        except Exception as exc:
            raise DPAPIError(f"DPAPI sealing failed for {key} ({type(exc).__name__}).") from None

        sealed_val = _validate_sealed_output(key, val, raw_sealed)
        new_lines.append(f"{prefix}{key}{eq_part}{sealed_val}{suffix}{eol}")
        sealed_set.add(key)
        changed = True

    if not dry_run:
        if changed:
            output_bytes = "".join(new_lines).encode("utf-8-sig" if has_bom else "utf-8")
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=str(target.parent),
            )
            tmp_path = Path(tmp_name)
            fd_closed = False
            try:
                os.chmod(tmp_path, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    fd_closed = True
                    handle.write(output_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, target)
                os.chmod(target, 0o600)
            except Exception:
                if not fd_closed:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        else:
            os.chmod(target, 0o600)

    if explicit_keys is not None:
        tracked_keys = list(explicit_keys)
    else:
        tracked_keys = list(DEFAULT_SECRET_ENV_KEYS) + sorted(discovered_extra_keys)

    active_pending = would_seal_set if dry_run else sealed_set
    sealed_keys = [k for k in tracked_keys if k in sealed_set]
    would_seal_keys = [k for k in tracked_keys if k in would_seal_set]
    already_sealed_keys = [k for k in tracked_keys if k in already_sealed_set and k not in active_pending]
    unset_keys = [k for k in tracked_keys if k not in active_pending and k not in already_sealed_set]

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "dpapi_available": bool(is_dpapi_available()),
        "env_file": _safe_display_name(target),
        "changed": False if dry_run else changed,
        "would_change": bool(would_seal_keys) if dry_run else changed,
        "sealed_keys": sealed_keys,
        "would_seal_keys": would_seal_keys,
        "already_sealed_keys": already_sealed_keys,
        "unset_keys": unset_keys,
        "permissions": None if dry_run else "0600",
    }


def _format_status_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Windows DPAPI available: {report['dpapi_available']}",
        f"Env file: {report['env_file']} ({'present' if report['env_file_exists'] else 'absent'})",
        f"Summary: sealed={summary['sealed']}, plaintext={summary['plaintext']}, unset={summary['unset']}",
    ]
    for key, info in report["keys"].items():
        lines.append(
            f"  {key}: effective={info['effective']} (source={info['source']}, "
            f"env={info['env']}, dotenv={info['dotenv']})"
        )
    return "\n".join(lines)


def _format_seal_text(result: dict[str, Any]) -> str:
    already = ", ".join(result["already_sealed_keys"]) if result["already_sealed_keys"] else "none"
    unset = ", ".join(result["unset_keys"]) if result["unset_keys"] else "none"
    if result["dry_run"]:
        would = ", ".join(result["would_seal_keys"]) if result["would_seal_keys"] else "none"
        return (
            f"DPAPI seal dry-run ({result['env_file']}, would_change={result['would_change']}, "
            f"dpapi_available={result['dpapi_available']})\n"
            f"  would_seal: {would}\n"
            f"  already_sealed: {already}\n"
            f"  unset: {unset}"
        )
    sealed = ", ".join(result["sealed_keys"]) if result["sealed_keys"] else "none"
    return (
        f"DPAPI seal complete ({result['env_file']}, permissions={result['permissions']}, "
        f"changed={result['changed']})\n"
        f"  sealed: {sealed}\n"
        f"  already_sealed: {already}\n"
        f"  unset: {unset}"
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that sanitizes error output to prevent path or secret leakage."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        safe_msg = _sanitize_output_text(message)
        self.exit(2, f"{self.prog}: error: {safe_msg}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="dpapi_vault_cli.py",
        description="Inspect or seal MiniMind secrets with Windows DPAPI (dpapi:v1:).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser(
        "status",
        help="Check whether managed secret keys are sealed, plaintext, or unset",
    )
    p_status.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    p_status.add_argument(
        "--key",
        action="append",
        dest="keys",
        default=None,
        help="Specific environment key name to inspect (repeatable)",
    )
    p_status.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON summary")

    p_seal = sub.add_parser(
        "seal",
        help="Seal plaintext managed secrets in-place inside a .env file",
    )
    p_seal.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    p_seal.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which keys would be sealed without modifying the file or invoking DPAPI",
    )
    p_seal.add_argument(
        "--scope",
        default="CurrentUser",
        choices=sorted(DPAPI_ALLOWED_SCOPES),
        help="Windows DPAPI protection scope (default: CurrentUser)",
    )
    p_seal.add_argument(
        "--key",
        action="append",
        dest="keys",
        default=None,
        help="Specific environment key name to seal (repeatable)",
    )
    p_seal.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON summary")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1

    if args.cmd == "status":
        try:
            report = collect_vault_status(args.env_file, keys=args.keys, environ=environ)
        except ValueError as exc:
            print(f"Error: {_sanitize_output_text(str(exc))}", file=sys.stderr)
            return 1
        except UnicodeDecodeError as exc:
            print(f"Error: target .env file is not valid UTF-8 ({type(exc).__name__}).", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"Error: could not read .env file ({type(exc).__name__}).", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error: unexpected status failure ({type(exc).__name__}).", file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(_format_status_text(report))
        return 0

    if args.cmd == "seal":
        try:
            result = seal_env_file(
                args.env_file,
                keys=args.keys,
                dry_run=bool(args.dry_run),
                scope=args.scope,
                runner=runner,
            )
        except FileNotFoundError:
            print("Error: target .env file not found.", file=sys.stderr)
            return 1
        except UnicodeDecodeError as exc:
            print(f"Error: target .env file is not valid UTF-8 ({type(exc).__name__}).", file=sys.stderr)
            return 1
        except (ValueError, DPAPIError) as exc:
            print(f"Error: {_sanitize_output_text(str(exc))}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"Error: failed to update .env file ({type(exc).__name__}).", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error: unexpected seal failure ({type(exc).__name__}).", file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(_format_seal_text(result))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
