"""Stdlib unittest suite for scripts/dpapi_vault_cli.py."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import dpapi_vault_cli
from scripts.api_security import DPAPI_PREFIX, resolve_dpapi_secret, seal_dpapi_secret, unseal_dpapi_secret


def _make_fake_dpapi_runner(*, fail_on_call: int | None = None, leak_text: str = ""):
    calls: list[dict[str, object]] = []

    def _xor(data: bytes, mask: bytes) -> bytes:
        return bytes(b ^ mask[i % len(mask)] for i, b in enumerate(data))

    def runner(cmd, *, capture_output=True, text=True, env=None, timeout=None, check=False, shell=False):
        call_index = len(calls) + 1
        calls.append({"cmd": list(cmd), "env": dict(env or {}), "shell": shell, "timeout": timeout})
        if fail_on_call is not None and call_index == fail_on_call:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=f"dpapi failure {leak_text}",
            )
        script = cmd[-1]
        in_bytes = base64.b64decode(env["_DPAPI_INPUT_B64"], validate=True)
        scope = env.get("_DPAPI_SCOPE", "CurrentUser").encode("utf-8")
        mask = b"MINIMIND-VAULT-CLI:" + scope
        if "::Protect(" in script:
            out_bytes = b"MV1|" + _xor(in_bytes, mask)
        elif "::Unprotect(" in script:
            if not in_bytes.startswith(b"MV1|"):
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="invalid blob")
            out_bytes = _xor(in_bytes[4:], mask)
        else:
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="unknown script")
        return subprocess.CompletedProcess(cmd, 0, stdout=base64.b64encode(out_bytes).decode("ascii"), stderr="")

    return runner, calls


class TestDpapiVaultCli(unittest.TestCase):
    def test_secret_key_matching_and_classification(self) -> None:
        for key in (
            "MINIMIND_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "GROK_API_KEY",
            "CUSTOM_API_KEY",
            "HF_TOKEN",
            "GITHUB_TOKEN",
            "WEBHOOK_SECRET",
            "REDIS_PASSWORD",
            "DB_PASSWORD",
        ):
            self.assertTrue(dpapi_vault_cli.is_secret_env_key(key), f"Expected {key} to be classified as secret")

        for non_secret in (
            "MINIMIND_MAX_TOKENS",
            "MINIMIND_API_BASE_URL",
            "MINIMIND_MODEL",
            "MINIMIND_TEMPERATURE",
            "MINIMIND_TRUST_PROXY",
            "MINIMIND_TRUST_REMOTE_CODE",
            "ACCESS_PROFILE",
            "_INTERNAL_TOKEN",
            "_API_KEY",
            "",
        ):
            self.assertFalse(
                dpapi_vault_cli.is_secret_env_key(non_secret),
                f"Expected {non_secret!r} NOT to be classified as secret",
            )

        self.assertEqual(dpapi_vault_cli.classify_secret_value(None), "unset")
        self.assertEqual(dpapi_vault_cli.classify_secret_value(""), "unset")
        self.assertEqual(dpapi_vault_cli.classify_secret_value("   "), "unset")
        self.assertEqual(dpapi_vault_cli.classify_secret_value("dpapi:v1:"), "unset")
        self.assertEqual(dpapi_vault_cli.classify_secret_value("dpapi:v1:   "), "unset")
        self.assertEqual(dpapi_vault_cli.classify_secret_value("dpapi:v1:AQIDBA=="), "sealed")
        self.assertEqual(dpapi_vault_cli.classify_secret_value("plain-secret"), "plaintext")

        self.assertEqual(dpapi_vault_cli._safe_display_name("/home/testuser/repo/.env"), ".env")
        self.assertEqual(dpapi_vault_cli._safe_display_name(r"C:\Users\TestUser\minimind\prod.env"), "prod.env")

    def test_status_reports_sealed_plaintext_unset_with_zero_leaks(self) -> None:
        runner, _ = _make_fake_dpapi_runner()
        sealed_mm = seal_dpapi_secret("mm-top-secret-alpha", runner=runner)
        sealed_webhook = seal_dpapi_secret("wh-secret-sealed-001", runner=runner)
        sealed_env_tok = seal_dpapi_secret("env-token-sealed-002", runner=runner)
        plain_openai = "sk-openai-plain-secret-111"
        plain_hf = "hf-token-plain-secret-222"
        plain_db_pw = "db-password-plain-333"
        plain_grok_env = "grok-env-plain-secret-444"

        with tempfile.TemporaryDirectory(prefix="mm-vault-status-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# MiniMind local config",
                        "MINIMIND_MAX_TOKENS=2048",
                        "MINIMIND_MODEL=minimind",
                        f"MINIMIND_API_KEY={sealed_mm}",
                        f'OPENAI_API_KEY="{plain_openai}"  # inline note',
                        "XAI_API_KEY=",
                        f"HF_TOKEN={plain_hf}",
                        f"WEBHOOK_SECRET={sealed_webhook}",
                        f"export DB_PASSWORD='{plain_db_pw}'",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            env_map = {
                "GROK_API_KEY": plain_grok_env,
                "SERVICE_TOKEN": sealed_env_tok,
                "MINIMIND_MAX_TOKENS": "1024",
            }

            # JSON output
            out_json, err_json = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_json), contextlib.redirect_stderr(err_json):
                rc_json = dpapi_vault_cli.main(
                    ["status", "--env-file", str(env_path), "--json"],
                    environ=env_map,
                )
            self.assertEqual(rc_json, 0)
            self.assertEqual(err_json.getvalue(), "")
            raw_json = out_json.getvalue()
            report = json.loads(raw_json)
            self.assertTrue(report["ok"])
            self.assertEqual(report["env_file"], ".env")
            self.assertTrue(report["env_file_exists"])
            self.assertTrue(report["has_plaintext"])
            self.assertNotIn("MINIMIND_MAX_TOKENS", report["keys"])
            self.assertNotIn("MINIMIND_MODEL", report["keys"])
            self.assertEqual(
                report["keys"]["MINIMIND_API_KEY"],
                {"env": "unset", "dotenv": "sealed", "effective": "sealed", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["OPENAI_API_KEY"],
                {"env": "unset", "dotenv": "plaintext", "effective": "plaintext", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["XAI_API_KEY"],
                {"env": "unset", "dotenv": "unset", "effective": "unset", "source": "none"},
            )
            self.assertEqual(
                report["keys"]["GROK_API_KEY"],
                {"env": "plaintext", "dotenv": "unset", "effective": "plaintext", "source": "env"},
            )
            self.assertEqual(
                report["keys"]["HF_TOKEN"],
                {"env": "unset", "dotenv": "plaintext", "effective": "plaintext", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["WEBHOOK_SECRET"],
                {"env": "unset", "dotenv": "sealed", "effective": "sealed", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["DB_PASSWORD"],
                {"env": "unset", "dotenv": "plaintext", "effective": "plaintext", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["SERVICE_TOKEN"],
                {"env": "sealed", "dotenv": "unset", "effective": "sealed", "source": "env"},
            )
            self.assertEqual(report["summary"], {"sealed": 3, "plaintext": 4, "unset": 1})

            # Human-readable text output
            out_txt, err_txt = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_txt), contextlib.redirect_stderr(err_txt):
                rc_txt = dpapi_vault_cli.main(
                    ["status", "--env-file", str(env_path)],
                    environ=env_map,
                )
            self.assertEqual(rc_txt, 0)
            self.assertEqual(err_txt.getvalue(), "")
            raw_txt = out_txt.getvalue()
            self.assertIn("Env file: .env (present)", raw_txt)
            self.assertIn("MINIMIND_API_KEY: effective=sealed", raw_txt)
            self.assertIn("OPENAI_API_KEY: effective=plaintext", raw_txt)
            self.assertIn("HF_TOKEN: effective=plaintext", raw_txt)
            self.assertIn("DB_PASSWORD: effective=plaintext", raw_txt)

            # Zero leakage of raw secrets, ciphertexts, prefix, or local paths
            for forbidden in (
                "mm-top-secret-alpha",
                "wh-secret-sealed-001",
                "env-token-sealed-002",
                plain_openai,
                plain_hf,
                plain_db_pw,
                plain_grok_env,
                sealed_mm,
                sealed_webhook,
                sealed_env_tok,
                DPAPI_PREFIX,
                tmp_dir,
                str(Path.home()),
                "Traceback",
            ):
                self.assertNotIn(forbidden, raw_json)
                self.assertNotIn(forbidden, raw_txt)

            # Explicit --key filtering
            out_filtered = io.StringIO()
            with contextlib.redirect_stdout(out_filtered):
                rc_f = dpapi_vault_cli.main(
                    ["status", "--env-file", str(env_path), "--key", "MINIMIND_API_KEY", "--json"],
                    environ=env_map,
                )
            self.assertEqual(rc_f, 0)
            filtered = json.loads(out_filtered.getvalue())
            self.assertEqual(list(filtered["keys"].keys()), ["MINIMIND_API_KEY"])
            self.assertFalse(filtered["has_plaintext"])

            # Missing .env file still succeeds cleanly
            missing_env = Path(tmp_dir) / "absent.env"
            out_missing = io.StringIO()
            with contextlib.redirect_stdout(out_missing):
                rc_m = dpapi_vault_cli.main(
                    ["status", "--env-file", str(missing_env), "--json"],
                    environ={"MINIMIND_API_KEY": sealed_mm},
                )
            self.assertEqual(rc_m, 0)
            m_report = json.loads(out_missing.getvalue())
            self.assertEqual(m_report["env_file"], "absent.env")
            self.assertFalse(m_report["env_file_exists"])
            self.assertFalse(m_report["has_plaintext"])
            self.assertEqual(m_report["keys"]["MINIMIND_API_KEY"]["effective"], "sealed")
            self.assertNotIn(tmp_dir, out_missing.getvalue())

    def test_seal_dry_run_previews_without_modifying_file_or_invoking_dpapi(self) -> None:
        runner, _ = _make_fake_dpapi_runner()
        sealed_xai = seal_dpapi_secret("xai-already-sealed-secret", runner=runner)
        plain_mm = "mm-dryrun-secret-001"
        plain_tok = "hf-dryrun-token-002"
        plain_sec = "jwt-dryrun-secret-003"
        plain_pw = "db-dryrun-password-004"

        def forbidden_runner(*args, **kwargs):
            raise AssertionError("DPAPI runner must never be invoked during --dry-run")

        with tempfile.TemporaryDirectory(prefix="mm-vault-dryrun-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original = (
                "# preserve me\n"
                f"MINIMIND_API_KEY={plain_mm}\n"
                f"XAI_API_KEY={sealed_xai}\n"
                "GROK_API_KEY=\n"
                f"HF_TOKEN={plain_tok}\n"
                f"JWT_SECRET={plain_sec}\n"
                f"DB_PASSWORD={plain_pw}\n"
                "MINIMIND_MAX_TOKENS=2048\n"
            )
            env_path.write_text(original, encoding="utf-8")
            os.chmod(env_path, 0o644)

            out_json, err_json = io.StringIO(), io.StringIO()
            with mock.patch("scripts.dpapi_vault_cli.is_dpapi_available", return_value=False):
                with contextlib.redirect_stdout(out_json), contextlib.redirect_stderr(err_json):
                    rc = dpapi_vault_cli.main(
                        ["seal", "--dry-run", "--env-file", str(env_path), "--json"],
                        runner=forbidden_runner,
                    )
            self.assertEqual(rc, 0)
            self.assertEqual(err_json.getvalue(), "")
            raw_json = out_json.getvalue()
            res = json.loads(raw_json)
            self.assertTrue(res["ok"])
            self.assertTrue(res["dry_run"])
            self.assertFalse(res["changed"])
            self.assertTrue(res["would_change"])
            self.assertEqual(res["env_file"], ".env")
            self.assertEqual(res["sealed_keys"], [])
            self.assertEqual(
                res["would_seal_keys"],
                ["MINIMIND_API_KEY", "DB_PASSWORD", "HF_TOKEN", "JWT_SECRET"],
            )
            self.assertEqual(res["already_sealed_keys"], ["XAI_API_KEY"])
            self.assertEqual(res["unset_keys"], ["OPENAI_API_KEY", "GROK_API_KEY"])
            self.assertIsNone(res["permissions"])

            # File bytes, permissions, and directory are untouched
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o644)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # Text output for dry-run
            out_txt, err_txt = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_txt), contextlib.redirect_stderr(err_txt):
                rc_txt = dpapi_vault_cli.main(
                    ["seal", "--dry-run", "--env-file", str(env_path)],
                    runner=forbidden_runner,
                )
            self.assertEqual(rc_txt, 0)
            self.assertEqual(err_txt.getvalue(), "")
            raw_txt = out_txt.getvalue()
            self.assertIn("DPAPI seal dry-run (.env", raw_txt)
            self.assertIn("would_seal: MINIMIND_API_KEY, DB_PASSWORD, HF_TOKEN, JWT_SECRET", raw_txt)

            for forbidden in (
                plain_mm,
                plain_tok,
                plain_sec,
                plain_pw,
                sealed_xai,
                DPAPI_PREFIX,
                tmp_dir,
                str(Path.home()),
            ):
                self.assertNotIn(forbidden, raw_json)
                self.assertNotIn(forbidden, raw_txt)

            # Dry-run when nothing needs sealing reports would_change=False
            sealed_only = Path(tmp_dir) / "clean.env"
            sealed_only.write_text(f"XAI_API_KEY={sealed_xai}\n", encoding="utf-8")
            clean_res = dpapi_vault_cli.seal_env_file(sealed_only, dry_run=True)
            self.assertTrue(clean_res["dry_run"])
            self.assertFalse(clean_res["changed"])
            self.assertFalse(clean_res["would_change"])
            self.assertEqual(clean_res["would_seal_keys"], [])
            self.assertEqual(clean_res["already_sealed_keys"], ["XAI_API_KEY"])

    def test_seal_atomic_mkstemp_fsync_replace_0600_and_roundtrip(self) -> None:
        runner, calls = _make_fake_dpapi_runner()
        plain_mm = "mm-plain-ğüşöçıİ-secret-001"
        plain_openai = 'sk-openai#hash-"quoted"-002'
        plain_hf = "hf-token-secret-003"
        plain_secret = "webhook-secret-004"
        plain_pw = "db-password-005"

        with tempfile.TemporaryDirectory(prefix="mm-vault-seal-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# MiniMind production env",
                        "MINIMIND_MAX_TOKENS=2048",
                        f"MINIMIND_API_KEY={plain_mm}",
                        f'OPENAI_API_KEY="sk-openai#hash-\\"quoted\\"-002"  # prod key',
                        f"export HF_TOKEN='{plain_hf}'",
                        f"WEBHOOK_SECRET={plain_secret} # webhook",
                        f"DB_PASSWORD={plain_pw}",
                        "XAI_API_KEY=",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(env_path, 0o644)

            events: list[str] = []
            chmod_modes: list[int] = []
            real_mkstemp = tempfile.mkstemp
            real_fsync = os.fsync
            real_replace = os.replace
            real_chmod = os.chmod

            def spy_mkstemp(*args, **kwargs):
                events.append("mkstemp")
                self.assertEqual(kwargs.get("dir"), str(env_path.parent))
                return real_mkstemp(*args, **kwargs)

            def spy_fsync(fd):
                events.append("fsync")
                return real_fsync(fd)

            def spy_replace(src, dst):
                events.append("replace")
                return real_replace(src, dst)

            def spy_chmod(path, mode, *args, **kwargs):
                chmod_modes.append(mode)
                return real_chmod(path, mode, *args, **kwargs)

            out_seal, err_seal = io.StringIO(), io.StringIO()
            with (
                mock.patch("scripts.dpapi_vault_cli.tempfile.mkstemp", side_effect=spy_mkstemp),
                mock.patch("scripts.dpapi_vault_cli.os.fsync", side_effect=spy_fsync),
                mock.patch("scripts.dpapi_vault_cli.os.replace", side_effect=spy_replace),
                mock.patch("scripts.dpapi_vault_cli.os.chmod", side_effect=spy_chmod),
                contextlib.redirect_stdout(out_seal),
                contextlib.redirect_stderr(err_seal),
            ):
                rc = dpapi_vault_cli.main(
                    ["seal", "--env-file", str(env_path), "--scope", "LocalMachine", "--json"],
                    runner=runner,
                )

            self.assertEqual(rc, 0)
            self.assertEqual(err_seal.getvalue(), "")
            raw_out = out_seal.getvalue()
            res = json.loads(raw_out)
            self.assertTrue(res["ok"])
            self.assertFalse(res["dry_run"])
            self.assertTrue(res["changed"])
            self.assertTrue(res["would_change"])
            self.assertEqual(res["env_file"], ".env")
            self.assertEqual(res["permissions"], "0600")
            self.assertEqual(
                res["sealed_keys"],
                ["MINIMIND_API_KEY", "OPENAI_API_KEY", "DB_PASSWORD", "HF_TOKEN", "WEBHOOK_SECRET"],
            )
            self.assertEqual(res["already_sealed_keys"], [])
            self.assertEqual(res["unset_keys"], ["XAI_API_KEY", "GROK_API_KEY"])

            # Verify atomic ordering: mkstemp -> fsync -> replace, plus 0o600 permissions
            self.assertEqual(events, ["mkstemp", "fsync", "replace"])
            self.assertTrue(all(mode == 0o600 for mode in chmod_modes))
            self.assertGreaterEqual(len(chmod_modes), 2)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o600)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # Verify scope forwarded & zero cmd-line secret interpolation
            self.assertEqual(len(calls), 5)
            for call in calls:
                self.assertFalse(call["shell"])
                self.assertEqual(call["env"]["_DPAPI_SCOPE"], "LocalMachine")
                joined_cmd = " ".join(call["cmd"])
                for secret in (plain_mm, plain_openai, plain_hf, plain_secret, plain_pw):
                    self.assertNotIn(secret, joined_cmd)

            # Verify disk content has zero plaintext and preserves structure
            disk_text = env_path.read_text(encoding="utf-8")
            for secret in (plain_mm, plain_openai, plain_hf, plain_secret, plain_pw):
                self.assertNotIn(secret, disk_text)
                self.assertNotIn(secret, raw_out)
            self.assertNotIn(DPAPI_PREFIX, raw_out)
            self.assertNotIn(tmp_dir, raw_out)
            self.assertIn("# MiniMind production env", disk_text)
            self.assertIn("MINIMIND_MAX_TOKENS=2048", disk_text)
            self.assertIn("  # prod key", disk_text)
            self.assertIn(" # webhook", disk_text)
            self.assertIn("export HF_TOKEN=dpapi:v1:", disk_text)

            # Verify every sealed value round-trips with scripts/api_security.py unseal/resolve
            parsed = dpapi_vault_cli.parse_env_file_values(env_path)
            expected = {
                "MINIMIND_API_KEY": plain_mm,
                "OPENAI_API_KEY": plain_openai,
                "HF_TOKEN": plain_hf,
                "WEBHOOK_SECRET": plain_secret,
                "DB_PASSWORD": plain_pw,
            }
            for key, plain_val in expected.items():
                self.assertTrue(parsed[key].startswith(DPAPI_PREFIX))
                self.assertEqual(
                    unseal_dpapi_secret(parsed[key], scope="LocalMachine", runner=runner),
                    plain_val,
                )
                self.assertEqual(
                    resolve_dpapi_secret(parsed[key], scope="LocalMachine", runner=runner),
                    plain_val,
                )

    def test_seal_idempotency_and_utf8_bom_preservation(self) -> None:
        runner, calls = _make_fake_dpapi_runner()
        with tempfile.TemporaryDirectory(prefix="mm-vault-idem-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_bytes(
                "\ufeffMINIMIND_API_KEY=plain-bom-secret\r\nMINIMIND_MODEL=minimind\r\n".encode("utf-8")
            )

            first = dpapi_vault_cli.seal_env_file(env_path, runner=runner)
            self.assertTrue(first["changed"])
            self.assertEqual(first["sealed_keys"], ["MINIMIND_API_KEY"])
            first_bytes = env_path.read_bytes()
            self.assertTrue(first_bytes.startswith(b"\xef\xbb\xbf"))
            self.assertIn(b"\r\n", first_bytes)
            self.assertEqual(len(calls), 1)

            # Second pass is idempotent: no rewrite, no DPAPI call even if DPAPI unavailable
            out_idem, err_idem = io.StringIO(), io.StringIO()
            with (
                mock.patch("scripts.dpapi_vault_cli.is_dpapi_available", return_value=False),
                mock.patch("scripts.dpapi_vault_cli.tempfile.mkstemp", side_effect=AssertionError("must not rewrite")),
                contextlib.redirect_stdout(out_idem),
                contextlib.redirect_stderr(err_idem),
            ):
                rc = dpapi_vault_cli.main(["seal", "--env-file", str(env_path), "--json"])
            self.assertEqual(rc, 0)
            self.assertEqual(err_idem.getvalue(), "")
            idem = json.loads(out_idem.getvalue())
            self.assertFalse(idem["changed"])
            self.assertEqual(idem["sealed_keys"], [])
            self.assertEqual(idem["already_sealed_keys"], ["MINIMIND_API_KEY"])
            self.assertEqual(env_path.read_bytes(), first_bytes)

            # Status reports zero plaintext after sealing
            status = dpapi_vault_cli.collect_vault_status(env_path, environ={})
            self.assertFalse(status["has_plaintext"])
            self.assertEqual(status["keys"]["MINIMIND_API_KEY"]["dotenv"], "sealed")

    def test_seal_fail_closed_and_cleanup_on_errors_without_leaks(self) -> None:
        secret_a = "mm-never-leak-secret-alpha-9001"
        secret_b = "hf-never-leak-secret-beta-9002"

        with tempfile.TemporaryDirectory(prefix="mm-vault-fail-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original = f"MINIMIND_API_KEY={secret_a}\nHF_TOKEN={secret_b}\n"
            env_path.write_text(original, encoding="utf-8")
            os.chmod(env_path, 0o644)

            # 1. DPAPI unavailable and no runner -> fail closed, file & mode untouched
            out1, err1 = io.StringIO(), io.StringIO()
            with mock.patch("scripts.dpapi_vault_cli.is_dpapi_available", return_value=False):
                with contextlib.redirect_stdout(out1), contextlib.redirect_stderr(err1):
                    rc1 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)])
            self.assertEqual(rc1, 1)
            self.assertEqual(out1.getvalue(), "")
            self.assertIn("MINIMIND_API_KEY", err1.getvalue())
            for forbidden in (secret_a, secret_b, tmp_dir, str(Path.home()), "Traceback"):
                self.assertNotIn(forbidden, err1.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o644)

            # 2. Partial failure on 2nd key with secret + local path in runner stderr -> all-or-nothing fail closed
            partial_runner, _ = _make_fake_dpapi_runner(
                fail_on_call=2,
                leak_text=f"{secret_b} /home/testuser/secret C:\\Users\\TestUser\\secret",
            )
            out2, err2 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
                rc2 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=partial_runner)
            self.assertEqual(rc2, 1)
            self.assertEqual(out2.getvalue(), "")
            self.assertIn("HF_TOKEN", err2.getvalue())
            for forbidden in (secret_a, secret_b, tmp_dir, "/home/testuser", r"C:\Users\TestUser", "Traceback"):
                self.assertNotIn(forbidden, err2.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # 3. Runner raising exception containing plaintext secret -> scrubbed, file untouched
            def exploding_runner(cmd, **kwargs):
                raise RuntimeError(f"boom {secret_a} at /home/testuser/vault")

            out3, err3 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out3), contextlib.redirect_stderr(err3):
                rc3 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=exploding_runner)
            self.assertEqual(rc3, 1)
            for forbidden in (secret_a, "/home/testuser", "Traceback"):
                self.assertNotIn(forbidden, err3.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)

            # 4. Passthrough/noop runner echoing base64 plaintext without encryption -> rejected fail-closed
            def passthrough_runner(cmd, *, env=None, **kwargs):
                return subprocess.CompletedProcess(cmd, 0, stdout=env["_DPAPI_INPUT_B64"], stderr="")

            out4, err4 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out4), contextlib.redirect_stderr(err4):
                rc4 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=passthrough_runner)
            self.assertEqual(rc4, 1)
            self.assertNotIn(secret_a, err4.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)

            # 5. Atomic replace OSError cleans up temporary file and omits local paths from stderr
            ok_runner, _ = _make_fake_dpapi_runner()
            out5, err5 = io.StringIO(), io.StringIO()
            with (
                mock.patch(
                    "scripts.dpapi_vault_cli.os.replace",
                    side_effect=PermissionError(13, "Permission denied", str(env_path)),
                ),
                contextlib.redirect_stdout(out5),
                contextlib.redirect_stderr(err5),
            ):
                rc5 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=ok_runner)
            self.assertEqual(rc5, 1)
            self.assertIn("PermissionError", err5.getvalue())
            self.assertNotIn(tmp_dir, err5.getvalue())
            self.assertNotIn("Traceback", err5.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # 6. Missing file, directory target, symlink target, and invalid UTF-8 fail cleanly without paths
            out6, err6 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out6), contextlib.redirect_stderr(err6):
                rc6 = dpapi_vault_cli.main(["seal", "--env-file", str(Path(tmp_dir) / "missing.env")], runner=ok_runner)
            self.assertEqual(rc6, 1)
            self.assertIn("not found", err6.getvalue().lower())
            self.assertNotIn(tmp_dir, err6.getvalue())

            out7, err7 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out7), contextlib.redirect_stderr(err7):
                rc7 = dpapi_vault_cli.main(["seal", "--env-file", tmp_dir], runner=ok_runner)
            self.assertEqual(rc7, 1)
            self.assertNotIn(tmp_dir, err7.getvalue())

            if os.name == "posix":
                link_path = Path(tmp_dir) / "link.env"
                link_path.symlink_to(env_path)
                out8, err8 = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out8), contextlib.redirect_stderr(err8):
                    rc8 = dpapi_vault_cli.main(["seal", "--env-file", str(link_path)], runner=ok_runner)
                self.assertEqual(rc8, 1)
                self.assertIn("symlink", err8.getvalue().lower())
                self.assertNotIn(tmp_dir, err8.getvalue())
                link_path.unlink()

            bad_utf8 = Path(tmp_dir) / "bad.env"
            bad_utf8.write_bytes(b"\xff\xfeMINIMIND_API_KEY=abc\n")
            out9, err9 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out9), contextlib.redirect_stderr(err9):
                rc9 = dpapi_vault_cli.main(["seal", "--env-file", str(bad_utf8)], runner=ok_runner)
            self.assertEqual(rc9, 1)
            self.assertIn("UTF-8", err9.getvalue())
            self.assertNotIn(tmp_dir, err9.getvalue())
            self.assertNotIn("Traceback", err9.getvalue())

    def test_cli_argument_errors_return_code_2_and_sanitize_paths_and_secrets(self) -> None:
        # Missing subcommand
        out0, err0 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out0), contextlib.redirect_stderr(err0):
            rc0 = dpapi_vault_cli.main([])
        self.assertEqual(rc0, 2)
        self.assertNotIn("Traceback", err0.getvalue())

        # Invalid subcommand
        out1, err1 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out1), contextlib.redirect_stderr(err1):
            rc1 = dpapi_vault_cli.main(["rotate"])
        self.assertEqual(rc1, 2)
        self.assertNotIn("Traceback", err1.getvalue())

        # Unrecognized arguments carrying local POSIX/Windows paths & secret tokens must be scrubbed
        out2, err2 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
            rc2 = dpapi_vault_cli.main(
                [
                    "status",
                    "--unknown-arg",
                    "/home/testuser/private/.env",
                    r"C:\Users\TestUser\private\.env",
                    "sk-leaktoken1234567890",
                    "dpapi:v1:c2VjcmV0Y2lwaGVy",
                ]
            )
        self.assertEqual(rc2, 2)
        stderr2 = err2.getvalue()
        for forbidden in (
            "/home/testuser",
            r"C:\Users\TestUser",
            "sk-leaktoken1234567890",
            "c2VjcmV0Y2lwaGVy",
            "Traceback",
        ):
            self.assertNotIn(forbidden, stderr2)

        # Invalid DPAPI scope
        out3, err3 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out3), contextlib.redirect_stderr(err3):
            rc3 = dpapi_vault_cli.main(["seal", "--scope", "InvalidScope"])
        self.assertEqual(rc3, 2)
        self.assertNotIn("Traceback", err3.getvalue())

        # Invalid --key value fails cleanly without echoing bad input
        out4, err4 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out4), contextlib.redirect_stderr(err4):
            rc4 = dpapi_vault_cli.main(["status", "--key", "BAD;INJECT/home/testuser"])
        self.assertEqual(rc4, 1)
        self.assertNotIn("BAD;INJECT", err4.getvalue())
        self.assertNotIn("/home/testuser", err4.getvalue())
        self.assertNotIn("Traceback", err4.getvalue())

        # --help succeeds cleanly without exposing local paths
        out_help, err_help = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_help), contextlib.redirect_stderr(err_help):
            rc_help = dpapi_vault_cli.main(["--help"])
        self.assertEqual(rc_help, 0)
        self.assertIn("dpapi_vault_cli.py", out_help.getvalue())
        self.assertNotIn(str(Path.home()), out_help.getvalue())

    def test_standalone_subprocess_stdlib_only_status_dry_run_and_fail_closed(self) -> None:
        cli_script = REPO_ROOT / "scripts" / "dpapi_vault_cli.py"
        plain_secret = "mm-subprocess-secret-7777"
        home_dir = str(Path.home())

        with tempfile.TemporaryDirectory(prefix="mm-vault-subproc-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original = f"MINIMIND_API_KEY={plain_secret}\nCUSTOM_TOKEN={plain_secret}\n"
            env_path.write_text(original, encoding="utf-8")

            child_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            # 1. status --json in isolated stdlib-only mode (-I -S -B)
            p_status = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(cli_script), "status", "--env-file", str(env_path), "--json"],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(p_status.returncode, 0, p_status.stderr)
            self.assertEqual(p_status.stderr, "")
            status_payload = json.loads(p_status.stdout)
            self.assertEqual(status_payload["env_file"], ".env")
            self.assertEqual(status_payload["keys"]["MINIMIND_API_KEY"]["effective"], "plaintext")
            self.assertEqual(status_payload["keys"]["CUSTOM_TOKEN"]["effective"], "plaintext")
            for forbidden in (plain_secret, tmp_dir, home_dir, "Traceback"):
                self.assertNotIn(forbidden, p_status.stdout)

            # 2. seal --dry-run --json in isolated stdlib-only mode (-I -S -B)
            p_dry = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(cli_script),
                    "seal",
                    "--dry-run",
                    "--env-file",
                    str(env_path),
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(p_dry.returncode, 0, p_dry.stderr)
            dry_payload = json.loads(p_dry.stdout)
            self.assertTrue(dry_payload["dry_run"])
            self.assertFalse(dry_payload["changed"])
            self.assertTrue(dry_payload["would_change"])
            self.assertEqual(dry_payload["would_seal_keys"], ["MINIMIND_API_KEY", "CUSTOM_TOKEN"])
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)
            for forbidden in (plain_secret, tmp_dir, home_dir):
                self.assertNotIn(forbidden, p_dry.stdout)

            # 3. seal without DPAPI on non-Windows fails closed
            if sys.platform != "win32":
                p_seal = subprocess.run(
                    [sys.executable, "-I", "-S", "-B", str(cli_script), "seal", "--env-file", str(env_path)],
                    capture_output=True,
                    text=True,
                    env=child_env,
                    check=False,
                    timeout=15,
                )
                self.assertEqual(p_seal.returncode, 1)
                self.assertEqual(env_path.read_text(encoding="utf-8"), original)
                for forbidden in (plain_secret, tmp_dir, home_dir, "Traceback"):
                    self.assertNotIn(forbidden, p_seal.stderr)
                    self.assertNotIn(forbidden, p_seal.stdout)

            # 4. Invalid CLI args exit 2 without leaking paths or traceback
            p_bad = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(cli_script), "status", "--bogus", "/home/testuser/x"],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(p_bad.returncode, 2)
            self.assertNotIn("/home/testuser", p_bad.stderr)
            self.assertNotIn(home_dir, p_bad.stderr)
            self.assertNotIn("Traceback", p_bad.stderr)


if __name__ == "__main__":
    unittest.main()
