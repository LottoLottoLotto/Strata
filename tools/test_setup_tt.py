"""Tests for setup.py --backend tt, the experimental Tenstorrent P150 engine (tt/): it stops with a clear message on
Windows, hands the run to tt/setup_tt.py before any GPU probe, offers only what the engine runs (the Coder IQ1_M, up to
32K of context), finds the card from the tt-kmd driver's files, and writes the config and BUILD.json the server and
setup read.  No card, no Docker, no network.

    python tools/test_setup_tt.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def load_setup_tt():
    spec = importlib.util.spec_from_file_location("setup_tt", ROOT / "tt" / "setup_tt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = load_setup_tt()


def quiet(fn, *a, **k):
    """fn's result, or ("exit", code) when it stopped; and what it said."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            rc = fn(*a, **k)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue()


class TtBackend(unittest.TestCase):
    def handoff(self, argv, win):
        with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call:
            rc, out = quiet(setup.tt_setup, argv)
        return rc, out, call

    def test_windows_stops_with_the_docs_pointer(self):
        rc, out, call = self.handoff(["--backend", "tt"], win=True)
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("docs/TT_P150.md", out)
        call.assert_not_called()

    def test_linux_hands_over_to_setup_tt_without_the_backend_flag(self):
        for argv in (["--yes", "--backend", "tt", "--port", "8085"], ["--yes", "--backend=tt", "--port", "8085"]):
            rc, out, call = self.handoff(argv, win=False)
            self.assertEqual(rc, 0)
            cmd = call.call_args[0][0]
            self.assertTrue(cmd[1].endswith(str(Path("tt") / "setup_tt.py")))
            self.assertEqual(cmd[2:], ["--yes", "--port", "8085"])

    def test_backend_tt_is_handed_over_before_any_gpu_probe(self):
        """A PC with only a P150 has no nvidia-smi and no amdgpu: setup must not ask them first."""
        probe = mock.Mock(side_effect=AssertionError("probed a GPU"))
        with mock.patch.object(sys, "argv", ["setup.py", "--backend", "tt", "--port", "8085"]), \
                mock.patch.object(setup, "gpus", probe), mock.patch.object(setup, "amd_gpus", probe), \
                mock.patch.object(setup, "data_folder", probe), \
                mock.patch.object(setup, "tt_setup", return_value=0) as handoff:
            rc, _ = quiet(setup.main)
        self.assertEqual(rc, 0)
        handoff.assert_called_once_with(["--backend", "tt", "--port", "8085"])

    def test_setup_tt_finds_the_setup_functions_it_needs(self):
        for name in T.NEEDED:
            self.assertTrue(callable(getattr(setup, name, None)), name)

    def test_replacements_take_every_argument_of_the_function_they_replace(self):
        """#870's lesson (sycl): every `S.<name> = <replacement>` in tt/setup_tt.py must accept all the parameters of
        setup.<name> (or take *args / **kwargs), or setup dies with a TypeError mid-install."""
        import ast
        import inspect
        tree = ast.parse((ROOT / "tt" / "setup_tt.py").read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        checked = 0
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "S"):
                continue
            name = n.targets[0].attr
            fn = n.value if isinstance(n.value, ast.Lambda) else defs.get(getattr(n.value, "id", ""))
            orig = getattr(setup, name, None)
            if fn is None or not callable(orig):
                continue
            a = fn.args
            if a.vararg is not None and a.kwarg is not None:
                checked += 1
                continue
            have = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            for param in inspect.signature(orig).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                self.assertIn(param.name, have, f"setup_tt's {name} lacks setup.{name}'s `{param.name}`")
            checked += 1
        self.assertGreaterEqual(checked, 12)


class Choices(unittest.TestCase):
    """Only what the engine was built and measured for: the Coder IQ1_M, contexts up to 32768, no images."""

    def setUp(self):
        for name in ("FAMILIES", "MODELS", "CONTEXTS"):
            p = mock.patch.object(setup, name, getattr(setup, name))
            p.start()
            self.addCleanup(p.stop)
        T.restrict()

    def parse(self, *argv):
        with mock.patch.object(sys, "argv", ["setup.py", *argv]), \
                mock.patch.object(setup, "data_folder", side_effect=AssertionError("parsed")):
            return quiet(setup.main)

    def test_only_the_coder_iq1_m(self):
        self.assertEqual(list(setup.FAMILIES), ["coder"])
        self.assertEqual(list(setup.MODELS), ["IQ1_M"])
        for argv in (["--model", "IQ3_S"], ["--family", "qwen"], ["--model", "UD-IQ4_XS"]):
            rc, _ = self.parse(*argv)
            self.assertEqual(rc, ("exit", 2), argv)        # argparse: invalid choice, before anything is done

    def test_contexts_stop_at_32k_and_the_recommendation_is_one_of_them(self):
        self.assertEqual(max(setup.CONTEXTS), 32768)
        rec = setup.ram_ctx("IQ1_M", 1024.0)               # the menu preselects CONTEXTS.index(rec)
        self.assertIn(rec, setup.CONTEXTS)

    def test_a_longer_context_or_several_cards_are_refused(self):
        for argv in (["--context", "65536"], ["--context=40000"], ["--gpus", "0,1"], ["--gpu", "1"], ["--calibrate"]):
            rc, out = quiet(T.tt_argv, argv)
            self.assertEqual(rc, ("exit", 1), argv)
        rc, _ = quiet(T.tt_argv, ["--context", "32768"])
        self.assertIsInstance(rc, list)

    def test_no_question_about_what_the_engine_does_not_do(self):
        """Images, the low-RAM mode, the KV precision and the projection are answered (images off even when asked)."""
        rc, out = quiet(T.tt_argv, ["--yes", "--vision", "yes", "--kv", "q4_0", "--parallel", "2"])
        get = lambda k: rc[rc.index(k) + 1]  # noqa: E731
        self.assertEqual(get("--vision"), "none")
        self.assertEqual(rc.count("--vision"), 1)
        self.assertEqual(get("--backend"), "hip")
        self.assertEqual(get("--low-ram"), "off")
        self.assertEqual(get("--experimental-speed-projection"), "off")
        self.assertNotIn("--parallel", rc)
        self.assertIn("images", out)


class Devices(unittest.TestCase):
    """The card from the tt-kmd driver's files (/dev/tenstorrent/N, /sys/class/tenstorrent/tenstorrent!N)."""

    def tree(self, cards, kmd="2.11.0"):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root, True)
        dev, sysfs = root / "dev", root / "sys"
        (dev / "tenstorrent").mkdir(parents=True)
        (sysfs / "module" / "tenstorrent").mkdir(parents=True)
        (sysfs / "module" / "tenstorrent" / "version").write_text(kmd + "\n")
        for n, card in enumerate(cards):
            (dev / "tenstorrent" / str(n)).write_text("")
            cls = sysfs / "class" / "tenstorrent" / f"tenstorrent!{n}"
            (cls / "device").mkdir(parents=True)
            (cls / "device" / "device").write_text("0xb140\n")
            if card:
                (cls / "tt_card_type").write_text(card + "\n")
        return str(dev), str(sysfs)

    def no_smi(self):
        raise AssertionError("tt-smi asked although the driver named the card")

    def test_a_p150_is_found_and_usable(self):
        dev, sysfs = self.tree(["p150a"])
        cards = T.tt_devices(dev, sysfs, snapshot=self.no_smi)
        self.assertEqual(len(cards), 1)
        c = cards[0]
        self.assertEqual((c["index"], c["name"], c["vram_gb"], c["kmd"]), (0, "Tenstorrent Blackhole P150a", 32.0, "2.11.0"))
        self.assertIsNone(T.tt_problem(c))

    def test_no_driver_no_card(self):
        root = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, root, True)
        self.assertEqual(T.tt_devices(root + "/dev", root + "/sys", snapshot=self.no_smi), [])

    def test_other_tenstorrent_cards_are_listed_but_not_used(self):
        dev, sysfs = self.tree(["p100a"])
        c = T.tt_devices(dev, sysfs, snapshot=self.no_smi)[0]
        self.assertIn("P150", T.tt_problem(c))

    def test_a_second_p150_is_not_the_one_the_engine_opens(self):
        dev, sysfs = self.tree(["p150b", "p150b"])
        cards = T.tt_devices(dev, sysfs, snapshot=self.no_smi)
        self.assertEqual([T.tt_problem(c) is None for c in cards], [True, False])

    def test_an_older_driver_without_the_card_type_is_named_by_tt_smi(self):
        dev, sysfs = self.tree([None])
        snap = {"device_info": [{"board_info": {"board_type": "p150b"}}]}
        c = T.tt_devices(dev, sysfs, snapshot=lambda: snap)[0]
        self.assertEqual((c["name"], c["vram_gb"]), ("Tenstorrent Blackhole P150b", 32.0))
        c = T.tt_devices(dev, sysfs, snapshot=lambda: {})[0]   # and without tt-smi: a Blackhole setup cannot size
        self.assertIsNotNone(T.tt_problem(c))


class ToTt(unittest.TestCase):
    """setup's config, written on its HIP path -> the TT engine's."""

    def cfg(self, **kw):
        return {"exe": "/s/tt/serve/strata", "cwd": "/s", "tokenizer": "/d/packs/coder-iq1_m/tokenizer", "port": 8080,
                "model_name": "qwen3.8-flash-next-coder-iq1_m", "log": "/s/strata-coder-iq1_m.log",
                "args": ["--pack", "/d/packs/coder-iq1_m", "--native", "/m/c-00001-of-00002.gguf", "--ple-gguf",
                         "/m/c-00002-of-00002.gguf", "--expert-profile", "/s/data/expert-profile-coder.bin",
                         "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp",
                         "/d/mtp/rt", "--max-context", "65536", "--kv", "int8", "--kv-resident", "32768",
                         "--pool-workers", "7", "--control-vector-scaled", "/x.gguf:1.0", "--control-vector-layer-range",
                         "4", "44", "--resident-experts"],
                "lib_dirs": ["/cuda/lib"], "backend": "hip", "gpu": 0, "gpus_asked": True, "parallel": 2,
                "env": {"STRATA_HIPBLASLT_TUNING": "/t.txt", "FNX_MOE_PLAN": "mine"}, **kw}

    def test_the_engine_gets_its_own_flags_and_the_host_paths(self):
        c = T.to_tt(self.cfg(), "/data/tt")
        self.assertEqual(c["args"], ["--pack", "/d/packs/coder-iq1_m", "--native", "/m/c-00001-of-00002.gguf",
                                     "--ple-gguf", "/m/c-00002-of-00002.gguf", "--prefill", "128", "--spec", "4",
                                     "--spec-min-p", "0.5", "--mtp", "/d/mtp/rt", "--max-context", "32768"])
        self.assertEqual((c["backend"], c["exe"], c["lib_dirs"]), ("tt", str(T.WRAPPER), []))
        for k in ("gpu", "gpus_asked", "parallel", "vision"):
            self.assertNotIn(k, c)

    def test_env_names_the_data_and_keeps_the_users_own(self):
        c = T.to_tt(self.cfg(), "/data/tt")
        self.assertEqual(c["env"], {"FNX_MOE_PLAN": "mine", "FNX_DATA": "/data/tt"})
        c = T.to_tt(self.cfg(), "/data/tt", image="mine/tt:dev")
        self.assertEqual(c["env"]["FNX_IMAGE"], "mine/tt:dev")
        c = T.to_tt(self.cfg(env={"FNX_IMAGE": "old"}), "/data/tt", image=T.DEFAULT_IMAGE)
        self.assertNotIn("FNX_IMAGE", c["env"])

    def test_the_shared_card_is_given_back_when_idle_unless_the_user_chose(self):
        self.assertEqual(T.to_tt(self.cfg(), "/d")["idle_unload_s"], 900)
        self.assertEqual(T.to_tt(self.cfg(idle_unload_s=0), "/d")["idle_unload_s"], 0)

    def test_a_missing_prefill_or_context_is_written(self):
        c = T.to_tt({"exe": "x", "args": ["--native", "/m/a.gguf"]}, "/d")
        self.assertEqual(c["args"], ["--native", "/m/a.gguf", "--max-context", "32768", "--prefill", "128"])


class BuildJson(unittest.TestCase):
    def test_setup_reads_the_numeric_version_and_keeps_the_prompt_chunk(self):
        """Without BUILD.json, setup reads a script engine as 0.1.12 and rewrites `--prefill auto` to 2048."""
        with tempfile.TemporaryDirectory() as d:
            meta = json.loads(T.build_json(Path(d), "img@sha256:1").read_text())
            self.assertEqual(meta, {"backend": "tt", "source": "docker", "image": "img@sha256:1",
                                    "version": setup.source_version(), "lib_dirs": []})
            exe = Path(d) / "tt-engine.sh"
            exe.write_text("#!/bin/sh\n")
            self.assertEqual(setup.engine_version(exe), tuple(int(x) for x in setup.source_version().split(".")))
            cfg_path = Path(d) / "strata-coder-iq1_m.json"
            for prefill in ("128", "auto"):
                cfg = {"exe": str(exe), "args": ["--prefill", prefill]}
                cfg_path.write_text(json.dumps(cfg))
                rc, _ = quiet(setup.upgrade_config, cfg_path, cfg)
                self.assertEqual(rc["args"], ["--prefill", prefill])


class RunScript(unittest.TestCase):
    def test_the_script_starts_server_tt_without_the_ready_timeout_and_passes_arguments_on(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            py = d / "python"
            py.write_text('#!/bin/sh\necho "READY_S=$STRATA_ENGINE_READY_S"\nfor a in "$@"; do echo "$a"; done\n')
            py.chmod(0o755)
            with mock.patch.object(setup, "ROOT", d), mock.patch.object(setup, "WIN", False), \
                    mock.patch.object(setup.sys, "executable", str(py)):
                script = T.run_script(setup.write_run_script, "coder-IQ1_M", d / "strata-coder-iq1_m.json", 8091, False)
            env = {k: v for k, v in os.environ.items() if k != "STRATA_ENGINE_READY_S"}
            out = subprocess.run(["/bin/sh", str(script), "--port", "8092"], capture_output=True, text=True,
                                 env={**env, "STRATA_ENGINE_READY_S": "900"}).stdout.split("\n")
        self.assertEqual(script.name, "run-coder-iq1_m.sh")
        self.assertEqual(out[0], "READY_S=0")
        self.assertEqual(out[1], str(d / "tt" / "serve" / "server_tt.py"))
        self.assertEqual(out[2:10], ["--engine", "strata", "--config", str(d / "strata-coder-iq1_m.json"),
                                     "--port", "8091", "--port", "8092"])


class Windows(unittest.TestCase):
    def test_setup_tt_itself_refuses_windows(self):
        with mock.patch.object(setup, "WIN", True):
            rc, out = quiet(T.install, [])
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("Linux", out)


if __name__ == "__main__":
    unittest.main()
