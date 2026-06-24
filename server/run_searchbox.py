#!/usr/bin/env python3
"""Orchestrator: run one searchbox job on a minimal Pi harness.

Inputs: a PROMPT, a DATAROOM (.zip or folder), and a BUDGET in TURNS.
The agent answers the prompt grounded ONLY in the dataroom, using two local-only retrieval tools
(jina-embeddings-v5-text-small + jina-reranker-v3 over the dataroom). No web.

DESIGN: keep it minimal. Pi is already a complete agent (it loops, calls tools, and
auto-compacts its own context). We add exactly ONE thing on top of vanilla Pi: do not let the
run finish until it has used its BUDGET of turns. So this file only:
  1. unzips the dataroom + boots the retrieval sidecar,
  2. sends the task once,
  3. re-nudges ("keep going") each time Pi goes idle while turns remain,
  4. stops when the turn budget is used and ANSWER.md exists.
No STATUS files, no saturation/consolidation machinery, no extra prompt scaffolding.

BUDGET = NUMBER OF TURNS. A "turn" is one agent cycle (one Continue. nudge -> agent works -> idle).
We stop at `turn >= budget`. Turns are user-legible and the only thing we can cleanly stop on at a
cycle boundary (we cannot interrupt mid-turn anyway).

TOKEN ACCOUNTING (still recorded every turn, just not the budget): pi has no cumulative counter,
and get_session_stats sums only in-memory messages, which compaction prunes - so it UNDERCOUNTS a
long run. The session JSONL is append-only (compaction only appends a summary; it never deletes
the assistant entries that carry usage), so we sum usage from the session file. We use the SAME
fields pi uses (input/output/cacheRead/cacheWrite) and persist them per-turn (turns.jsonl) and in
run_meta.json for the UI - the experiment still sees token cost per turn; it just isn't the stop
condition.

ABLATION (all via env, no code edits):
  base model : LLAMA_URL / MODEL_ID / CONTEXT_WINDOW
  tools      : SEARCHBOX_TOOLS="sentence_embed,passage_rerank" | "sentence_embed" | "" (none)
  retrieval  : EMBED_MODEL / RERANK_MODEL
  budget     : TURN_BUDGET (number of turns)
"""
import argparse, json, os, subprocess, sys, time, zipfile, signal, socket, threading, urllib.request, urllib.error, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Which session-usage field the per-turn TOKEN ACCOUNTING reports (informational only; the budget
# is now a TURN count). Default: fresh prefill ("input").
BUDGET_METRIC = os.environ.get("BUDGET_METRIC", "input")


class _BudgetAlreadySpent(Exception):
    """Resume re-entered drive() with the turn budget already fully used; stop without a cycle."""


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def write_pi_config(agent_dir: Path, llama_url: str):
    """Per-job Pi config: default model = the configured local model.

    Compaction is ON. We DO tune reserveTokens: Pi auto-compacts when
    contextTokens > contextWindow - reserveTokens (default reserve 16384, so trigger ~114688 of
    131072). But compaction itself calls the LLM to summarize, and that request carries
    keepRecentTokens (~20000) of recent context PLUS the history to summarize PLUS maxTokens of
    output. With only ~16k headroom the summarization request itself overflows the context window
    -> 'Auto-compaction failed: Summarization failed: Context size has been exceeded' (observed on
    long/resumed jobs). Fixing by triggering compaction EARLIER (larger reserveTokens) so the
    summarization call has room. reserve 40000 -> trigger ~91072, leaving ~40k for the summary
    request. Tunable via COMPACTION_RESERVE_TOKENS."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    ctx = int(os.environ.get("CONTEXT_WINDOW", os.environ.get("CTX_SIZE", "131072")))
    model_id = os.environ.get("MODEL_ID", "qwen3.6")
    max_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))
    compaction_reserve = int(os.environ.get("COMPACTION_RESERVE_TOKENS", "40000"))
    compaction_keep_recent = int(os.environ.get("COMPACTION_KEEP_RECENT_TOKENS", "20000"))
    (agent_dir / "models.json").write_text(json.dumps({
        "providers": {
            "local": {
                "baseUrl": f"{llama_url}/v1",
                "api": "openai-completions",
                "apiKey": os.environ.get("LLAMA_API_KEY", "sk-local"),
                "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                "models": [{"id": model_id, "contextWindow": ctx, "maxTokens": max_tokens}],
            }
        }
    }, indent=2))
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultProvider": "local",
        "defaultModel": model_id,
        "defaultThinkingLevel": os.environ.get("THINKING_LEVEL", "high"),
        "enableInstallTelemetry": False,
        # Trigger auto-compaction earlier so the summarization request itself fits (see docstring).
        "compaction": {
            "enabled": True,
            "reserveTokens": compaction_reserve,
            "keepRecentTokens": compaction_keep_recent,
        },
    }, indent=2))


def boot_dataroom(job_dir: Path, dataroom_dir: Path, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["DATAROOM_DIR"] = str(dataroom_dir)
    env["DATAROOM_PORT"] = str(port)
    env["DATAROOM_CACHE_DIR"] = str(job_dir / ".dataroom_cache")
    # /embed writes jsonl here; this is pi's cwd (parent of dataroom/), readable by the model.
    env["WORK_DIR"] = str(dataroom_dir.parent)
    logf = open(job_dir / "dataroom.log", "a")
    logf.write(f"\n===== DATAROOM SIDECAR @ {time.ctime()} =====\n"); logf.flush()
    proc = subprocess.Popen([sys.executable, str(HERE / "dataroom_service.py")],
                            env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    logf.close()
    return proc


def wait_http(url: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3); return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(2)
    return False


def _strip_single_wrapper(dataroom_dir: Path):
    """If everything sits under one top-level folder (the common zip wrapper, e.g. a GitHub
    `repo-main/` or a hand-made `dataroom/`), hoist its contents up one level so the dataroom root
    is the real content, not a redundant nesting."""
    entries = [p for p in dataroom_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(dataroom_dir / item.name))
        inner.rmdir()


def prepare_dataroom(src: Path, dataroom_dir: Path):
    dataroom_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for item in src.iterdir():
            dst = dataroom_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        _strip_single_wrapper(dataroom_dir)
        return
    if src.is_file() and zipfile.is_zipfile(src):
        base = dataroom_dir.resolve()
        with zipfile.ZipFile(src) as z:
            for member in z.namelist():
                tgt = (dataroom_dir / member).resolve()
                if str(tgt).startswith(str(base)):
                    z.extract(member, dataroom_dir)
        _strip_single_wrapper(dataroom_dir)
        return
    raise SystemExit(f"ERROR: dataroom must be a .zip or a folder: {src}")


def session_file(agent_dir: Path) -> Path | None:
    """The newest Pi session JSONL under the per-job agent dir."""
    sd = agent_dir / "sessions"
    if not sd.exists():
        return None
    files = sorted(sd.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def session_usage(sfile: Path | None) -> dict:
    """Sum assistant-message usage from the append-only session JSONL.

    This is the authoritative, compaction-safe token total (see module docstring). Uses the same
    fields Pi's getSessionStats uses: input / output / cacheRead / cacheWrite."""
    out = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    if not sfile or not sfile.exists():
        return out
    for line in open(sfile, errors="ignore"):
        if '"usage"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        msg = o.get("message") if isinstance(o, dict) else None
        if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
            continue
        u = msg.get("usage")
        if isinstance(u, dict):
            for k in out:
                out[k] += int(u.get(k) or 0)
    out["total"] = out["input"] + out["output"] + out["cacheRead"] + out["cacheWrite"]
    return out


def count_tool_calls(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return sum(1 for raw in open(log_path, "rb") if b'"type":"tool_execution_start"' in raw)


def answer_present(work_dir: Path) -> bool:
    a = work_dir / "ANSWER.md"
    return a.exists() and a.stat().st_size > 200


# The task instruction lives in the SYSTEM PROMPT (appended via --append-system-prompt), not in
# a skill. Reason: a skill body is injected only once as an early user message, so pi's
# compaction summarizes/dilutes it on long runs. The system prompt is present on EVERY turn and
# is never compacted, so the task framing stays stable for the whole budget. There is no skill.
# We ask the model to write ANSWER.md: without that instruction the LLM tends not to enter the
# read->edit working loop (it just answers in chat). The harness ALSO captures the model's final
# non-thinking message to ANSWER.md each turn as a backstop, so we get an answer either way.
SYSTEM_TASK_TMPL = (
    "Today is {today}.\n"
    "Answer the question below using the dataroom/ folder in your working directory as your "
    "source. You have no network access. You can use all tools you have or build new tools or "
    "workflow using existing tools. Write your final answer to ANSWER.md in the working "
    "directory (not inside dataroom/)."
)


def build_system_task():
    """System prompt with the current date injected (so time-relative questions like
    'who left in 2026' have a clock). TZ via SEARCHBOX_TZ env, default America/Los_Angeles."""
    tz_name = os.environ.get("SEARCHBOX_TZ", "America/Los_Angeles")
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        today = datetime.now(ZoneInfo(tz_name)).strftime("%A, %B %d, %Y")
    except Exception:
        from datetime import datetime
        today = datetime.now().strftime("%A, %B %d, %Y")
    return SYSTEM_TASK_TMPL.format(today=today)
# The first user message is just the question (no skill expansion).
TASK_COMMAND = "{query}"
# Sent only to keep the run going until the input-token budget is spent (the one mechanism we
# add over vanilla pi). No guidance on method or content.
KEEP_GOING = "Continue."


def drive(job_dir, work_dir, agent_dir, dataroom_dir, args, budget):
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_SKIP_VERSION_CHECK"] = "1"
    cmd = [os.environ.get("PI_BIN", "pi"), "--mode", "rpc",
           "--no-skills",
           "--append-system-prompt", build_system_task(),
           "--extension", str(REPO / "pi" / "extensions" / "dataroom-search.ts")]
    # On resume, continue the prior pi session (same agent_dir) instead of starting fresh.
    # Bind to the EXACT session file rather than --continue ("most recent"): the token accounting
    # in session_usage() reads session_file() = newest jsonl by mtime, and pi --continue picks the
    # most-recent session. Those happen to agree today (one session per isolated agent_dir), but
    # that is an implicit coupling. Pinning --session <path> makes resume + accounting provably
    # read the SAME file, so a preempted run prefills from its real history and the final token
    # total / answer evolution match an uninterrupted N-turn run. Fall back to --continue only if
    # no session file is found yet (shouldn't happen on a real resume).
    if getattr(args, "resume", False):
        prior_sess = session_file(agent_dir)
        if prior_sess is not None:
            cmd += ["--session", str(prior_sess)]
        else:
            cmd.append("--continue")
    log = open(job_dir / "pi.log", "a")
    log.write(f"\n\n===== RPC SESSION @ {time.ctime()} =====\n"); log.flush()
    log_path = job_dir / "pi.log"

    # cwd = work_dir (sandbox: only dataroom/ + ANSWER.md). Plumbing lives in the parent job_dir.
    proc = subprocess.Popen(cmd, cwd=str(work_dir), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    lock = threading.Lock()

    def send(obj):
        with lock:
            try:
                proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()
            except Exception:
                pass

    hard = {"reason": None}

    def hard_kill(reason):
        hard["reason"] = reason
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    global_wd = threading.Timer(args.max_seconds + 30, lambda: hard_kill("ceiling_seconds"))
    global_wd.start()

    # Wall-clock timing accumulators. pi emits no timing/tps (verified against source), so we
    # stopwatch its event stream here: LLM busy = message_start..message_end, tool busy =
    # tool_execution_start..tool_execution_end. Per-tool totals power the UI breakdown.
    timing = {"llm_ms": 0.0, "tool_ms": 0.0, "by_tool_ms": {}, "by_tool_n": {}}
    tmark = {"msg": None, "tool": {}}
    timing_path = job_dir / "timing.json"

    def record_timing(line: str):
        now = time.time() * 1000.0
        try:
            if '"type":"message_start"' in line:
                tmark["msg"] = now
            elif '"type":"message_end"' in line:
                if tmark["msg"] is not None:
                    timing["llm_ms"] += now - tmark["msg"]; tmark["msg"] = None
            elif '"type":"tool_execution_start"' in line:
                ev = json.loads(line)
                tmark["tool"][ev.get("toolCallId", "")] = (now, ev.get("toolName", "?"))
            elif '"type":"tool_execution_end"' in line:
                ev = json.loads(line)
                cid = ev.get("toolCallId", "")
                st = tmark["tool"].pop(cid, None)
                if st:
                    dur, name = now - st[0], st[1]
                    timing["tool_ms"] += dur
                    timing["by_tool_ms"][name] = timing["by_tool_ms"].get(name, 0.0) + dur
                    timing["by_tool_n"][name] = timing["by_tool_n"].get(name, 0) + 1
        except Exception:
            pass

    def flush_timing(wall_ms):
        try:
            out = {"wall_ms": wall_ms, "llm_ms": round(timing["llm_ms"]),
                   "tool_ms": round(timing["tool_ms"]),
                   "by_tool_ms": {k: round(v) for k, v in timing["by_tool_ms"].items()},
                   "by_tool_n": timing["by_tool_n"]}
            timing_path.write_text(json.dumps(out))
        except Exception:
            pass

    # Per-turn snapshots live in job_dir/snapshots (OUTSIDE work/, so pi's sandbox can't see
    # them - this is meta/experiment data, not dataroom). turns.jsonl gets one row EVERY turn;
    # an ANSWER-t{turn}.md file is written ONLY when ANSWER.md changed vs the previous snapshot
    # (dedup), so the snapshot set shows just the turns where the answer actually evolved.
    snap_dir = job_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    turns_jsonl = snap_dir / "turns.jsonl"
    last_answer = {"body": None}  # body of the most recently SAVED ANSWER-t*.md

    # RESUME CONTINUITY: turns.jsonl and ANSWER-t*.md persist across preempt/resume cycles
    # (each resume re-enters drive()). The driver's turn counter must CONTINUE from the highest
    # turn already recorded, not restart at 0 - otherwise a preempted job writes turn 1,2,3...
    # again and turns.jsonl ends up like [1..5][1..9][1..10] (duplicate/garbage turn ids).
    # Likewise last_answer must be seeded from the last saved snapshot so the change-dedup keeps
    # working across the boundary (else the first turn after resume always re-saves a dup).
    def _prior_turn_and_answer():
        max_turn = 0
        if turns_jsonl.exists():
            try:
                for line in turns_jsonl.read_text(errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line).get("turn")
                    if isinstance(t, int) and t > max_turn:
                        max_turn = t
            except Exception:
                pass
        body = None
        try:
            snaps = sorted(snap_dir.glob("ANSWER-t*.md"),
                           key=lambda p: int(p.stem[len("ANSWER-t"):]) if p.stem[len("ANSWER-t"):].isdigit() else -1)
            if snaps:
                body = snaps[-1].read_text(errors="ignore")
        except Exception:
            pass
        return max_turn, body

    prior_turn, prior_body = _prior_turn_and_answer()
    last_answer["body"] = prior_body

    def snapshot(turn_no: int, spent: int, usage: dict, elapsed: float):
        try:
            ans = work_dir / "ANSWER.md"
            body = ans.read_text(errors="ignore") if ans.exists() else ""
            ans_chars = len(body)
            # Only persist a new ANSWER-t{n}.md when the content differs from the last one we
            # saved. The very first turn always saves (last_answer.body is None).
            answer_changed = body != last_answer["body"]
            if answer_changed:
                (snap_dir / f"ANSWER-t{turn_no}.md").write_text(body)
                last_answer["body"] = body
            row = {
                "turn": turn_no,
                "ts": round(time.time(), 3),
                "elapsed_seconds": round(elapsed, 1),
                "budget": budget,                    # budget is now in TURNS
                "budget_pct": round(100 * turn_no / budget, 2) if budget else None,
                # per-turn TOKEN ACCOUNTING (informational; no longer the stop condition)
                "token_metric": BUDGET_METRIC,
                "tokens_spent": spent,
                "tokens": usage,
                "tool_calls": count_tool_calls(log_path),
                "answer_present": answer_present(work_dir),
                "answer_chars": ans_chars,
                "answer_changed": answer_changed,
                "answer_snapshot": f"ANSWER-t{turn_no}.md" if answer_changed else None,
            }
            with open(turns_jsonl, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:
            pass

    start = time.time()
    # Continue the global turn counter across resumes (see _prior_turn_and_answer above).
    turn, stop_reason = prior_turn, "error_pi_exited"
    sfile = None

    def spent_now():
        nonlocal sfile
        if sfile is None:
            sfile = session_file(agent_dir)
        u = session_usage(sfile)
        return u.get(BUDGET_METRIC, 0), u

    # Fresh run: send the question as the first user message. Resume: the session already has
    # the full history (pi --continue), so just nudge it to keep going.
    if getattr(args, "resume", False):
        send({"type": "prompt", "message": KEEP_GOING})
    else:
        send({"type": "prompt", "message": TASK_COMMAND.format(query=args.query)})

    ans_path = work_dir / "ANSWER.md"

    def extract_text(line: str) -> str:
        """From a message_end event, return the concatenated non-thinking text blocks."""
        try:
            ev = json.loads(line)
            content = (ev.get("message") or {}).get("content") or []
            parts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            return "".join(parts).strip()
        except Exception:
            return ""

    try:
        # If a resume re-entered drive() with the turn budget already fully spent (e.g. the run
        # was preempted right at completion), don't run another wasted cycle - stop immediately.
        if args.force_budget and turn >= budget:
            stop_reason = "budget_spent"
            raise _BudgetAlreadySpent()
        while True:
            cycle_wd = threading.Timer(max(1, args.turn_timeout), lambda: send({"type": "abort"}))
            cycle_wd.start()
            ended = False
            turn_text = ""  # last non-thinking assistant text this turn
            try:
                while True:
                    line = proc.stdout.readline()
                    if line == "":
                        break
                    if '"type":"message_update"' in line:
                        continue
                    log.write(line)
                    record_timing(line)
                    if '"type":"message_end"' in line:
                        t = extract_text(line)
                        if t:
                            turn_text = t
                    if '"type":"agent_end"' in line:
                        ended = True; break
            finally:
                cycle_wd.cancel()
            flush_timing(round((time.time() - start) * 1000))

            # The model is asked to write ANSWER.md itself (drives the read->edit loop). As a
            # BACKSTOP, if it has not produced ANSWER.md yet, save its final non-thinking message
            # this turn so a run always yields an answer. Never clobber a model-written file.
            if turn_text and not ans_path.exists():
                try:
                    ans_path.write_text(turn_text)
                except Exception:
                    pass

            if not ended:
                stop_reason = hard["reason"] or "error_pi_exited"; break

            turn += 1
            log.flush()

            cf = job_dir / "control"
            if cf.exists():
                ctl = cf.read_text(errors="ignore").strip()
                # 'pause' = system preempt, 'hold' = user pause; both checkpoint to paused. 'cancel' legacy.
                if ctl in ("cancel", "pause", "hold"):
                    stop_reason = "cancelled" if ctl == "cancel" else "paused"
                    break

            spent, usage = spent_now()
            elapsed = time.time() - start
            snapshot(turn, spent, usage, elapsed)
            print(f"[searchbox] cycle {turn}/{budget} turns "
                  f"({round(100*turn/budget,1) if budget else 0}%) "
                  f"{BUDGET_METRIC}_tokens={spent} "
                  f"tools={count_tool_calls(log_path)} ans={answer_present(work_dir)}", flush=True)

            if elapsed > args.max_seconds:
                stop_reason = "ceiling_seconds"; break

            # We captured this turn's answer above (ANSWER.md = model's final message text).
            # Force-budget OFF: one natural pass is enough -> stop now.
            if not args.force_budget:
                stop_reason = "first_turn_done"; break

            # Force-budget ON: keep going until the TURN budget is used up.
            if turn >= budget:
                stop_reason = "budget_spent"; break

            send({"type": "prompt", "message": KEEP_GOING})
    except _BudgetAlreadySpent:
        pass
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        global_wd.cancel()
        send({"type": "abort"})
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        log.flush()
        flush_timing(round((time.time() - start) * 1000))

    _, usage = spent_now()
    return turn, stop_reason, usage


def write_run_meta(job_dir: Path, **fields):
    try:
        (job_dir / "run_meta.json").write_text(json.dumps(fields, indent=2))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--dataroom", required=True)
    ap.add_argument("--out", default="./out")
    # --budget is the TURN budget: how many agent cycles to run (force-budget ON).
    ap.add_argument("--budget", type=int, default=int(os.environ.get("TURN_BUDGET", "10")))
    # Force-budget (default ON): budget is a FLOOR - keep nudging "Continue." until the turn
    # budget is used, even if the model already wrote ANSWER.md. When OFF, the run stops right
    # after the model's FIRST turn ends (one natural pass), no "Continue." nudges.
    ap.add_argument("--force-budget", dest="force_budget", action="store_true", default=True)
    ap.add_argument("--no-force-budget", dest="force_budget", action="store_false")
    # Resume a previously-paused/preempted run: keep the on-disk dataroom + pi session, continue.
    ap.add_argument("--resume", action="store_true", default=False)
    # Wall-clock safety backstop only (the turn budget is the real stop condition).
    ap.add_argument("--max-seconds", type=int, default=int(os.environ.get("MAX_SECONDS", "21600")))
    ap.add_argument("--turn-timeout", type=int, default=int(os.environ.get("TURN_TIMEOUT", "1200")))
    args = ap.parse_args()
    if args.budget < 1:
        print("ERROR: --budget (turns) must be >= 1", file=sys.stderr); sys.exit(2)

    signal.signal(signal.SIGTERM, lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()))

    llama_url = os.environ.get("LLAMA_URL", "http://localhost:8080")
    job_dir = Path(args.out).resolve(); job_dir.mkdir(parents=True, exist_ok=True)
    # work_dir is pi's cwd: a clean sandbox holding ONLY dataroom/ and the model's ANSWER.md.
    # All plumbing (input.zip, logs, meta, .pi-agent) stays in job_dir, the parent, which pi's
    # cwd cannot see - so the model can't ls/cat/unzip the raw input or read its own logs.
    work_dir = job_dir / "work"; work_dir.mkdir(parents=True, exist_ok=True)
    dataroom_dir = work_dir / "dataroom"
    agent_dir = job_dir / ".pi-agent"

    # On resume the dataroom is already unzipped on disk and the pi session is continued; do not
    # re-extract (it would reset the sandbox). Only prepare on a fresh run.
    if not (args.resume and dataroom_dir.exists() and any(dataroom_dir.iterdir())):
        prepare_dataroom(Path(args.dataroom).resolve(), dataroom_dir)
    (job_dir / "query.txt").write_text(args.query)

    port = free_port()
    os.environ["DATAROOM_INDEX_URL"] = f"http://127.0.0.1:{port}"
    write_pi_config(agent_dir, llama_url)
    cs = boot_dataroom(job_dir, dataroom_dir, port)
    if not wait_http(f"http://127.0.0.1:{port}/stats", int(os.environ.get("DATAROOM_BOOT_TIMEOUT", "600"))):
        print("ERROR: dataroom sidecar did not come up", file=sys.stderr)
        try:
            os.killpg(os.getpgid(cs.pid), signal.SIGTERM)
        except Exception:
            cs.terminate()
        write_run_meta(job_dir, stop_reason="error_dataroom_boot", turns=0, done=False)
        sys.exit(3)

    start = time.time()
    turn, stop_reason, usage = 0, "error_pi_exited", {}
    try:
        turn, stop_reason, usage = drive(job_dir, work_dir, agent_dir, dataroom_dir, args, args.budget)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        try:
            os.killpg(os.getpgid(cs.pid), signal.SIGTERM)
            try:
                cs.wait(timeout=10)
            except Exception:
                os.killpg(os.getpgid(cs.pid), signal.SIGKILL)
        except Exception:
            cs.terminate()

    spent = usage.get(BUDGET_METRIC, 0)
    done = stop_reason in ("budget_spent", "first_turn_done") and answer_present(work_dir)
    write_run_meta(job_dir, stop_reason=stop_reason, turns=turn, done=done,
                   budget=args.budget,                    # budget is in TURNS
                   budget_pct=round(100*turn/args.budget, 1) if args.budget else None,
                   token_metric=BUDGET_METRIC, tokens_spent=spent,
                   tokens=usage, answer_present=answer_present(work_dir),
                   tool_calls=count_tool_calls(job_dir / "pi.log"),
                   elapsed_seconds=round(time.time() - start, 1),
                   tools_enabled=os.environ.get("SEARCHBOX_TOOLS", "all"),
                   model_id=os.environ.get("MODEL_ID", "qwen3.6"))
    print(f"[searchbox] done (stop_reason={stop_reason}, turns={turn}/{args.budget})")
    print(json.dumps({"out": str(job_dir), "turns": turn, "done": done,
                      "stop_reason": stop_reason, "tokens": usage, "budget_turns": args.budget,
                      "answer": str(work_dir / "ANSWER.md")}))


if __name__ == "__main__":
    main()
