# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ingest vLLM benchmark JSON results into ClickHouse.

Expects the following environment variables:
  CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
  CLICKHOUSE_PASS, CLICKHOUSE_DB
"""

import glob
import json
import logging
import os
import re
import sys
import time
import uuid
from argparse import ArgumentParser
from typing import Any

import clickhouse_connect
from utils import read_benchmark_results

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RESULTS_TABLE = "results_v3"
# Upstream-aligned table (pytorch/test-infra benchmark.oss_ci_benchmark_v3) plus our run_id.
# Written ALONGSIDE results_v3, never instead of it: the HUD reads the flat table today.
VLLM_V3_TABLE = "vllm_results_v3"

# uuid5 namespace for v2 run identity. Must match the frameworks writers byte for byte --
# a different namespace mints a run_id that joins to nothing.
V2_NAMESPACE = uuid.UUID("cb0af9bf-2858-5eab-9211-f51190531bf3")


def v2_canonical_arch(arch: str) -> str:
    """amd64/x86 fold to x86_64. Folded INSIDE the hash, so both spellings of one machine
    produce ONE run_id -- otherwise the same run lands twice, unjoinable to each other."""
    a = (arch or "").strip()
    return "x86_64" if a in ("amd64", "x86") else a


def v2_run_id(source: str, external_run_id: str, arch: str, test_type: str) -> str:
    """uuid5 over source|external_run_id|arch|test_type. Empty on incomplete input:
    a partial key would collide every such run onto one id."""
    src = (source or "").strip()
    ext = (external_run_id or "").strip()
    a = v2_canonical_arch(arch)
    tt = (test_type or "").strip()
    if not (src and ext and a and tt):
        return ""
    return str(uuid.uuid5(V2_NAMESPACE, f"{src}|{ext}|{a}|{tt}"))


METADATA_TABLE = "run_metadata"


def parse_args() -> Any:
    parser = ArgumentParser("Ingest vLLM benchmark results into ClickHouse")

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="directory containing benchmark result JSON files",
    )
    parser.add_argument("--workflow", type=str, default="vLLM Benchmark")
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--sha", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--job-id", type=str, default="0")
    parser.add_argument("--pr-number", type=str, default="0")
    parser.add_argument(
        "--arch",
        type=str,
        default=os.environ.get("BENCHMARK_ARCH", "x86_64"),
        help="hardware architecture the benchmark ran on (e.g. x86_64, ppc64le, s390x)",
    )
    parser.add_argument(
        "--v2-run-id",
        type=str,
        default=os.environ.get("V2_RUN_ID", ""),
        help="An ALREADY-DERIVED v2 run_id (a uuid). Jenkins passes the orchestrator's own "
        "params.RUN_ID here and it is used VERBATIM -- re-hashing an already-hashed id "
        "mints a third identity that joins to nothing. Mutually exclusive with the "
        "derive-from-GHA path below.",
    )
    parser.add_argument(
        "--gha-run-id",
        type=str,
        default=os.environ.get("GITHUB_RUN_ID", ""),
        help="GitHub Actions run id. Used to DERIVE a v2 run_id when --v2-run-id is absent. "
        "Distinct flags rather than sniffing the shape of one value: a numeric id and a "
        "uuid must not be told apart by guessing.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default=os.environ.get("TRIGGER_TYPE", "perf"),
        help="Tier for the run_id hash. 'perf' for a benchmark leg.",
    )
    parser.add_argument(
        "--rpm-lock",
        type=str,
        default=os.environ.get("SPYRE_RPM_LOCK", "spyre-rpms.lock"),
        help="Path to spyre-rpms.lock. On the GHA path this file IS the content identity of "
        "the stack under test -- the leg builds nothing, it restores a cache keyed on this "
        "file -- so each pinned RPM's artifact_id is recovered from it and an "
        "artifact_results row is written per artifact. Empty disables the artifact write.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print rows instead of inserting into ClickHouse",
    )

    return parser.parse_args()


# Scalar metrics per vLLM bench schema. Each entry is the JSON key vLLM writes
# to --output-json (latency, throughput) or --result-filename (serve). Only
# scalar (single-number) metrics are ingested; list fields (latencies, itls,
# ttfts, ...) are the raw samples behind these aggregates and are skipped.
_LATENCY_METRICS = ("avg_latency",)
_THROUGHPUT_METRICS = (
    "elapsed_time",
    "requests_per_second",
    "tokens_per_second",
)
_SERVE_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)


def extract_vllm_metrics(record: dict[str, Any]) -> list[tuple[str, float]]:
    """Return (metric_name, value) pairs from one vLLM-native benchmark record.

    Detects the vLLM bench schema (latency / throughput / serve) by the keys
    the record carries and pulls out the scalar metrics for each. The three
    schemas are disjoint on their signature keys, so a record maps to exactly
    one. `percentiles` (latency) is a nested {percentile: value} dict and is
    flattened to `p{percentile}_latency` metrics.
    """
    pairs: list[tuple[str, float]] = []

    def _add(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        pairs.append((name, float(value)))

    if "avg_latency" in record:
        for key in _LATENCY_METRICS:
            if key in record:
                _add(key, record[key])
        percentiles = record.get("percentiles")
        if isinstance(percentiles, dict):
            for pct, value in percentiles.items():
                _add(f"p{pct}_latency", value)
    elif "requests_per_second" in record or "tokens_per_second" in record:
        for key in _THROUGHPUT_METRICS:
            if key in record:
                _add(key, record[key])
    elif "request_throughput" in record or "output_throughput" in record:
        for key in _SERVE_METRICS:
            if key in record:
                _add(key, record[key])

    return pairs


def extract_pytorch_metrics(record: dict[str, Any]) -> list[tuple[str, float]]:
    """Return (metric_name, value) pairs from one PyTorch-format record.

    This is the schema `convert_to_pytorch_benchmark_format` writes to
    `*.pytorch.json`, produced only when SAVE_TO_PYTORCH_BENCHMARK_FORMAT is
    set. Kept for compatibility with runs that enable it.
    """
    if "benchmark" not in record or "metric" not in record:
        return []
    metric = record["metric"]
    metric_name = metric.get("name", "unknown")
    return [(metric_name, float(v)) for v in metric.get("benchmark_values", [])]


def _model_from_record(record: dict[str, Any], filename: str) -> str:
    """Best-effort model name from a benchmark record, falling back to the file."""
    # vLLM-native JSON writes "model" as a top-level string
    raw_model = record.get("model")
    if isinstance(raw_model, str) and raw_model:
        return raw_model

    benchmark = record.get("benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    model_info = raw_model if isinstance(raw_model, dict) else {}
    return (
        benchmark.get("model")
        or benchmark.get("model_name")
        or model_info.get("name")
        or record.get("model_id")
        or filename.replace(".pytorch.json", "").replace(".json", "")
    )


def extract_rows(
    results_dir: str,
    branch: str,
    sha: str,
    run_id: str,
    job_id: str,
    workflow: str,
    pr_number: int,
    arch: str = "x86_64",
) -> list[dict[str, Any]]:
    """Extract ClickHouse rows from vLLM benchmark JSON files.

    The vLLM benchmark runner writes native `{test_name}.json` files
    (latency / throughput / serve schemas). When SAVE_TO_PYTORCH_BENCHMARK_FORMAT
    is set it ALSO writes `{test_name}.pytorch.json`. This reads both: the
    PyTorch-format files via their `benchmark`/`metric` schema, and every other
    `*.json` via the native vLLM schema. A `.pytorch.json` file is not read
    twice (it is excluded from the native pass).
    """
    rows = []
    ts = int(time.time() * 1000)

    all_json = set(glob.glob(f"{results_dir}/*.json"))
    pytorch_files = set(glob.glob(f"{results_dir}/*.pytorch.json"))
    native_files = sorted(all_json - pytorch_files)
    log.info(
        "Found %d vLLM-native and %d PyTorch-format benchmark JSON files in %s",
        len(native_files),
        len(pytorch_files),
        results_dir,
    )

    def _emit(filename: str, model: str, metric_name: str, value: float) -> None:
        extra = json.dumps(
            {
                "device": "spyre",
                "arch": arch,
                "hardware_type": "IBM_Spyre",
                "model": model,
                "test_name": filename.replace(".pytorch.json", "").replace(".json", ""),
                "head_sha": sha,
                "pr_number": pr_number,
                "value": value,
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "schema_version": "v3",
                "name": "spyre_e2e_benchmark",
                "metric": metric_name,
                "actual": float(value),
                "target": 0.0,
                "repo": "spyre-inference",
                "head_branch": branch,
                "workflow_id": int(run_id) if run_id.isdigit() else 0,
                "job_id": int(job_id) if job_id.isdigit() else 0,
                "run_attempt": 1,
                "extra": extra,
            }
        )

    for file, extractor in [
        *[(f, extract_pytorch_metrics) for f in sorted(pytorch_files)],
        *[(f, extract_vllm_metrics) for f in native_files],
    ]:
        filename = os.path.basename(file)

        try:
            records = read_benchmark_results(file)
        except Exception:
            log.exception("Failed to read benchmark results from %s", filename)
            continue

        if not records:
            log.warning("No results in %s", filename)
            continue

        before_rows = len(rows)

        for record in records:
            if not isinstance(record, dict):
                continue
            model = _model_from_record(record, filename)
            for metric_name, value in extractor(record):
                _emit(filename, model, metric_name, value)

        extracted = len(rows) - before_rows
        if extracted:
            log.info("Extracted %d rows from %s", extracted, filename)
        else:
            log.warning("No usable metrics in %s", filename)

    log.info("Total rows extracted: %d", len(rows))
    return rows


# ── GHA artifact identity ────────────────────────────────────────────────────────────────
# A GHA perf leg builds nothing: it restores a cache keyed on spyre-rpms.lock and extracts
# those exact RPMs. So the honest content identity of what it measured is the LOCK, and the
# artifact_id of each pinned RPM is recoverable from it -- the builder embeds the same id12
# in the NEVRA that it puts in artifact_id and in the artifact_refs glob.
#   NEVRA: ibm-flex-2.0.0-0.main.495+495.a86bb35a.3a6b688cc40a.a86bb35.el10
#                                                 ^^^^^^^^^^^^ id12
# This is why the GHA path does NOT need a digest threaded from Jenkins.

_NEVRA_ID12 = re.compile(r"\.([0-9a-f]{12})\.")
# name-<version>... : the package name is everything before the first -<digit>.
_NEVRA_NAME = re.compile(r"^(.+?)-\d")


def parse_rpm_lock(lock_path: str) -> list[tuple[str, str]]:
    """[(package_name, id12)] for each pinned RPM. Skips any line without exactly one
    id12-shaped token rather than guessing which to take -- a wrong artifact_id is worse
    than an absent one, because it attributes results to the wrong build.
    """
    out: list[tuple[str, str]] = []
    try:
        with open(lock_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return out
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids = _NEVRA_ID12.findall(line)
        name = _NEVRA_NAME.match(line)
        if len(ids) == 1 and name:
            out.append((name.group(1), ids[0]))
        else:
            log.warning("spyre-rpms.lock: cannot derive a unique id12 from %r — skipped", line)
    return out


def rpm_artifact_ids(lock_path: str, arch: str) -> list[str]:
    """artifact_id for every RPM the leg installed: {component}|{name}|{id12}|{arch}.

    The component is the RPM name minus its `ibm-` vendor prefix and any `-devel`/`-headers`
    suffix, which is how the builder names it. Verified against prod: `ibm-flex-devel` and
    `ibm-flex` share one id12 and one component, so the two rows collapse to one artifact.
    """
    a = v2_canonical_arch(arch)
    if not a:
        return []
    seen: dict[str, None] = {}
    for name, id12 in parse_rpm_lock(lock_path):
        base = name[4:] if name.startswith("ibm-") else name
        # -devel/-headers ship alongside the base package from ONE build and share its id12.
        # Verified against prod: the builder registers only the base name (ibm-flex, never
        # ibm-flex-devel), so a per-subpackage artifact_id would name a row that cannot exist.
        for suffix in ("-devel", "-headers"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        component = base
        for suffix in ("-core", "-dd2", "-e2e"):
            if component.endswith(suffix):
                component = component[: -len(suffix)]
                break
        artifact_name = base if base.startswith("ibm-") else f"ibm-{base}"
        seen.setdefault(f"{component}|{artifact_name}|{id12}|{a}", None)
    return list(seen)


def _parse_input_shapes(test_name: str) -> dict[str, Any]:
    """tp1_in64_out64 -> the shape Map upstream carries in `inputs`.

    These discriminate two runs of the SAME benchmark. The flat table kept them only
    inside the test_name string, so a tp1 and a tp4 result were indistinguishable
    without substring parsing at read time.
    """
    out: dict[str, Any] = {}
    for token in (test_name or "").split("_"):
        for prefix, key in (("tp", "tensor_parallel"), ("in", "input_len"), ("out", "output_len")):
            rest = token[len(prefix) :]
            if token.startswith(prefix) and rest.isdigit():
                out[key] = ("int", {"value": rest})
    return out


def to_vllm_v3_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    """Reshape flat results_v3 rows into the upstream-aligned nested shape.

    Derived from the SAME rows the flat table gets, so the two tables can never disagree
    about a number -- only about shape. Recovers four things the flat write dropped:
    head_sha as a real column (blank on every flat row, hidden in extra), the benchmark
    name (the flat write hardcodes one constant for all benchmarks), model.backend (the
    HUD's pivot axis, absent entirely), and the metric samples as an Array.
    """
    out = []
    for r in rows:
        extra = json.loads(r["extra"])
        test_name = extra.get("test_name", "")
        # vLLM's own mode: latency | throughput | serve. Already the test_name prefix.
        mode = test_name.split("_")[0] if test_name else ""
        out.append(
            {
                "run_id": run_id,
                "timestamp": r["timestamp"],
                "schema_version": "v3",
                # The benchmark name, not a constant -- this is what makes per-benchmark
                # history possible at all.
                "name": test_name or r["metric"],
                "repo": r["repo"],
                "head_branch": r["head_branch"],
                "head_sha": extra.get("head_sha", ""),
                "workflow_id": r["workflow_id"],
                "run_attempt": int(r.get("run_attempt") or 0),
                "job_id": int(r.get("job_id") or 0),
                "runners": [(extra.get("hardware_type", ""), extra.get("device", ""))],
                "benchmark": (test_name, mode, "", {}),
                # backend is a COLUMN, never a hash input: it is the axis a cross-backend
                # comparison pivots ON, so folding it into identity would make the two
                # sides of the comparison different benchmarks. That was v1's mistake.
                "model": (
                    extra.get("model", ""),
                    "llm",
                    extra.get("device", ""),
                    ["huggingface"],
                ),
                # Array, so variance and percentiles stay recomputable downstream.
                "metric": (r["metric"], [float(r["actual"])], float(r["target"] or 0.0), {}),
                "inputs": _parse_input_shapes(test_name),
                "dependencies": {},
                # Only what is NOT already a first-class column, so the same value is not
                # stored twice and cannot drift between the two copies.
                "extra": {
                    k: str(v)
                    for k, v in extra.items()
                    if k not in ("head_sha", "model", "test_name")
                },
            }
        )
    return out


def resolve_v2_run_id(args) -> str:
    """The v2 run_id for this leg, or "" when it cannot be derived.

    Two paths, kept explicit. Jenkins already HOLDS the orchestrator's v2 run_id
    (params.RUN_ID) -- use it verbatim. GHA holds only its own integer run id, so the id is
    derived from (gha, run id, arch, test_type). Empty means the v2 write is skipped rather
    than writing an unjoinable row, which downstream cannot tell apart from "no perf ran".
    """
    verbatim = (getattr(args, "v2_run_id", "") or "").strip()
    if verbatim:
        try:
            uuid.UUID(verbatim)
        except (ValueError, AttributeError, TypeError):
            log.warning("--v2-run-id %r is not a uuid; v2 rows skipped", verbatim)
            return ""
        return verbatim
    gha = (getattr(args, "gha_run_id", "") or "").strip()
    if not gha:
        return ""
    return v2_run_id("gha", gha, args.arch, getattr(args, "test_type", "perf"))


def _write_artifact_results(client, v2_rows, run_id_value: str, rpm_lock: str, arch: str) -> None:
    """One artifact_results row per RPM the leg installed, linking perf to what it measured.

    Why per RPM and not one row: a GHA perf leg has no single built image. Its stack is the set
    of pinned RPMs, so every one of them is an artifact the run exercised, and pointing the
    result at all of them is what makes each component's artifact page show the perf that ran
    against it.

    result_kind='performance' with test_type='perf', matching the rows Jenkins pushArtifactResult
    already writes -- this is the same contract from the other launcher, not a new one.

    Contained: this is the FIRST writer to artifact_results from Actions (prod has 4,921 Jenkins
    rows and zero GHA), so a failure here must not cost the benchmark rows already written.
    """
    if not rpm_lock:
        return
    try:
        ids = rpm_artifact_ids(rpm_lock, arch)
        if not ids:
            log.info("no artifact_id derivable from %s — artifact link skipped", rpm_lock)
            return
        if not client.command("EXISTS TABLE artifact_results"):
            log.info("artifact_results absent — artifact link skipped")
            return
        # artifact_results is a plain MergeTree with no dedup key, so a re-ingest of one leg
        # DOUBLES its rows -- and every per-artifact counter is derived from them. Check first.
        already = client.query(
            "SELECT count() FROM artifact_results "
            "WHERE run_id = {rid:UUID} AND result_kind = 'performance'",
            parameters={"rid": run_id_value},
        ).result_rows
        if already and already[0][0] > 0:
            log.info("artifact link already present for run_id=%s — skipping", run_id_value)
            return
        first = v2_rows[0]
        cols = [
            "artifact_id",
            "run_id",
            "result_kind",
            "test_type",
            "state",
            "arch",
            "total_tests",
            "passed",
            "failed",
            "errors",
            "skipped",
            "duration_s",
            "props",
        ]
        # total_tests counts BENCHMARKS, not metrics: 26 metrics of one benchmark is one
        # measurement, and counting metrics would inflate every perf leg ~26x.
        benchmarks = len({r["name"] for r in v2_rows})
        rows = [
            [
                aid,
                run_id_value,
                "performance",
                "perf",
                "passed",
                v2_canonical_arch(arch),
                benchmarks,
                benchmarks,
                0,
                0,
                0,
                0.0,
                {
                    "source": "gha",
                    "workflow_id": str(first["workflow_id"]),
                    "rpm_lock": rpm_lock,
                    "head_sha": first["head_sha"],
                },
            ]
            for aid in ids
        ]
        client.insert("artifact_results", rows, column_names=cols)
        log.info("Linked %d artifact(s) to run_id=%s in artifact_results", len(rows), run_id_value)
    except Exception as exc:  # noqa: BLE001
        log.warning("artifact_results link failed, benchmark rows unaffected: %r", exc)


def insert_to_clickhouse(
    rows: list[dict[str, Any]],
    v2_run_id_value: str = "",
    rpm_lock: str = "",
    arch: str = "",
) -> None:
    """Insert rows into ClickHouse using environment-configured connection."""
    clickhouse_env_vars = {
        "CLICKHOUSE_HOST": os.environ.get("CLICKHOUSE_HOST"),
        "CLICKHOUSE_USER": os.environ.get("CLICKHOUSE_USER"),
        "CLICKHOUSE_PASS": os.environ.get("CLICKHOUSE_PASS"),
        "CLICKHOUSE_DB": os.environ.get("CLICKHOUSE_DB"),
    }
    missing = [k for k, v in clickhouse_env_vars.items() if not v]
    if missing:
        raise OSError(f"Missing required environment variables: {', '.join(missing)}")

    host = clickhouse_env_vars["CLICKHOUSE_HOST"]
    port = int(os.environ.get("CLICKHOUSE_PORT") or "8123")
    user = clickhouse_env_vars["CLICKHOUSE_USER"]
    password = clickhouse_env_vars["CLICKHOUSE_PASS"]
    database = clickhouse_env_vars["CLICKHOUSE_DB"]

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
    )

    if not rows:
        log.warning("No rows to insert")
        return

    columns = list(rows[0].keys())
    data = [[row[col] for col in columns] for row in rows]

    client.insert(
        RESULTS_TABLE,
        data,
        column_names=columns,
    )
    log.info("Inserted %d rows into %s", len(rows), RESULTS_TABLE)

    # Upstream-aligned copy, additive. Contained: results_v3 is what the HUD reads today, so
    # a failure here must never cost those rows -- and an absent table is the normal state
    # until the DDL lands, not an error.
    if v2_run_id_value:
        try:
            if client.command(f"EXISTS TABLE {VLLM_V3_TABLE}"):
                v2_rows = to_vllm_v3_rows(rows, v2_run_id_value)
                v2_cols = list(v2_rows[0].keys())
                client.insert(
                    VLLM_V3_TABLE,
                    [[r[c] for c in v2_cols] for r in v2_rows],
                    column_names=v2_cols,
                )
                log.info(
                    "Inserted %d rows into %s under run_id=%s",
                    len(v2_rows),
                    VLLM_V3_TABLE,
                    v2_run_id_value,
                )
                _write_artifact_results(client, v2_rows, v2_run_id_value, rpm_lock, arch)
            else:
                log.info("%s absent — upstream-shaped rows skipped", VLLM_V3_TABLE)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s write failed, %s unaffected: %r", VLLM_V3_TABLE, RESULTS_TABLE, exc)
    else:
        # Loud: without a run_id the perf numbers cannot reach an artifact, and a blank
        # artifact page reads as "no perf ran" rather than "not linked".
        log.warning(
            "no v2 run_id (pass --v2-run-id on Jenkins, or --gha-run-id + --arch on Actions) "
            "— upstream-shaped rows skipped, %s still written",
            RESULTS_TABLE,
        )

    # Insert metadata rows (required for dashboard commit picker)
    metadata_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        extra_data = json.loads(row["extra"])
        key = (row["workflow_id"], row["metric"], extra_data.get("model", ""))
        if key in seen:
            continue
        seen.add(key)
        metadata_rows.append(
            {
                "timestamp": row["timestamp"],
                "repo": row["repo"],
                "head_branch": row["head_branch"],
                "head_sha": extra_data.get("head_sha", ""),
                "workflow_id": row["workflow_id"],
                "benchmark_name": row["name"],
                "model_name": extra_data.get("model", ""),
                "metric_name": row["metric"],
                "device": extra_data.get("device", "spyre"),
                "arch": extra_data.get("arch", "x86_64"),
            }
        )

    if metadata_rows:
        meta_columns = list(metadata_rows[0].keys())
        meta_data = [[r[col] for col in meta_columns] for r in metadata_rows]
        client.insert(METADATA_TABLE, meta_data, column_names=meta_columns)
        log.info("Inserted %d rows into %s", len(metadata_rows), METADATA_TABLE)


def main() -> None:
    args = parse_args()

    pr_number = int(args.pr_number) if args.pr_number else 0

    rows = extract_rows(
        results_dir=args.results_dir,
        branch=args.branch,
        sha=args.sha,
        run_id=args.run_id,
        job_id=args.job_id,
        workflow=args.workflow,
        pr_number=pr_number,
        arch=args.arch,
    )

    if not rows:
        log.warning("No benchmark results found in %s", args.results_dir)
        sys.exit(1)

    if args.dry_run:
        log.info("Dry run: would insert %d rows:", len(rows))
        for row in rows[:5]:
            print(json.dumps(row, indent=2))
        if len(rows) > 5:
            print(f"... and {len(rows) - 5} more")
        return

    insert_to_clickhouse(rows, resolve_v2_run_id(args), getattr(args, "rpm_lock", ""), args.arch)


if __name__ == "__main__":
    main()
