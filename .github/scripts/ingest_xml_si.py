#!/usr/bin/env python3
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
Parses spyre-inference's pytest JUnit XML into si_test_runs / si_test_cases /
si_run_properties.

Usage (called by the GHA workflow):
    python3 ingest_xml_si.py \
        --xml-dir xml_artifacts \
        --workflow "test_each_commit" \
        --branch   "main" \
        --sha      "abcdef1234..." \
        --run-id   "3f2b9c1e-...-uuid" \
        --gha-run-id "12345678" \
        --triggered-at "2026-04-25T14:20:45Z" \
        --pr-number 2271 \
        --platform "x86_64"
"""

import argparse
import os
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import clickhouse_connect
from lxml import etree

# ---------------------------------------------------------------------------
# si_test_runs / si_test_cases / si_run_properties are provisioned out of
# band; this script never creates them. If they're missing, ingest is a
# silent no-op rather than a failure.
# ---------------------------------------------------------------------------


def tables_exist(client, db: str) -> bool:
    return bool(client.command(f"EXISTS TABLE {db}.si_test_runs"))


# ---------------------------------------------------------------------------
# TEST-RESULT XML parsing
# ---------------------------------------------------------------------------


def classify_testcase(tc_el):
    """Return (status, fail_message, skip_message) for one <testcase>."""
    failure_el = tc_el.find("failure")
    error_el = tc_el.find("error")
    skipped_el = tc_el.find("skipped")

    if error_el is not None:
        msg = (error_el.get("message", "") + "\n" + (error_el.text or "")).strip()
        return "error", msg, ""

    if failure_el is not None:
        ftype = (failure_el.get("type") or "").lower()
        msg = (failure_el.get("message", "") + "\n" + (failure_el.text or "")).strip()
        if "xfail" in ftype:
            return "xpass", msg, ""
        return "failed", msg, ""

    if skipped_el is not None:
        stype = (skipped_el.get("type") or "").lower()
        msg = (skipped_el.get("message") or skipped_el.text or "").strip()
        if "xfail" in stype:
            return "xfail", "", msg
        return "skipped", "", msg

    return "passed", "", ""


def extract_properties(tc_el):
    props = []
    props_el = tc_el.find("properties")
    if props_el is None:
        return props
    for p in props_el.findall("property"):
        name = p.get("name", "").strip()
        value = p.get("value", "").strip()
        if name:
            props.append((name, value))
    return props


def promote_xpass(raw_cases, suite_attrs):
    """Mirror torch-spyre's ingest_xml.py: pytest's plain <failures> count
    lumps strict and non-strict xfail-passed cases together with real
    failures, so bare-passed cases must be promoted to "xpass" to reconcile
    the suite-level failure count."""
    failures = int(suite_attrs.get("failures", 0))
    true_fail_raw = sum(1 for c in raw_cases if c["status"] in ("failed", "error"))
    strict_xpass_raw = sum(1 for c in raw_cases if c["status"] == "xpass")
    non_strict = max(0, failures - true_fail_raw - strict_xpass_raw)

    promoted = 0
    for c in raw_cases:
        if promoted >= non_strict:
            break
        if c["_is_bare"]:
            c["status"] = "xpass"
            promoted += 1


def parse_test_xml(xml_path: Path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    suites = root.findall(".//testsuite")
    if not suites:
        print(f"  [warn] No <testsuite> found in {xml_path.name}", file=sys.stderr)
        return None, []

    suite = suites[0]
    suite_attrs = suite.attrib

    ts_str = suite_attrs.get("timestamp", "")
    try:
        triggered_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        triggered_at = datetime.now(UTC)

    raw_cases = []
    for tc in suite.findall(".//testcase"):
        status, fail_msg, skip_msg = classify_testcase(tc)
        properties = extract_properties(tc)
        raw_cases.append(
            {
                "case_id": str(uuid.uuid4()),
                "classname": tc.get("classname", ""),
                "name": tc.get("name", ""),
                "status": status,
                "duration_s": float(tc.get("time", 0) or 0),
                "fail_message": fail_msg,
                "skip_message": skip_msg,
                "properties": properties,
                "_is_bare": (status == "passed"),
                "triggered_at": triggered_at,
            }
        )

    promote_xpass(raw_cases, suite_attrs)

    counts = Counter(c["status"] for c in raw_cases)
    run = {
        "suite_name": suite_attrs.get("name", xml_path.stem),
        "filename": xml_path.name,
        "triggered_at": triggered_at,
        "total_tests": len(raw_cases),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0) + counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "xfail": counts.get("xfail", 0),
        "errors": counts.get("error", 0),
        "xpass": counts.get("xpass", 0),
        "duration_s": float(suite_attrs.get("time", 0) or 0),
    }
    return run, raw_cases


# ---------------------------------------------------------------------------
# ClickHouse insertion
# ---------------------------------------------------------------------------


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


def insert_run(client, run_id: str, run: dict, args):
    client.insert(
        "si_test_runs",
        [
            [
                run_id,
                args.workflow,
                run["suite_name"],
                run["filename"],
                args.branch,
                (args.sha or "").ljust(40)[:40],
                int(args.pr_number) if args.pr_number.strip() else 0,
                _runner_run_id(args, run_id),
                run["triggered_at"].replace(tzinfo=None),
                run["total_tests"],
                run["passed"],
                run["failed"],
                run["skipped"],
                run["xfail"],
                run["errors"],
                run["xpass"],
                run["duration_s"],
                args.platform,
                args.trigger_type or "unknown",
                args.img_digest,
            ]
        ],
        column_names=[
            "run_id",
            "workflow",
            "suite_name",
            "filename",
            "branch",
            "commit_sha",
            "pr_number",
            "runner_run_id",
            "triggered_at",
            "total_tests",
            "passed",
            "failed",
            "skipped",
            "xfail",
            "errors",
            "xpass",
            "duration_s",
            "platform",
            "test_type",
            "img_digest",
        ],
    )


def insert_cases(client, run_id: str, cases: list[dict], workflow: str = ""):
    if not cases:
        return
    client.insert(
        "si_test_cases",
        [
            [
                run_id,
                c["case_id"],
                c["classname"],
                c["name"],
                c["status"],
                c["duration_s"],
                c["skip_message"][:8192],
                c["fail_message"][:8192],
                c["triggered_at"].replace(tzinfo=None),
                workflow,
            ]
            for c in cases
        ],
        column_names=[
            "run_id",
            "case_id",
            "classname",
            "name",
            "status",
            "duration_s",
            "skip_message",
            "fail_message",
            "triggered_at",
            "workflow",
        ],
    )


def insert_properties(client, run_id: str, cases: list[dict]):
    rows = [
        {
            "run_id": run_id,
            "case_id": c["case_id"],
            "prop_name": pname,
            "prop_value": pvalue,
            "triggered_at": c["triggered_at"],
        }
        for c in cases
        for pname, pvalue in c["properties"]
    ]
    if rows:
        client.insert(
            "si_run_properties",
            [
                [
                    r["run_id"],
                    r["case_id"],
                    r["prop_name"],
                    r["prop_value"],
                    r["triggered_at"].replace(tzinfo=None),
                ]
                for r in rows
            ],
            column_names=[
                "run_id",
                "case_id",
                "prop_name",
                "prop_value",
                "triggered_at",
            ],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _runner_run_id(args, run_id: str) -> str:
    """This leg's own run id: --gha-run-id when GHA-dispatched, else the same uuid as run_id."""
    raw = (getattr(args, "gha_run_id", "") or "").strip()
    if raw:
        try:
            int(raw)
            return raw
        except (ValueError, TypeError):
            pass
    return run_id


def _threaded_run_id(args) -> str:
    """--run-id when it is a real UUID, else "" so the caller mints one.

    Only a well-formed uuid is honoured: the column is a UUID join key, so any
    other value (a build number, a GHA run id) must be ignored, not stored.
    """
    raw = (getattr(args, "run_id", "") or "").strip()
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# ── SCHEMA v2: test_cases + test_case_runs ─────────────────────────────────
#
# ADDITIVE. Everything above still writes the v1 tables exactly as before; this
# path writes the two v2 tables alongside them and is skipped entirely if they do
# not exist, so the script is safe to deploy before the v2 migration lands.
#
# Both ids are DERIVED, never minted. Four writers (this script, the two sibling
# product ingests, and the orchestrator's pushToClickhouse.pushArtifactResult)
# compute them independently with no threading contract -- which is the only thing
# that makes the tables joinable: v1 minted four unrelated identity schemes and
# artifact_results.run_id consequently joined test_runs.run_id in 2 of 1,266 rows.
#
# BYTE-EXACTNESS IS THE CONTRACT. Disagree about the namespace, the separator, the
# field order or the normalisation and you mint a different uuid for the same row --
# and an orphaned row is indistinguishable from "no tests ran", so the failure is
# silent. The reference implementation, its rule list and the golden values every
# port must reproduce live in spyre-frameworks pipelines/lib/run_identity.py and
# pipelines/lib/test_run_identity.py. Keep this block in sync with it.
# ---------------------------------------------------------------------------

# The product this script ingests for. Replaces v1's hf_/si_ table-name prefixes: one
# v2 table pair serves all three products, discriminated by this column. It is also a
# test_case_id hash input, so it cannot drift from the identity it is stamped on.
V2_COMPONENT = "spyre-inference"

V2_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")
V2_SEP = "|"


def _v2_norm(value) -> str:
    """Canonical scalar form. Lowercasing is not cosmetic: the same tier arrives as
    'Regression' from a Jenkins parameter and 'regression' from a GHA input."""
    return ("" if value is None else str(value)).strip().lower()


def v2_canonical_arch(arch) -> str:
    """amd64/x86/x86-64 all mean x86_64 -- a leg labelled 'amd64' by Jenkins and
    'x86_64' by GHA is ONE leg, and must hash as one."""
    a = _v2_norm(arch)
    return "x86_64" if a in ("amd64", "x86", "x86-64", "x86_64") else a


def v2_run_uid(source: str, external_run_id: str, arch: str, test_type: str) -> str:
    """Identity of one TEST-EXECUTION LEG: (source, external_run_id, arch, test_type).

    arch and test_type are IN the key because the real execution grain measured
    (run, arch, tier) at 19,867 legs under 16,381 CI runs. external_run_id is a
    STRING: typed numerically, every Jenkins leg would be 0 and collide into one id.
    Returns '' when a field is missing -- an all-defaults hash is a real uuid that
    every incomplete leg would share, which is worse than a blank.
    """
    fields = (source, external_run_id, arch, test_type)
    if not all(_v2_norm(f) for f in fields):
        return ""
    return str(
        uuid.uuid5(
            V2_NAMESPACE,
            V2_SEP.join(
                (
                    _v2_norm(source),
                    _v2_norm(external_run_id),
                    v2_canonical_arch(arch),
                    _v2_norm(test_type),
                )
            ),
        )
    )


def v2_test_case_id(component: str, classname: str, name: str, tags) -> str:
    """Content identity of a TEST, so the same test reconciles across runs. v1 minted
    uuid4 per row: 37,322,701 identities for 58,711 distinct (classname, name) pairs.

    `tags` are deduped and SORTED -- they are a set and source order is incidental,
    so an unsorted join makes two writers disagree about the same test. They are
    INSIDE the hash, so re-tagging mints a new identity; trend queries must
    therefore group on (component, classname, name), never on test_case_id.
    """
    norm = sorted({t for t in (_v2_norm(x) for x in (tags or [])) if t})
    return str(
        uuid.uuid5(
            V2_NAMESPACE,
            V2_SEP.join(
                (_v2_norm(component), _v2_norm(classname), _v2_norm(name), ",".join(norm))
            ),
        )
    )


def v2_tags_for_case(case: dict) -> list:
    """The case's tags as an ARRAY of `namespace__value` strings.

    Array, not Map: `testtype` carries up to 5 values on 91.7% of cases, so a Map
    would silently keep one and drop the rest. The v1 shape is a (prop_name,
    prop_value) list where the only prop_name is literally 'tag' and the real
    key is encoded inside the value -- so the VALUE is the tag.
    """
    tags = set()
    for pname, pvalue in case.get("properties", []) or []:
        if pname == "tag":
            if pvalue:
                tags.add(pvalue)
        elif "__" in pname:
            # Some emitters put the namespace__value in the property NAME instead.
            tags.add(pname)
    return sorted(tags)


def v2_source_and_external_run_id(args, run_id: str):
    """(source, external_run_id) for this leg, from whichever CI dispatched it.

    A numeric --gha-run-id means GHA dispatched it. Otherwise the leg is
    Jenkins-dispatched and its own externalizable id ('folder/job#123') is the run
    coordinate -- the SAME value the orchestrator hashes on its side of the join, so
    neither side has to thread a minted uuid.
    `source` is required precisely because a GHA run id and a Jenkins build number
    share a number space.
    """
    gha = (getattr(args, "gha_run_id", "") or "").strip()
    if gha:
        try:
            int(gha)
            return "gha", gha
        except (ValueError, TypeError):
            pass
    jenkins_key = (getattr(args, "jenkins_run_key", "") or "").strip()
    if jenkins_key:
        return "jenkins", jenkins_key
    # No CI coordinate at all: fall back to the run uuid so the rows are still
    # self-consistent and joinable WITHIN this ingest, just not to an artifact.
    return "local", run_id


def v2_tables_present(client) -> bool:
    """v2 write path is skipped unless BOTH tables exist, so this script can be
    deployed before the migration without erroring on every run."""
    return all(
        bool(client.command(f"EXISTS TABLE {t}"))
        for t in ("test_cases", "test_case_runs")
    )


def v2_already_ingested(client, run_uid: str, component: str) -> bool:
    """test_case_runs is a plain MergeTree with no dedup key, so a double ingest of one
    leg DOUBLES its counts -- and the v2 schema dropped the stored counters precisely
    because they are derived from these rows. This check is what keeps that correct.
    Scoped by component as well as run_uid to hit the ORDER BY prefix."""
    rows = client.query(
        "SELECT count() FROM test_case_runs "
        "WHERE component = {component:String} AND run_uid = {run_uid:UUID}",
        parameters={"component": component, "run_uid": run_uid},
    ).result_rows
    return bool(rows and rows[0][0] > 0)


def insert_v2(client, component: str, run_uid: str, cases: list) -> int:
    """Write test_cases (identity) + test_case_runs (outcome) for one leg.

    Dropped from v2 deliberately: filename, suite_name, runner_run_id, and every
    stored counter -- all derivable, and a stored counter invites drift.
    """
    if not cases:
        return 0
    ident_rows, run_rows = {}, []
    for c in cases:
        tags = v2_tags_for_case(c)
        classname, name = c.get("classname", ""), c.get("name", "")
        tcid = v2_test_case_id(component, classname, name, tags)
        # Deduped by id within the leg: identical identity rows are one fact, and
        # test_cases is a plain MergeTree (a content hash re-writes an identical row,
        # so collapsing would only be cosmetic -- but writing it N times is not).
        ident_rows[tcid] = [tcid, component, classname, name, tags]
        run_rows.append(
            [
                run_uid,
                tcid,
                component,
                c.get("status", ""),
                float(c.get("duration_s", 0) or 0),
                (c.get("fail_message") or "")[:8192],
            ]
        )
    client.insert(
        "test_cases",
        list(ident_rows.values()),
        column_names=["test_case_id", "component", "classname", "name", "tags"],
    )
    client.insert(
        "test_case_runs",
        run_rows,
        column_names=[
            "run_uid",
            "test_case_id",
            "component",
            "status",
            "duration_s",
            "fail_message",
        ],
    )
    return len(run_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=None)
    parser.add_argument("--xml-file", default=None)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--gha-run-id", default="")
    parser.add_argument("--triggered-at", default="")
    parser.add_argument("--pr-number", default="")
    parser.add_argument(
        "--jenkins-run-key",
        default="",
        help="This leg's own Jenkins externalizable id, e.g. 'Spyre/component-build#417'. "
        "Hashed into the schema-v2 run_uid, which is how the orchestrator's "
        "artifact_results row and these per-case rows join without threading a uuid.",
    )
    parser.add_argument(
        "--trigger-type",
        default="",
        help="Suite tier, e.g. smoke | core | full | trunk | weekly | nightly",
    )
    parser.add_argument(
        "--platform",
        default="",
        help="Hardware platform the suite ran on, e.g. x86_64 | s390x | ppc64le",
    )
    parser.add_argument(
        "--img-digest",
        default="",
        help="Digest of the runner image the suite ran against, if known",
    )
    args = parser.parse_args()

    if args.xml_file:
        xml_root = Path(args.xml_file).parent
        xml_files = [Path(args.xml_file)]
    elif args.xml_dir:
        xml_root = Path(args.xml_dir)
        xml_files = sorted(xml_root.rglob("*.xml"))
    else:
        print("Error: provide --xml-dir or --xml-file")
        sys.exit(1)

    if not xml_files:
        print("No XML files found — nothing to ingest.")
        sys.exit(0)

    print(
        f"Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("Connected.\n")

    db = os.environ.get("CLICKHOUSE_DB", "spyre")
    if not tables_exist(client, db):
        print(f"{db}.si_test_runs does not exist — nothing to ingest into. Silent no-op.")
        sys.exit(0)

    total_cases = 0

    for xml_path in xml_files:
        print(f"Processing: {xml_path.name}")

        run, cases = parse_test_xml(xml_path)
        if run is None:
            continue

        # Different suites can independently produce a JUnit XML with the
        # same basename (e.g. GitHub Actions strips the model-key
        # subdirectory when an artifact is a single file), so the path
        # relative to the XML root -- not the bare basename -- is what makes
        # `filename` actually unique for both dedup and storage.
        run["filename"] = str(xml_path.relative_to(xml_root))

        # One run_id per TEST RUN, not per XML file: the dispatching orchestrator generates
        # a uuid and threads it down as --run-id, stamping the SAME value on
        # artifact_results, so the two tables join. `filename` stays the per-file
        # discriminator among the rows that share it. Falls back to a fresh uuid4 when
        # --run-id is absent or not a uuid (standalone / GHA-only run): the rows are still
        # valid, just not linked to an artifact.
        run_id = _threaded_run_id(args) or str(uuid.uuid4())

        # Dedup on (run_id, filename): a re-ingest of the SAME test run must be idempotent,
        # but two distinct runs must never collapse. runner_run_id mirrors run_id for a
        # Jenkins/standalone leg, so it's only an independent signal for a GHA numeric id.
        runner_run_id = _runner_run_id(args, run_id)
        existing = client.query(
            "SELECT count() FROM si_test_runs "
            "WHERE run_id = {run_id:String} AND filename = {filename:String}",
            parameters={"run_id": run_id, "filename": run["filename"]},
        )
        if existing.result_rows[0][0] == 0 and runner_run_id and runner_run_id != run_id:
            # A GHA re-ingest mints a fresh uuid4, so fall back to the numeric
            # run id to keep that path idempotent.
            existing = client.query(
                "SELECT count() FROM si_test_runs WHERE "
                "runner_run_id = {runner_run_id:String} AND filename = {filename:String}",
                parameters={"runner_run_id": runner_run_id, "filename": run["filename"]},
            )
        if existing.result_rows[0][0] > 0:
            print(f"  Already ingested — skipping {run['filename']}")
            continue

        print(
            f"  run_id={run_id}  tests={run['total_tests']}  "
            f"passed={run['passed']}  failed={run['failed']}  "
            f"xpass={run['xpass']}  xfail={run['xfail']}  skipped={run['skipped']}"
        )

        insert_run(client, run_id, run, args)
        insert_cases(client, run_id, cases, workflow=args.workflow)
        insert_properties(client, run_id, cases)
        # v2 tables, alongside v1. Guarded so this script still runs against a
        # database where the migration has not landed.
        if v2_tables_present(client):
            _v2_source, _v2_ext = v2_source_and_external_run_id(args, run_id)
            _v2_tier = (getattr(args, "trigger_type", "") or "").strip()
            _v2_arch = (args.platform or run.get("platform") or "").strip()
            _v2_run_uid = v2_run_uid(_v2_source, _v2_ext, _v2_arch, _v2_tier)
            if not _v2_run_uid:
                # Loud, because a blank run_uid means these cases reach v2 unjoinable
                # to any artifact -- and that reads downstream as "no tests ran".
                print(
                    f"  [warn] v2 skipped: run_uid not derivable "
                    f"(source={_v2_source} ext={_v2_ext!r} arch={_v2_arch!r} "
                    f"tier={_v2_tier!r}); --trigger-type is the field usually missing",
                    file=sys.stderr,
                )
            elif v2_already_ingested(client, _v2_run_uid, V2_COMPONENT):
                print(f"  v2: already ingested run_uid={_v2_run_uid} — skipping")
            else:
                _n = insert_v2(client, V2_COMPONENT, _v2_run_uid, cases)
                print(f"  v2: {_n} test_case_runs under run_uid={_v2_run_uid}")


        total_cases += len(cases)
        print(
            f"  Inserted {len(cases)} test cases + "
            f"{sum(len(c['properties']) for c in cases)} properties"
        )

    print(f"\nDone. {len(xml_files)} file(s) processed.")
    print(f"  Test cases ingested:  {total_cases}")


if __name__ == "__main__":
    main()
