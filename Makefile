SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# TEST_TYPE selects which subset of tests to run (uniform knob across the
# product repos: torch-spyre, hf-adapters, spyre-inference):
#   smoke — fast per-op unit tests only
#   core  — all spyre-native tests (per-op + attention + distributed);
#           excludes the heavy upstream-vLLM suites
#   full  — everything (default)
# Empty / unset defaults to "full".
TEST_TYPE ?= full

# Flags passed verbatim to pytest. Mirrors the CI invocation so `make test`
# reproduces CI verbosity; override e.g. `make test PYTEST_ARGS="-x -q"`.
PYTEST_ARGS ?= -s -vvv

# When set, write JUnit XML here (both CI callers -- GHA's _test_matrix.yaml
# and spyre-frameworks' Jenkinsfile.product-test -- set this to collect
# results for artifact upload / ClickHouse ingestion). Unset = no JUnit file.
JUNIT_XML ?=
ifneq ($(JUNIT_XML),)
JUNIT_ARGS := --junitxml=$(JUNIT_XML)
else
JUNIT_ARGS :=
endif

# Map TEST_TYPE to a pytest -m marker expression. full -> no filter (all tests).
# MARK_OVERRIDE bypasses TEST_TYPE entirely for callers that need a marker
# expression finer than the 3 coarse tiers (e.g. CI splitting the "full"-only
# upstream suites into separate parallel jobs) -- set MARK_OVERRIDE and the
# TEST_TYPE mapping below is skipped.
ifneq ($(MARK_OVERRIDE),)
MARK_EXPR := -m "$(MARK_OVERRIDE)"
else ifeq ($(TEST_TYPE),full)
MARK_EXPR :=
else ifeq ($(TEST_TYPE),smoke)
MARK_EXPR := -m "not (distributed or upstream or attention)"
else ifeq ($(TEST_TYPE),core)
MARK_EXPR := -m "not upstream"
else
$(error Invalid TEST_TYPE '$(TEST_TYPE)'. Valid values: smoke | core | full)
endif

.PHONY: help test tests

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: TEST_TYPE=smoke|core|full (default full), MARK_OVERRIDE (raw -m expr, bypasses TEST_TYPE), PYTEST_ARGS (default '$(PYTEST_ARGS)'), JUNIT_XML (path to write JUnit results, unset = no JUnit file)"

test: ## Run tests. Narrow scope with TEST_TYPE=smoke|core|full (default full).
	# Port of the CI "Run tests" env setup: ibm-aiu-setup.sh ends with a chmod of
	# root-owned /tmp/etc that fails on the Spyre image; env vars are already
	# exported by then, so tolerate that failure. rm -f so local re-runs don't
	# fail when topo.json is absent.
	unset _IBM_AIU_SETUP
	rm -f /tmp/etc/ibm/spyre/topo.json
	set +e
	source "$$HOME/.bashrc"
	source /etc/profile.d/ibm-aiu-setup.sh
	set -e
	echo "Running tests for TEST_TYPE=$(TEST_TYPE)..."
	uv run pytest $(PYTEST_ARGS) $(MARK_EXPR) $(JUNIT_ARGS)

tests: test  ## Alias for `test`.
