VENV ?= .venv
PYTHON_VERSION ?= 3.12.12
UV ?= uv
PY := $(VENV)/bin/python

SRC = src/
TESTS = tests/
ALL_CODE = ${SRC} ${TESTS}

TEST_ARGS ?=
KAGGLE_RANDOM_RUN_NAME ?= taaf-random-kaggle
KAGGLE_RANDOM_JOB_DIR ?=
KAGGLE_RANDOM_KERNEL_SLUG ?= $(KAGGLE_RANDOM_RUN_NAME)
KAGGLE_RANDOM_KERNEL_TITLE ?=
KAGGLE_RANDOM_DATASET_REF ?=
KAGGLE_RANDOM_MAX_RUNTIME_MINUTES ?= 10
KAGGLE_RANDOM_GAME ?=
KAGGLE_RANDOM_GAME_LIMIT ?= 25
KAGGLE_RANDOM_MAX_ACTIONS ?=
KAGGLE_RANDOM_SEED ?= 0
KAGGLE_RANDOM_WAIT ?= false
KAGGLE_RANDOM_DRY_RUN ?= false
KAGGLE_RANDOM_CPU_ONLY ?= true

$(PY):
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)

install: $(PY)
	$(UV) sync --python $(PY) --locked --no-dev

install-dev: $(PY)
	$(UV) sync --python $(PY) --locked
	$(UV) run nbstripout --install
	$(UV) run python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (tufa-taaf)"

test: install-dev
	$(PY) -m pytest $(TEST_ARGS)

test-all: install-dev
	$(PY) -m pytest -m "" $(TEST_ARGS)

format: install-dev
	$(UV) run ruff check ${ALL_CODE} --select I --fix
	$(UV) run ruff format ${ALL_CODE}
	$(UV) run nbstripout notebooks/*.ipynb

check-format: install-dev
	$(UV) run ruff check ${ALL_CODE} --select I
	$(UV) run ruff format --check ${ALL_CODE}

lint: install-dev
	$(UV) run ruff check ${ALL_CODE}

# grep the whole output, not tail -n1: pyright prints an upgrade nag after
# its summary line whenever a newer release exists on npm, which would
# otherwise become the last line and fail the check on every new pyright.
static: install-dev
	@output=$$( $(UV) run pyright ${SRC} 2>&1 ) ; rc=$$? ; \
		printf '%s\n' "$$output" ; \
		if [ $$rc -ne 0 ]; then exit $$rc ; fi ; \
		if ! printf '%s\n' "$$output" | grep -qE '^0 errors, 0 warnings, 0 informations\b'; then \
			echo "ERROR: pyright reported warnings; R1.4 requires zero errors AND warnings" >&2 ; \
			exit 1 ; \
		fi

prepare: format lint static test

prepare-ci: check-format lint static test-all

refresh-examples: install-dev
	$(PY) scripts/refresh_examples.py

kaggle-random: $(PY)
	PATH="$(abspath $(VENV))/bin:$$PATH" $(PY) -m taaf.kaggle_random \
		--run-name "$(KAGGLE_RANDOM_RUN_NAME)" \
		$(if $(strip $(KAGGLE_RANDOM_JOB_DIR)),--job-dir "$(KAGGLE_RANDOM_JOB_DIR)",) \
		$(if $(strip $(KAGGLE_RANDOM_KERNEL_SLUG)),--kernel-slug "$(KAGGLE_RANDOM_KERNEL_SLUG)",) \
		$(if $(strip $(KAGGLE_RANDOM_KERNEL_TITLE)),--kernel-title "$(KAGGLE_RANDOM_KERNEL_TITLE)",) \
		$(if $(strip $(KAGGLE_RANDOM_DATASET_REF)),--dataset-ref "$(KAGGLE_RANDOM_DATASET_REF)",) \
		$(if $(strip $(KAGGLE_RANDOM_GAME)),--game "$(KAGGLE_RANDOM_GAME)",) \
		$(if $(strip $(KAGGLE_RANDOM_GAME_LIMIT)),--game-limit "$(KAGGLE_RANDOM_GAME_LIMIT)",) \
		--max-runtime-minutes "$(KAGGLE_RANDOM_MAX_RUNTIME_MINUTES)" \
		$(if $(strip $(KAGGLE_RANDOM_MAX_ACTIONS)),--max-actions-per-game "$(KAGGLE_RANDOM_MAX_ACTIONS)",) \
		--seed "$(KAGGLE_RANDOM_SEED)" \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_RANDOM_WAIT)),--wait,--no-wait) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_RANDOM_DRY_RUN)),--dry-run,) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_RANDOM_CPU_ONLY)),--cpu-only,--no-cpu-only)

# Serve examples/ over HTTP so Chrome can load the HTMLs (and play the MP4s,
# which Chrome blocks under file://). Port 8765 is auto-forwarded by VSCode
# over SSH, so opening http://localhost:8765/ on your local machine just works.
EXAMPLES_PORT ?= 8765
serve-examples:
	@echo "Open http://localhost:$(EXAMPLES_PORT)/comparison/ in your browser"
	$(PY) -m http.server $(EXAMPLES_PORT) -d examples/diagnostics

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache

.PHONY: install install-dev test test-all format check-format lint static prepare prepare-ci refresh-examples kaggle-random serve-examples clean
