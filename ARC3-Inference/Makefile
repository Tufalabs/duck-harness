.DEFAULT_GOAL := interactive

.PHONY: install install-local-sources prepare-ci server stop-server check-server interactive sbatch kaggle-duck run _taaf-run chat eval score_run significance view trace traces zip clean cluster_view_queue

-include .env

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLUSTER_USER ?= $(if $(strip $(USER)),$(USER),$(shell whoami))
SHARED_STORAGE_ROOT ?= /shared/$(CLUSTER_USER)
RUNS_DIR ?= $(REPO_ROOT)/runs
RUNS_ZIP ?= $(REPO_ROOT)/runs.zip

UV ?= uv
UV_RUN ?= $(UV) run --no-sync
PYTHON ?= python3
CONFIG_PATH_ORIGIN := $(origin CONFIG_PATH)
CONFIG_PATH ?= configs/inference.json
CONFIG_VALUE = $(PYTHON) inference/tools/config_value.py "$(CONFIG_PATH)"
SBATCH_CONFIG_PATH ?= configs/inference.json
ifeq ($(CONFIG_PATH_ORIGIN),undefined)
SBATCH_EFFECTIVE_CONFIG_PATH := $(SBATCH_CONFIG_PATH)
else
SBATCH_EFFECTIVE_CONFIG_PATH := $(CONFIG_PATH)
endif

# Keep Hugging Face downloads off home-directory storage on cluster nodes.
HF_HOME ?= $(if $(wildcard $(SHARED_STORAGE_ROOT)),$(SHARED_STORAGE_ROOT),$(HOME)/.cache/huggingface)
HF_HUB_CACHE ?= $(HF_HOME)/hub
TRANSFORMERS_CACHE ?= $(HF_HUB_CACHE)
HF_DATASETS_CACHE ?= $(HF_HOME)/datasets
HF_ASSETS_CACHE ?= $(HF_HOME)/assets
HF_XET_CACHE ?= $(HF_HOME)/xet
HUGGINGFACE_HUB_CACHE ?= $(HF_HUB_CACHE)
HUGGINGFACE_ASSETS_CACHE ?= $(HF_ASSETS_CACHE)
SERVER_STATE_ROOT ?= $(if $(wildcard $(SHARED_STORAGE_ROOT)),$(SHARED_STORAGE_ROOT)/arc3_runtime,$(REPO_ROOT)/.cache/arc3_runtime)
SERVER_CACHE_ROOT ?= $(SERVER_STATE_ROOT)/cache
SERVER_TMP_DIR ?= /tmp/arc3_$(CLUSTER_USER)
TORCHINDUCTOR_CACHE_DIR ?= $(SERVER_CACHE_ROOT)/torchinductor
TORCHINDUCTOR_FX_GRAPH_CACHE_DIR ?= $(SERVER_CACHE_ROOT)/torchinductor_fx_graph
TRITON_CACHE_DIR ?= $(SERVER_CACHE_ROOT)/triton
XDG_CACHE_HOME ?= $(SERVER_CACHE_ROOT)/xdg

GAME ?= $(shell $(CONFIG_VALUE) environment.games ls20-cb3b57cc)
DATASET ?= $(shell $(CONFIG_VALUE) environment.dataset)
GAME_TAGS ?= $(shell $(CONFIG_VALUE) environment.include_tags)
EXCLUDE_GAME_TAGS ?= $(shell $(CONFIG_VALUE) environment.exclude_tags)
ENVIRONMENTS_DIR ?= $(shell $(CONFIG_VALUE) environment.environments_dir)
MAX_ACTIONS ?= $(shell $(CONFIG_VALUE) environment.max_steps)
MAX_RUNTIME_MINUTES ?= $(shell $(CONFIG_VALUE) environment.max_runtime_minutes)
MAX_EXPERIMENT_RUNTIME_HOURS ?= $(shell $(CONFIG_VALUE) environment.max_experiment_runtime_hours)
MAX_EXPERIMENT_RUNTIME_MINUTES ?= $(shell $(CONFIG_VALUE) environment.max_experiment_runtime_minutes)
MAX_EXPERIMENT_RUNTIME_ARGS = $(if $(strip $(MAX_EXPERIMENT_RUNTIME_MINUTES)),--max-experiment-runtime-minutes $(MAX_EXPERIMENT_RUNTIME_MINUTES),$(if $(strip $(MAX_EXPERIMENT_RUNTIME_HOURS)),--max-experiment-runtime-hours $(MAX_EXPERIMENT_RUNTIME_HOURS),))
EXPERIMENTS_DIR ?= $(shell $(CONFIG_VALUE) experiments.root_dir)
EXPERIMENT_DIR ?=
N_PASSES ?= $(shell $(CONFIG_VALUE) environment.n_passes 1)
CONCURRENT_JOBS ?= $(shell $(CONFIG_VALUE) environment.concurrent_jobs 1)
SIMULATE_COMPETITION_ARCADE ?= false
COMPETITION_CLONE_RUNS ?=
MODEL ?= $(shell $(CONFIG_VALUE) analyzer.model_id $(SHARED_MODEL_NAME))
AGENT ?= inference
RUN_NAME ?=

DEPLOYMENT_TARGET ?= $(shell $(CONFIG_VALUE) deployment.target inline)
DEPLOYMENT_WAIT ?= $(shell $(CONFIG_VALUE) deployment.wait false)
DEPLOYMENT_SOURCE_REPOS ?= $(shell $(CONFIG_VALUE) deployment.source_repos [])
DEPLOYMENT_SLURM_GPU ?= $(shell $(CONFIG_VALUE) deployment.slurm.gpu B200)
DEPLOYMENT_SLURM_GPU_COUNT ?= $(shell $(CONFIG_VALUE) deployment.slurm.gpu_count 1)
DEPLOYMENT_SLURM_TIME ?= $(shell $(CONFIG_VALUE) deployment.slurm.time 06:00:00)
DEPLOYMENT_SLURM_IMAGE ?= $(shell $(CONFIG_VALUE) deployment.slurm.image)
DEPLOYMENT_SLURM_PARTITION ?= $(shell $(CONFIG_VALUE) deployment.slurm.partition)
DEPLOYMENT_SLURM_NODELIST ?= $(shell $(CONFIG_VALUE) deployment.slurm.nodelist)
DEPLOYMENT_SLURM_EXTRA_SBATCH_FLAGS ?= $(shell $(CONFIG_VALUE) deployment.slurm.extra_sbatch_flags [])
KAGGLE_DUCK_PUBLIC_HARNESS ?= false
KAGGLE_MAKE_SHARE_VERSION ?= false
KAGGLE_USERNAME ?=
KAGGLE_KERNEL_SLUG ?= $(if $(strip $(RUN_NAME)),taaf-$(RUN_NAME)$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_MAKE_SHARE_VERSION)),-share),)
KAGGLE_KERNEL_TITLE ?=
KAGGLE_DATASET_REF ?=
KAGGLE_ADDITIONAL_DATASET_SOURCES ?=
KAGGLE_PUBLIC ?= false
KAGGLE_ENABLE_INTERNET ?= false
KAGGLE_CPU_ONLY ?= false
KAGGLE_ACCELERATOR ?= NvidiaRtxPro6000
KAGGLE_RUN_AS_SUBMISSION ?= false
KAGGLE_DRY_RUN ?= false
KAGGLE_DATASET_VERSION_MESSAGE ?=

SERVER_BACKEND ?= $(shell $(CONFIG_VALUE) server.backend vllm)
SERVER_HOST ?= $(shell $(CONFIG_VALUE) server.host 127.0.0.1)
SERVER_PORT_ORIGIN := $(origin SERVER_PORT)
SERVER_PORT ?= $(shell $(CONFIG_VALUE) server.port 1234)
ARC3_SERVER_PORT_EXPLICIT ?= $(if $(filter environment command line file override,$(SERVER_PORT_ORIGIN)),true,false)
SERVER_START_TIMEOUT ?= 900
SERVER_STATUS_EVERY ?= 10
SERVER_LOG_TAIL_LINES ?= 80
SERVER_TAIL_ON_WAIT ?= false
SERVER_HEALTH_URL ?= http://127.0.0.1:$(SERVER_PORT)/v1/models

SHARED_MODEL_NAME ?= $(shell $(CONFIG_VALUE) shared.model_name)
SHARED_BASE_URL ?= $(shell $(CONFIG_VALUE) shared.base_url http://127.0.0.1:$(SERVER_PORT)/v1)
SHARED_PROVIDER ?= $(shell $(CONFIG_VALUE) shared.provider vllm)
SHARED_CONTEXT_WINDOW ?= $(shell $(CONFIG_VALUE) shared.context_window 65536)

SERVER_SHARED_MODEL_NAME ?= $(shell $(CONFIG_VALUE) shared.model_name)
SERVER_SHARED_CONTEXT_WINDOW ?= $(shell $(CONFIG_VALUE) shared.context_window 65536)
SERVER_MODEL ?= $(shell $(CONFIG_VALUE) server.model $(SERVER_SHARED_MODEL_NAME))
SERVER_SERVED_MODEL_NAME ?= $(shell $(CONFIG_VALUE) server.served_model_name $(SERVER_SHARED_MODEL_NAME))
SERVER_TENSOR_PARALLEL_SIZE ?= $(shell $(CONFIG_VALUE) server.tensor_parallel_size 1)
SERVER_MAX_MODEL_LEN ?= $(shell $(CONFIG_VALUE) server.max_model_len $(SERVER_SHARED_CONTEXT_WINDOW))
SERVER_DTYPE ?= $(shell $(CONFIG_VALUE) server.dtype)
SERVER_GPU_MEMORY_UTILIZATION ?= $(shell $(CONFIG_VALUE) server.gpu_memory_utilization)
SERVER_TRUST_REMOTE_CODE ?= $(shell $(CONFIG_VALUE) server.trust_remote_code false)
SERVER_GENERATION_CONFIG ?= $(shell $(CONFIG_VALUE) server.generation_config vllm)
SERVER_LANGUAGE_MODEL_ONLY ?= $(shell $(CONFIG_VALUE) server.language_model_only false)
SERVER_ENABLE_PREFIX_CACHING ?= $(shell $(CONFIG_VALUE) server.enable_prefix_caching true)
SERVER_DEFAULT_CHAT_TEMPLATE_KWARGS ?= $(shell $(CONFIG_VALUE) server.default_chat_template_kwargs)
SERVER_TOOL_CALL_PARSER ?= $(shell $(CONFIG_VALUE) server.tool_call_parser qwen3_coder)
SERVER_REASONING_PARSER ?= $(shell $(CONFIG_VALUE) server.reasoning_parser)
SERVER_ENABLE_LORA ?= false
SERVER_MAX_LORAS ?= 1
SERVER_MAX_LORA_RANK ?= 16
SERVER_LORA_DTYPE ?= auto
SERVER_LORA_ALIAS ?= policy
SERVER_LORA_PATH ?=
SERVER_ALLOW_RUNTIME_LORA_UPDATING ?= $(if $(filter true TRUE 1 yes YES on ON,$(SERVER_ENABLE_LORA)),true,false)
SERVER_LOG ?= $(SERVER_STATE_ROOT)/arc3-inference-server.log
SERVER_PID ?= $(SERVER_STATE_ROOT)/arc3-inference-server.pid
SERVER_REQUIRE_API_KEY ?= true
SERVER_API_KEY ?=
SERVER_API_KEY_FILE ?= $(SERVER_STATE_ROOT)/server-api-key
SERVER_VENV_PYTHON ?= $(REPO_ROOT)/.venv/bin/python
SERVER_VENV_VLLM ?= $(dir $(SERVER_VENV_PYTHON))vllm
SERVER_VENV_BIN ?= $(patsubst %/,%,$(dir $(SERVER_VENV_PYTHON)))
SERVER_LOAD_LORA_URL ?= http://127.0.0.1:$(SERVER_PORT)/v1/load_lora_adapter

LOCAL_ANALYZER_BASE_URL ?= $(shell $(CONFIG_VALUE) analyzer.base_url $(SHARED_BASE_URL))
LOCAL_ANALYZER_PROVIDER ?= $(shell $(CONFIG_VALUE) analyzer.provider $(SHARED_PROVIDER))
LOCAL_ANALYZER_MODEL_ID ?= $(shell $(CONFIG_VALUE) analyzer.model_id $(SERVER_SERVED_MODEL_NAME))
LOCAL_ANALYZER_CONTEXT_WINDOW ?= $(shell $(CONFIG_VALUE) analyzer.context_window $(SHARED_CONTEXT_WINDOW))
LOCAL_ANALYZER_MAX_OUTPUT ?= $(shell $(CONFIG_VALUE) analyzer.max_output 0)
LOCAL_ANALYZER_TIMEOUT ?= $(shell $(CONFIG_VALUE) analyzer.timeout 120)
LOCAL_ANALYZER_TEMPERATURE ?= $(shell $(CONFIG_VALUE) analyzer.temperature 0.6)
LOCAL_ANALYZER_TOP_P ?= $(shell $(CONFIG_VALUE) analyzer.top_p 0.95)
LOCAL_ANALYZER_TOP_K ?= $(shell $(CONFIG_VALUE) analyzer.top_k 20)
LOCAL_ANALYZER_TOOL_STEPS ?= $(shell $(CONFIG_VALUE) analyzer.tool_steps 0)
LOCAL_ANALYZER_TOOL_TIMEOUT ?= $(shell $(CONFIG_VALUE) analyzer.tool_timeout 30)
LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS ?= $(shell $(CONFIG_VALUE) analyzer.tool_output_tokens 1024)
LOCAL_ANALYZER_YIELD_SECONDS ?= $(shell $(CONFIG_VALUE) analyzer.yield_seconds 60)
LOCAL_ANALYZER_ENABLE_THINKING ?= $(shell $(CONFIG_VALUE) analyzer.thinking true)
LOCAL_ANALYZER_SITE_URL ?= $(shell $(CONFIG_VALUE) analyzer.site_url)
LOCAL_ANALYZER_APP_NAME ?= $(shell $(CONFIG_VALUE) analyzer.app_name ARC3 Agent Harness)
ANALYZER_TIMEOUT ?= $(shell $(CONFIG_VALUE) analyzer.timeout 120)
ANALYZER_SAVE_REQUEST_LOGS ?= $(shell $(CONFIG_VALUE) analyzer.save_request_logs false)
DEPLOYMENT_SLURM_START_LOCAL_SERVER ?= $(shell $(CONFIG_VALUE) deployment.slurm.start_local_server $(if $(filter vllm,$(LOCAL_ANALYZER_PROVIDER)),true,false))
DEPLOYMENT_SLURM_LOCAL_SERVER_CONFIG ?= $(CONFIG_PATH)

MULTIMODAL_CONTEXT ?= $(shell $(CONFIG_VALUE) multimodal.context)
MULTIMODAL_UPSCALE ?= $(shell $(CONFIG_VALUE) multimodal.upscale 16)

CHAT_BASE_URL ?= $(shell $(CONFIG_VALUE) chat.base_url $(LOCAL_ANALYZER_BASE_URL))
CHAT_PROVIDER ?= $(shell $(CONFIG_VALUE) chat.provider $(LOCAL_ANALYZER_PROVIDER))
CHAT_MODEL ?= auto
CHAT_MAX_TOKENS ?= $(shell $(CONFIG_VALUE) chat.max_tokens 2048)
CHAT_TEMPERATURE ?= $(shell $(CONFIG_VALUE) chat.temperature 0.7)
CHAT_TOP_P ?= $(shell $(CONFIG_VALUE) chat.top_p 0.8)
CHAT_TOP_K ?= $(shell $(CONFIG_VALUE) chat.top_k 20)
CHAT_THINKING ?= $(shell $(CONFIG_VALUE) chat.thinking true)
CHAT_TIMEOUT ?= 600
CHAT_SYSTEM ?=
PROMPT ?=

EVAL_CONFIG_PATH ?= configs/eval.json
SCORE_OUTPUT_PATH ?=
SCORE_RUN_DIR ?=
SIGNIFICANCE_CONFIG_PATH ?= configs/significance.json
BASELINE_SCORE ?=
CANDIDATE_SCORE ?=
ALLOW_INTERSECTION ?=
CONFIDENCE_THRESHOLD ?=
BOOTSTRAP_SAMPLES ?=
BOOTSTRAP_SEED ?=

VIEW_HOST ?= 127.0.0.1
VIEW_PORT ?= $(shell $(CONFIG_VALUE) viewer.port 8011)
VIEW_RUNS_DIR ?= $(RUNS_DIR)
# Default to the bundled example-run at the repo root (this exposition ships one),
# so a bare `make view` opens it. Falls back to VIEW_RUNS_DIR when absent.
VIEW_RUN_DIR ?= $(wildcard $(REPO_ROOT)/../example-run)

export LOCAL_ANALYZER_BASE_URL
export LOCAL_ANALYZER_PROVIDER
export LOCAL_ANALYZER_MODEL_ID
export LOCAL_ANALYZER_CONTEXT_WINDOW
export LOCAL_ANALYZER_MAX_OUTPUT
export LOCAL_ANALYZER_TIMEOUT
export LOCAL_ANALYZER_TEMPERATURE
export LOCAL_ANALYZER_TOP_P
export LOCAL_ANALYZER_TOP_K
export LOCAL_ANALYZER_TOOL_STEPS
export LOCAL_ANALYZER_TOOL_TIMEOUT
export LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS
export LOCAL_ANALYZER_YIELD_SECONDS
export LOCAL_ANALYZER_ENABLE_THINKING
export LOCAL_ANALYZER_SITE_URL
export LOCAL_ANALYZER_APP_NAME
export MULTIMODAL_CONTEXT
export MULTIMODAL_UPSCALE
export CHAT_PROVIDER
export OPENROUTER_API_KEY
export OPENAI_API_KEY
export LOCAL_ANALYZER_API_KEY
export CHAT_API_KEY
export SERVER_API_KEY
export SERVER_PORT
export ARC3_SERVER_PORT_EXPLICIT
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export HF_DATASETS_CACHE
export HF_ASSETS_CACHE
export HF_XET_CACHE
export HUGGINGFACE_HUB_CACHE
export HUGGINGFACE_ASSETS_CACHE
export SERVER_STATE_ROOT
export SERVER_CACHE_ROOT
export SERVER_TMP_DIR
export TORCHINDUCTOR_CACHE_DIR
export TORCHINDUCTOR_FX_GRAPH_CACHE_DIR
export TRITON_CACHE_DIR
export XDG_CACHE_HOME

install:
	@if [ ! -x "$(REPO_ROOT)/.venv/bin/python" ]; then \
		$(UV) venv --python 3.12.12 $(REPO_ROOT)/.venv; \
	fi
	env -u VIRTUAL_ENV $(UV) sync --locked --all-extras --python $(REPO_ROOT)/.venv/bin/python
	@$(MAKE) install-local-sources
	@echo "Project, test, and inference server dependencies are installed in $(REPO_ROOT)/.venv."

install-local-sources:
	@if [ ! -x "$(REPO_ROOT)/.venv/bin/python" ]; then \
		echo "Project virtualenv is missing. Run 'make install' first."; \
		exit 1; \
	fi
	@if [ -d "$(REPO_ROOT)/../tufa-arc-agi-framework" ]; then \
		$(UV) pip install --python "$(REPO_ROOT)/.venv/bin/python" \
			--no-deps \
			-e "$(REPO_ROOT)/../tufa-arc-agi-framework"; \
	fi

prepare-ci:
	$(UV) run --locked --extra dev ruff check .
	LOCAL_ANALYZER_PROVIDER=vllm $(UV) run --locked --extra dev pytest

server:
	@if [ "$(SERVER_BACKEND)" != "vllm" ]; then \
		echo "Unsupported server backend '$(SERVER_BACKEND)'. This branch only supports vLLM."; \
		exit 1; \
	fi
	@if [ -z "$(strip $(SERVER_MODEL))" ]; then \
		echo "Server model is required. Set server.model or shared.model_name in $(CONFIG_PATH)."; \
		exit 1; \
	fi
	@if [ -z "$(strip $(SERVER_SERVED_MODEL_NAME))" ]; then \
		echo "Served model name is required. Set server.served_model_name or shared.model_name in $(CONFIG_PATH)."; \
		exit 1; \
	fi
	@if [ ! -x "$(SERVER_VENV_VLLM)" ]; then \
		echo "Server dependencies are missing; running 'make install'."; \
		$(MAKE) install; \
	fi
	@server_api_key=""; \
	if [ "$(SERVER_REQUIRE_API_KEY)" != "false" ] && [ "$(SERVER_REQUIRE_API_KEY)" != "FALSE" ] && [ "$(SERVER_REQUIRE_API_KEY)" != "0" ]; then \
		server_api_key="$$(PYTHON="$(PYTHON)" "$(REPO_ROOT)/scripts/ensure_server_api_key.sh" "$(SERVER_API_KEY_FILE)")"; \
		echo "Server API key file: $(SERVER_API_KEY_FILE)"; \
	fi; \
	server_http_status="$$("$(REPO_ROOT)/scripts/server_curl.sh" "$(SERVER_API_KEY_FILE)" -sS -o /dev/null -w "%{http_code}" "$(SERVER_HEALTH_URL)" 2>/dev/null || true)"; \
	if [ "$$server_http_status" = "200" ]; then server_ready=1; else server_ready=0; fi; \
	if [ "$$server_http_status" = "401" ] || [ "$$server_http_status" = "403" ]; then \
		echo "Inference server at $(SERVER_HEALTH_URL) rejected this run's API key."; \
		echo "Refusing to share an existing vLLM server. Choose a unique SERVER_PORT or stop the existing server."; \
		exit 1; \
	fi; \
	if [ "$$server_ready" = "1" ]; then \
		if [ -f "$(SERVER_PID)" ] && kill -0 "$$(cat "$(SERVER_PID)")" >/dev/null 2>&1; then \
			echo "Inference server already running at $(SERVER_HEALTH_URL)"; \
		else \
			echo "Inference server is already reachable at $(SERVER_HEALTH_URL), but it is not owned by this run."; \
			echo "Refusing to share an existing vLLM server. Choose a unique SERVER_PORT or stop the existing server."; \
			exit 1; \
		fi; \
	else \
		mkdir -p "$(SERVER_STATE_ROOT)" "$(SERVER_CACHE_ROOT)" "$(SERVER_TMP_DIR)" "$(TORCHINDUCTOR_CACHE_DIR)" "$(TORCHINDUCTOR_FX_GRAPH_CACHE_DIR)" "$(TRITON_CACHE_DIR)" "$(XDG_CACHE_HOME)"; \
		rm -f "$(SERVER_PID)"; \
		if [ -f "$(SERVER_LOG)" ]; then mv "$(SERVER_LOG)" "$(SERVER_LOG).prev" 2>/dev/null || true; fi; \
		: >"$(SERVER_LOG)"; \
			echo "Starting vLLM inference server for $(SERVER_MODEL)"; \
			echo "Logs: $(SERVER_LOG)"; \
				TMPDIR="$(SERVER_TMP_DIR)" TMP="$(SERVER_TMP_DIR)" TEMP="$(SERVER_TMP_DIR)" XDG_CACHE_HOME="$(XDG_CACHE_HOME)" TORCHINDUCTOR_CACHE_DIR="$(TORCHINDUCTOR_CACHE_DIR)" TORCHINDUCTOR_FX_GRAPH_CACHE_DIR="$(TORCHINDUCTOR_FX_GRAPH_CACHE_DIR)" TRITON_CACHE_DIR="$(TRITON_CACHE_DIR)" VLLM_ALLOW_RUNTIME_LORA_UPDATING="$(SERVER_ALLOW_RUNTIME_LORA_UPDATING)" VLLM_API_KEY="$$server_api_key" ARC3_PATCH_VLLM_RUNTIME_LORA="1" PYTHONPATH="$(REPO_ROOT):$$PYTHONPATH" PATH="$(SERVER_VENV_BIN):$$PATH" nohup "$(SERVER_VENV_VLLM)" serve "$(SERVER_MODEL)" \
					--host "$(SERVER_HOST)" \
				--port "$(SERVER_PORT)" \
				--served-model-name "$(SERVER_SERVED_MODEL_NAME)" \
				--tensor-parallel-size "$(SERVER_TENSOR_PARALLEL_SIZE)" \
				--max-model-len "$(SERVER_MAX_MODEL_LEN)" \
				$(if $(SERVER_DTYPE),--dtype "$(SERVER_DTYPE)") \
				$(if $(SERVER_GPU_MEMORY_UTILIZATION),--gpu-memory-utilization "$(SERVER_GPU_MEMORY_UTILIZATION)") \
				$(if $(filter true TRUE 1 yes YES on ON,$(SERVER_TRUST_REMOTE_CODE)),--trust-remote-code) \
				$(if $(SERVER_GENERATION_CONFIG),--generation-config "$(SERVER_GENERATION_CONFIG)") \
				$(if $(filter true,$(SERVER_LANGUAGE_MODEL_ONLY)),--language-model-only) \
				$(if $(filter true TRUE 1 yes YES on ON,$(SERVER_ENABLE_PREFIX_CACHING)),--enable-prefix-caching) \
				$(if $(filter true TRUE 1 yes YES on ON,$(SERVER_ENABLE_LORA)),--enable-lora --max-loras "$(SERVER_MAX_LORAS)" --max-lora-rank "$(SERVER_MAX_LORA_RANK)" --lora-dtype "$(SERVER_LORA_DTYPE)") \
				$(if $(SERVER_DEFAULT_CHAT_TEMPLATE_KWARGS),--default-chat-template-kwargs '$(SERVER_DEFAULT_CHAT_TEMPLATE_KWARGS)') \
				--enable-auto-tool-choice \
				--tool-call-parser "$(SERVER_TOOL_CALL_PARSER)" \
				$(if $(SERVER_REASONING_PARSER),--reasoning-parser "$(SERVER_REASONING_PARSER)") \
			>"$(SERVER_LOG)" 2>&1 & echo $$! >"$(SERVER_PID)"; \
		echo "PID: $$(cat "$(SERVER_PID)")"; \
		echo "First startup can take several minutes while vLLM loads weights and builds compile/cache artifacts."; \
			echo "Follow startup logs with: tail -f $(SERVER_LOG)"; \
	fi
	@for i in $$(seq 1 $(SERVER_START_TIMEOUT)); do \
		if [ -f "$(SERVER_PID)" ] && ! kill -0 "$$(cat "$(SERVER_PID)")" >/dev/null 2>&1; then \
			echo "Inference server process exited before becoming ready."; \
			rm -f "$(SERVER_PID)"; \
			if [ -f "$(SERVER_LOG)" ]; then \
				echo "--- final server log ---"; \
				tail -n $(SERVER_LOG_TAIL_LINES) "$(SERVER_LOG)" || true; \
				echo "--- end server log ---"; \
			fi; \
			exit 1; \
		fi; \
		if "$(REPO_ROOT)/scripts/server_curl.sh" "$(SERVER_API_KEY_FILE)" -fsS "$(SERVER_HEALTH_URL)" >/dev/null 2>&1; then server_ready=1; else server_ready=0; fi; \
		if [ "$$server_ready" = "1" ]; then \
			echo "Inference server is ready at $(SERVER_HEALTH_URL)"; \
			if [ "$(filter true TRUE 1 yes YES on ON,$(SERVER_ENABLE_LORA))" != "" ] && [ -n "$(SERVER_LORA_PATH)" ]; then \
				echo "Loading initial LoRA adapter $(SERVER_LORA_ALIAS) from $(SERVER_LORA_PATH)"; \
				"$(REPO_ROOT)/scripts/server_curl.sh" "$(SERVER_API_KEY_FILE)" -fsS -X POST "$(SERVER_LOAD_LORA_URL)" \
					-H "Content-Type: application/json" \
					--data "{\"lora_name\":\"$(SERVER_LORA_ALIAS)\",\"lora_path\":\"$(SERVER_LORA_PATH)\",\"load_inplace\":true}" >/dev/null; \
			fi; \
			exit 0; \
		fi; \
		if [ $$i -eq 1 ] || [ $$((i % $(SERVER_STATUS_EVERY))) -eq 0 ]; then \
			echo "[wait $$i/$(SERVER_START_TIMEOUT)s] waiting for $(SERVER_HEALTH_URL)"; \
			if [ "$(SERVER_TAIL_ON_WAIT)" = "true" ] && [ -f "$(SERVER_LOG)" ]; then \
				echo "--- recent server log ---"; \
				tail -n $(SERVER_LOG_TAIL_LINES) "$(SERVER_LOG)" || true; \
				echo "--- end server log ---"; \
			fi; \
		fi; \
		sleep 1; \
	done; \
	echo "Timed out waiting for inference server. Check $(SERVER_LOG)"; \
	if [ -f "$(SERVER_LOG)" ]; then \
		echo "--- final server log ---"; \
		tail -n $(SERVER_LOG_TAIL_LINES) "$(SERVER_LOG)" || true; \
		echo "--- end server log ---"; \
	fi; \
	exit 1

stop-server:
	@if [ -f "$(SERVER_PID)" ]; then \
		if kill "$$(cat "$(SERVER_PID)")" >/dev/null 2>&1; then \
			rm -f "$(SERVER_PID)" && echo "Stopped inference server"; \
		else \
			rm -f "$(SERVER_PID)" && echo "Removed stale PID file"; \
		fi; \
	else \
		echo "No PID file at $(SERVER_PID)"; \
	fi

check-server:
	@if "$(REPO_ROOT)/scripts/server_curl.sh" "$(SERVER_API_KEY_FILE)" -fsS "$(SERVER_HEALTH_URL)" >/dev/null 2>&1; then server_ready=1; else server_ready=0; fi; \
	if [ "$$server_ready" != "1" ]; then \
		echo "Inference server is not reachable at $(SERVER_HEALTH_URL)"; \
		echo "Run 'make server' and inspect $(SERVER_LOG) if it fails."; \
		exit 1; \
	fi

run: interactive

interactive: DEPLOYMENT_TARGET = inline
interactive: DEPLOYMENT_WAIT = true
interactive: _taaf-run

sbatch: CONFIG_PATH = $(SBATCH_EFFECTIVE_CONFIG_PATH)
sbatch: DEPLOYMENT_TARGET = slurm
sbatch: install-local-sources
sbatch: _taaf-run

kaggle-duck: DEPLOYMENT_TARGET = kaggle
kaggle-duck: KAGGLE_DUCK_PUBLIC_HARNESS = true
kaggle-duck: AGENT = duck-harness
kaggle-duck: GAME =
kaggle-duck: DATASET =
kaggle-duck: GAME_TAGS =
kaggle-duck: EXCLUDE_GAME_TAGS =
kaggle-duck: MODEL = local
kaggle-duck: RUN_NAME = duck-harness-kaggle
kaggle-duck: MAX_RUNTIME_MINUTES = 132
kaggle-duck: MAX_EXPERIMENT_RUNTIME_MINUTES = 540
kaggle-duck: MAX_EXPERIMENT_RUNTIME_HOURS =
kaggle-duck: N_PASSES = 1
kaggle-duck: CONCURRENT_JOBS = 28
kaggle-duck: ANALYZER_TIMEOUT = 900
kaggle-duck: _taaf-run

_taaf-run:
	@if [ -z "$(strip $(MODEL))" ]; then \
		echo "Analyzer model is required. Set analyzer.model_id in $(CONFIG_PATH) or override MODEL=..."; \
		exit 1; \
	fi
	@local_analyzer_api_key="$${LOCAL_ANALYZER_API_KEY:-}"; \
	case "$(LOCAL_ANALYZER_PROVIDER)" in \
		openrouter) \
			if [ -z "$$local_analyzer_api_key" ] && [ -z "$${OPENROUTER_API_KEY:-}" ] && [ -z "$${OPENAI_API_KEY:-}" ]; then \
				echo "OpenRouter analyzer config requires OPENROUTER_API_KEY, LOCAL_ANALYZER_API_KEY, or OPENAI_API_KEY."; \
				exit 1; \
			fi; \
			;; \
		vllm|openai|openai-compatible|compat|"") \
			if [ -z "$$local_analyzer_api_key" ] && [ -f "$(SERVER_API_KEY_FILE)" ]; then \
				local_analyzer_api_key="$$(head -n 1 "$(SERVER_API_KEY_FILE)")"; \
			fi; \
			;; \
	esac; \
	LOCAL_ANALYZER_API_KEY="$$local_analyzer_api_key" $(UV_RUN) inference-taaf-run \
		$(if $(filter-out [],$(strip $(GAME))),--game '$(GAME)',) \
		$(if $(strip $(DATASET)),--dataset '$(DATASET)',) \
		$(if $(filter-out [],$(strip $(GAME_TAGS))),--include-tags '$(GAME_TAGS)',) \
		$(if $(filter-out [],$(strip $(EXCLUDE_GAME_TAGS))),--exclude-tags '$(EXCLUDE_GAME_TAGS)',) \
		$(if $(strip $(ENVIRONMENTS_DIR)),--environments-dir '$(ENVIRONMENTS_DIR)',) \
		$(if $(RUN_NAME),--run-name '$(RUN_NAME)',) \
		$(if $(strip $(EXPERIMENTS_DIR)),--experiments-dir '$(EXPERIMENTS_DIR)',) \
		$(if $(strip $(EXPERIMENT_DIR)),--experiment-dir '$(EXPERIMENT_DIR)',) \
		$(if $(strip $(MAX_ACTIONS)),--max-actions $(MAX_ACTIONS),) \
		$(if $(strip $(MAX_RUNTIME_MINUTES)),--max-runtime-minutes $(MAX_RUNTIME_MINUTES),) \
		$(MAX_EXPERIMENT_RUNTIME_ARGS) \
		--n-passes $(N_PASSES) \
		--concurrent-jobs $(CONCURRENT_JOBS) \
		$(if $(filter true TRUE 1 yes YES on ON,$(SIMULATE_COMPETITION_ARCADE)),--simulate-competition-arcade,) \
		$(if $(strip $(COMPETITION_CLONE_RUNS)),--competition-clone-runs $(COMPETITION_CLONE_RUNS),) \
		--agent "$(AGENT)" \
		--model $(MODEL) \
		--analyzer-timeout "$(ANALYZER_TIMEOUT)" \
		--deployment-target "$(DEPLOYMENT_TARGET)" \
		$(if $(filter true TRUE 1 yes YES on ON,$(DEPLOYMENT_WAIT)),--deployment-wait,--no-deployment-wait) \
		$(if $(filter-out [],$(strip $(DEPLOYMENT_SOURCE_REPOS))),--deployment-source-repos '$(DEPLOYMENT_SOURCE_REPOS)',) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_DUCK_PUBLIC_HARNESS)),--kaggle-duck-public-harness,) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_MAKE_SHARE_VERSION)),--kaggle-make-share-version,) \
		$(if $(strip $(KAGGLE_USERNAME)),--kaggle-username '$(KAGGLE_USERNAME)',) \
		$(if $(strip $(KAGGLE_KERNEL_SLUG)),--kaggle-kernel-slug '$(KAGGLE_KERNEL_SLUG)',) \
		$(if $(strip $(KAGGLE_KERNEL_TITLE)),--kaggle-kernel-title '$(KAGGLE_KERNEL_TITLE)',) \
		$(if $(strip $(KAGGLE_DATASET_REF)),--kaggle-dataset-ref '$(KAGGLE_DATASET_REF)',) \
		$(if $(filter-out [],$(strip $(KAGGLE_ADDITIONAL_DATASET_SOURCES))),--kaggle-additional-dataset-sources '$(KAGGLE_ADDITIONAL_DATASET_SOURCES)',) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_PUBLIC)),--kaggle-public,) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_ENABLE_INTERNET)),--kaggle-enable-internet,) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_CPU_ONLY)),--kaggle-cpu-only,) \
		$(if $(strip $(KAGGLE_ACCELERATOR)),--kaggle-accelerator '$(KAGGLE_ACCELERATOR)',) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_RUN_AS_SUBMISSION)),--kaggle-run-as-submission,) \
		$(if $(filter true TRUE 1 yes YES on ON,$(KAGGLE_DRY_RUN)),--kaggle-dry-run,) \
		$(if $(strip $(KAGGLE_DATASET_VERSION_MESSAGE)),--kaggle-dataset-version-message '$(KAGGLE_DATASET_VERSION_MESSAGE)',) \
		--slurm-gpu "$(DEPLOYMENT_SLURM_GPU)" \
		--slurm-gpu-count "$(DEPLOYMENT_SLURM_GPU_COUNT)" \
		--slurm-time "$(DEPLOYMENT_SLURM_TIME)" \
		$(if $(strip $(DEPLOYMENT_SLURM_IMAGE)),--slurm-image '$(DEPLOYMENT_SLURM_IMAGE)',) \
		$(if $(strip $(DEPLOYMENT_SLURM_PARTITION)),--slurm-partition '$(DEPLOYMENT_SLURM_PARTITION)',) \
		$(if $(strip $(DEPLOYMENT_SLURM_NODELIST)),--slurm-nodelist '$(DEPLOYMENT_SLURM_NODELIST)',) \
		$(if $(filter true TRUE 1 yes YES on ON,$(DEPLOYMENT_SLURM_START_LOCAL_SERVER)),--slurm-start-local-server --slurm-local-server-config '$(DEPLOYMENT_SLURM_LOCAL_SERVER_CONFIG)',) \
		$(if $(filter-out [],$(strip $(DEPLOYMENT_SLURM_EXTRA_SBATCH_FLAGS))),--slurm-extra-sbatch-flags '$(DEPLOYMENT_SLURM_EXTRA_SBATCH_FLAGS)',) \
		$(if $(filter true TRUE 1 yes YES on ON,$(ANALYZER_SAVE_REQUEST_LOGS)),--save-request-logs,--no-save-request-logs)

chat:
	@chat_api_key="$${CHAT_API_KEY:-}"; \
	case "$(CHAT_PROVIDER)" in \
		vllm|openai|openai-compatible|compat|"") \
			if [ -z "$$chat_api_key" ] && [ -f "$(SERVER_API_KEY_FILE)" ]; then \
				chat_api_key="$$(head -n 1 "$(SERVER_API_KEY_FILE)")"; \
			fi; \
			;; \
	esac; \
	CHAT_API_KEY="$$chat_api_key" $(UV_RUN) python -m inference.tools.chat \
		--base-url "$(CHAT_BASE_URL)" \
		--provider "$(CHAT_PROVIDER)" \
		--model "$(CHAT_MODEL)" \
		--max-tokens "$(CHAT_MAX_TOKENS)" \
		--temperature "$(CHAT_TEMPERATURE)" \
		--top-p "$(CHAT_TOP_P)" \
		--top-k "$(CHAT_TOP_K)" \
		$(if $(filter true TRUE 1 yes YES,$(CHAT_THINKING)),--thinking,--no-thinking) \
		--timeout "$(CHAT_TIMEOUT)" \
		$(if $(CHAT_SYSTEM),--system "$(CHAT_SYSTEM)") \
		$(if $(PROMPT),--prompt "$(PROMPT)")

eval:
	$(UV_RUN) inference-eval --config "$(EVAL_CONFIG_PATH)" $(if $(SCORE_OUTPUT_PATH),--score-output "$(SCORE_OUTPUT_PATH)")

score_run:
	@if [ -z "$(strip $(SCORE_RUN_DIR))" ]; then \
		echo "SCORE_RUN_DIR is required."; \
		exit 1; \
	fi
	$(UV_RUN) inference-eval "$(SCORE_RUN_DIR)" $(if $(SCORE_OUTPUT_PATH),--score-output "$(SCORE_OUTPUT_PATH)")

significance:
	$(UV_RUN) inference-significance \
		--config "$(SIGNIFICANCE_CONFIG_PATH)" \
		$(if $(BASELINE_SCORE),--baseline "$(BASELINE_SCORE)") \
		$(if $(CANDIDATE_SCORE),--candidate "$(CANDIDATE_SCORE)") \
		$(if $(CONFIDENCE_THRESHOLD),--confidence-threshold "$(CONFIDENCE_THRESHOLD)") \
		$(if $(BOOTSTRAP_SAMPLES),--bootstrap-samples "$(BOOTSTRAP_SAMPLES)") \
		$(if $(BOOTSTRAP_SEED),--bootstrap-seed "$(BOOTSTRAP_SEED)") \
		$(if $(filter true TRUE 1 yes YES on ON,$(ALLOW_INTERSECTION)),--allow-intersection)

view:
	$(UV_RUN) inference-view \
		--host "$(VIEW_HOST)" \
		--port "$(VIEW_PORT)" \
		--runs-dir "$(VIEW_RUNS_DIR)" \
		$(if $(VIEW_RUN_DIR),--run-dir "$(VIEW_RUN_DIR)")

trace: traces

zip:
	@if [ ! -d "$(RUNS_DIR)" ]; then \
		echo "Runs directory not found: $(RUNS_DIR)"; \
		exit 1; \
	fi
	@rm -f "$(RUNS_ZIP)"
	@cd "$(dir $(RUNS_DIR))" && zip -rq "$(RUNS_ZIP)" "$(notdir $(RUNS_DIR))"
	@echo "Created $(RUNS_ZIP)"

traces:
	$(UV_RUN) inference-traces

cluster_view_queue:
	squeue -u "$$USER" -t R,PD

clean:
	rm -rf runs/* runs/.[!.]* runs/..?*
