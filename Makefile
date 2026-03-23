# ── BioLLMComposition: Deploy & Sync ──────────────────────────────────────
#
# Usage (TARGET defaults to "cluster"):
#   make push                          # push code to biocluster
#   make push TARGET=compute           # push code to zcorn-compute
#   make submit JOB=attention_focal    # submit on biocluster
#   make submit JOB=attention_focal TARGET=compute
#   make pull TARGET=compute
#   make setup TARGET=compute
#
# Targets: cluster (biocluster), compute (zcorn-compute)
#
# ──────────────────────────────────────────────────────────────────────────

# ── Target selection (cluster | compute) ─────────────────────────────────
TARGET     ?= cluster
LOCAL_DATA := /home/zcorn/Projects/proteinDNA_data

ifeq ($(TARGET),cluster)
  REMOTE_HOST    := biocluster
  REMOTE_USER    := yumingz5
  REMOTE_PROJECT := ~/Projects/BioLLMComposition
  REMOTE_DATA    := ~/Projects/proteinDNA_data
  SBATCH_SCRIPT  := slurm/train.sbatch
  SETUP_SCRIPT   := slurm/setup_env.sh
else ifeq ($(TARGET),compute)
  REMOTE_HOST    := zcorn-compute
  REMOTE_USER    := zcorn
  REMOTE_PROJECT := ~/Projects/BioLLMComposition
  REMOTE_DATA    := ~/Projects/proteinDNA_data
  SBATCH_SCRIPT  := slurm/train-compute.sbatch
  SETUP_SCRIPT   := slurm/setup_env_compute.sh
else
  $(error Unknown TARGET '$(TARGET)'. Use: cluster, compute)
endif

# ── Derived ──────────────────────────────────────────────────────────────
REMOTE           := $(REMOTE_HOST)
REMOTE_DEST      := $(REMOTE):$(REMOTE_PROJECT)
REMOTE_DATA_DEST := $(REMOTE):$(REMOTE_DATA)
JOBS_CONF        := slurm/jobs.conf

.PHONY: help push push-data pull setup setup-env submit submit-all \
        status logs cancel ssh clean-logs

help: ## Show this help
	@echo "TARGET=$(TARGET)  ($(REMOTE_HOST))"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Sync code ───────────────────────────────────────────────────────────
push: ## Rsync code to remote (excludes data, results, runs)
	@echo "── [$(TARGET)] Pushing code to $(REMOTE_DEST) ──"
	rsync -avz --delete \
		--filter=':- .gitignore' \
		--exclude='.git/' \
		--exclude='runs/' \
		--exclude='results/' \
		--exclude='archive/' \
		--exclude='.cursor/' \
		--exclude='slurm/logs/' \
		--exclude='notebooks/' \
		--exclude='*.egg-info/' \
		./ $(REMOTE_DEST)/
	@echo "── Code sync complete ──"

# ── Sync training data ──────────────────────────────────────────────────
push-data: ## Rsync only embeddings/ and split_dataset/
	@echo "── [$(TARGET)] Pushing data to $(REMOTE_DATA_DEST) ──"
	ssh $(REMOTE) 'mkdir -p $(REMOTE_DATA)/processed'
	rsync -avzP --include='embeddings/***' --include='split_dataset/***' --exclude='*' $(LOCAL_DATA)/processed/ $(REMOTE_DATA_DEST)/processed/
	@echo "── Data sync complete ──"

# ── Fetch results ────────────────────────────────────────────────────────
pull: ## Rsync results + runs from remote back to local
	@echo "── [$(TARGET)] Pulling results from $(REMOTE_DEST) ──"
	rsync -avzP --include='results/***' --include='runs/***' --include='slurm/logs/***' --exclude='*' $(REMOTE_DEST)/ ./
	@echo "── Results sync complete ──"

# ── First-time setup ────────────────────────────────────────────────────
setup: push push-data setup-env ## First-time: push code + data, create conda env

setup-env: ## Run environment setup on remote
	@echo "── [$(TARGET)] Setting up remote environment ──"
	ssh $(REMOTE) 'cd $(REMOTE_PROJECT) && bash -l $(SETUP_SCRIPT)'

# ── Job submission ──────────────────────────────────────────────────────
submit: ## Submit a job: make submit JOB=attention_focal [EXTRA="--lr 1e-4"]
ifndef JOB
	$(error JOB is required. Available jobs: $$(cut -d= -f1 $(JOBS_CONF) | tr '\n' ' '))
endif
	$(eval ENTRY := $(shell grep '^$(JOB)=' $(JOBS_CONF)))
	$(if $(ENTRY),,$(error Unknown JOB '$(JOB)'. Available: $$(cut -d= -f1 $(JOBS_CONF) | tr '\n' ' ')))
	$(eval SCRIPT := $(word 1,$(subst |, ,$(lastword $(subst =, ,$(ENTRY))))))
	$(eval CONFIG := $(word 2,$(subst |, ,$(lastword $(subst =, ,$(ENTRY))))))
	@echo "── [$(TARGET)] Submitting $(JOB): $(SCRIPT) + $(CONFIG) ──"
	ssh $(REMOTE) 'cd $(REMOTE_PROJECT) && mkdir -p slurm/logs && sbatch --job-name=biollm-$(JOB) $(SBATCH_SCRIPT) $(SCRIPT) $(CONFIG) $(EXTRA)'

submit-all: ## Submit all jobs defined in slurm/jobs.conf
	@while IFS='=' read -r name rest; do \
		[ -z "$$name" ] && continue; \
		echo "── [$(TARGET)] Submitting $$name ──"; \
		script=$$(echo "$$rest" | cut -d'|' -f1); \
		config=$$(echo "$$rest" | cut -d'|' -f2); \
		ssh $(REMOTE) "cd $(REMOTE_PROJECT) && mkdir -p slurm/logs && sbatch --job-name=biollm-$$name $(SBATCH_SCRIPT) $$script $$config"; \
	done < $(JOBS_CONF)

# ── Monitoring ──────────────────────────────────────────────────────────
status: ## Check SLURM queue on remote
	@ssh $(REMOTE) 'squeue -u $(REMOTE_USER) -o "%.10i %.20j %.8T %.10M %.6D %.4C %.10m %R"'

logs: ## Tail a SLURM log: make logs JOB_ID=12345
ifndef JOB_ID
	@echo "Recent logs on $(REMOTE_HOST):"
	@ssh $(REMOTE) 'ls -lt $(REMOTE_PROJECT)/slurm/logs/*.out 2>/dev/null | head -10 || echo "No logs found"'
else
	@ssh $(REMOTE) 'tail -100 $(REMOTE_PROJECT)/slurm/logs/*_$(JOB_ID).out 2>/dev/null || echo "Log not found for job $(JOB_ID)"'
endif

cancel: ## Cancel a SLURM job: make cancel JOB_ID=12345
ifndef JOB_ID
	$(error JOB_ID is required)
endif
	ssh $(REMOTE) 'scancel $(JOB_ID)'

# ── Utilities ───────────────────────────────────────────────────────────
ssh: ## Open interactive SSH session to remote
	ssh $(REMOTE)

clean-logs: ## Remove local SLURM logs
	rm -f slurm/logs/*.out slurm/logs/*.err
