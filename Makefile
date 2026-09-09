UV := uv
# Every `uv run` otherwise re-resolves and re-syncs, which fights the mutually
# exclusive PyTorch extras: a bare `uv run` picks its own flavour and undoes
# whichever one is installed.
export UV_NO_SYNC := 1

#: Everything whose Python this repository owns. Not `.`: see `lint`.
OWNED := src tests notebooks

.PHONY: help lint check test test-cov test-notebooks hooks

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

lint: ## Lint, check formatting, and check prose
	$(UV) run ruff check $(OWNED)
	$(UV) run ruff format --check $(OWNED)
	$(UV) run numpydoc lint $$(git ls-files 'src/npcc/*.py' 'src/npcc/**/*.py')
# Tracked files, for two reasons: a shell glob would also read whatever
# untracked notes are in the working tree, so a local run could fail where CI
# passes; and `.claude/` holds agent definitions rather than project prose.
	git ls-files -z ':!:.claude/*' | xargs -0 $(UV) run codespell --

check: lint ## `lint` plus the type check
	$(UV) run ty check

test: ## Run the test suite in parallel
	$(UV) run pytest tests/ -v -n auto

test-cov: ## Run the test suite with a coverage report
	$(UV) run pytest tests/ --cov=src/npcc --cov-report=term-missing -v -n auto

test-notebooks: ## Execute the notebooks; needs TABPFN_TOKEN
	$(UV) run pytest --nbmake notebooks/ -v

hooks: ## Install the pre-commit hooks
	$(UV) run pre-commit install
