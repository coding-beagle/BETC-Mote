.PHONY: clean clean-venv clean-build clean-cache venv build install install-dev test

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?##/ { printf "\033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

clean: ## Clean any generated files and environments
	{ \
		make clean-venv ;\
		make clean-build ;\
		make clean-cache ;\
	}


clean-venv: ## Clean virtual environment
	rm -rf .venv

clean-build: ## Clean build folder
	rm -rf dist

clean-cache: ## Clean .pyc and .egg-info files
	{ \
    	rm -rf build/ dist/ *.egg-info/ ;\
    	find . -type d -name __pycache__ -delete ;\
    	find . -name "*.pyc" -delete	;\
	}
	
venv: ## Make a new virtual environment
	python -m venv .venv

build: ## Create a wheel distribution
	python setup.py sdist bdist_wheel

install: ## Install the module
	{ \
		. .venv/scripts/activate ;\
		pip install -e . ;\
	}

install-dev: ## Install the module + dev tools
	{ \
		. .venv/scripts/activate ;\
		pip install -e .[dev] ;\
	}

test: ## Run pytest tests
	{ \
		. .venv/scripts/activate ;\
		python -m pytest Tests/ -v ;\
	}
