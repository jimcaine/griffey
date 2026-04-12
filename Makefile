SHELL := /bin/bash
PROJECT_NAME := $(shell sed -n 's/^name = "\(.*\)"/\1/p' pyproject.toml)
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

about:
	@echo "PROJECT_NAME: $(PROJECT_NAME)"
	@echo "VERSION: $(VERSION)"

increment-patch-version:
	@echo "Incrementing patch version..."
	@CURRENT_VERSION=$$(toml get --toml-path pyproject.toml project.version) ; \
	IFS='.' read -r major minor patch <<< "$$CURRENT_VERSION" ; \
	patch=$$((patch + 1)) ; \
	NEW_VERSION="$${major}.$${minor}.$$patch" ; \
	toml set --toml-path pyproject.toml project.version "$$NEW_VERSION" ; \
	echo "Version updated to $$NEW_VERSION"

pr: test
	@echo "Creating PR"
	@gh pr create --base main --head ${CURRENT_BRANCH} --title "Release: ${VERSION}"

pr-dev: test
	@echo "Creating PR"
	@gh pr create --base dev --head ${CURRENT_BRANCH} --title "${CURRENT_BRANCH}"

test:
	@pytest -s -v tests --log-cli-level=INFO