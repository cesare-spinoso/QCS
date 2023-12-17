# Signifies our desired python version
# Makefile macros (or variables) are defined a little bit differently than traditional bash, keep in mind that in the Makefile there's top-level Makefile-only syntax, and everything else is bash script syntax.
PYTHON = python3

# .PHONY defines parts of the makefile that are not dependent on any specific file
# This is most often used to store functions
.PHONY = help setup test
TEST_VAR = $(shell pip list | grep pytest)

# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type make setup"
	@echo "To run the project's tests type make test"
	@echo "------------------------------------"

# BEWARE OF MAKEFILE SPACING
test:
	@pytest -vv

# To update the requirements pip list --format=freeze --local > requirements.txt

setup:
	@echo "Setting up the project"
	@pip install -e .
	@echo "Installing requirements"
	@pip install -r requirements.txt
	@echo "Successfully installed requirements"
