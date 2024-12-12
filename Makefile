# Variables
# IMAGE_NAME = ia_app
# CONTAINER_NAME = ia_container
# APP_NAME = ia
PYTHON = python3
PIP = $(PYTHON) -m pip
TEST_DIR = tests
# DOCKER_PORT = 5050

# Commands
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# run:
# 	@echo "Running ia app locally..."
# 	@echo "Checking and freeing port 5050 if in use..."
# 	fuser -k 5050/tcp || true
# 	$(PYTHON) app.py

# test:
# 	$(PYTHON) -m unittest discover -s tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# docker-build:
# 	@echo "Building Docker image..."
# 	docker build -t $(IMAGE_NAME) .

# docker-run: docker-build
# 	@echo "Running pesca_ia in Docker container..."
# 	docker run -d \
# 		--name $(CONTAINER_NAME) \
# 		-e OPENAI_API_KEY=$(OPENAI_KEY) \
# 		-p $(DOCKER_PORT):5001 \
# 		$(IMAGE_NAME)

# docker-stop:
# 	docker stop $(CONTAINER_NAME)
# 	docker rm $(CONTAINER_NAME)

# docker-rebuild: docker-stop docker-build docker-run
# 	@echo "Rebuilding and running Docker container..."

# docker-clean:
# 	docker rmi $(APP_NAME)

# .PHONY: install run test clean docker-build docker-run docker-stop docker-rebuild docker-clean
