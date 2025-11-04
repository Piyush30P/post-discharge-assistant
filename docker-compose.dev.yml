services:
  post-discharge-assistant-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: post-discharge-assistant-dev
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
    env_file:
      - .env
    volumes:
      # Mount source code for hot reload
      - ./src:/app/src
      - ./data:/app/data
      - ./logs:/app/logs
      - ./vectorstore:/app/vectorstore
    restart: unless-stopped
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
