services:
  post-discharge-assistant:
    build: .
    container_name: post-discharge-assistant
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
    env_file:
      - .env
    volumes:
      # Mount data directories to persist data
      - ./data:/app/data
      - ./logs:/app/logs
      - ./vectorstore:/app/vectorstore
    restart: unless-stopped
    networks:
      - app-network

  # Optional: Add a database service if needed
  # postgres:
  #   image: postgres:15
  #   container_name: postgres-db
  #   environment:
  #     POSTGRES_DB: post_discharge
  #     POSTGRES_USER: postgres
  #     POSTGRES_PASSWORD: password
  #   ports:
  #     - "5432:5432"
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   networks:
  #     - app-network

networks:
  app-network:
    driver: bridge
# volumes:
#   postgres_data:
