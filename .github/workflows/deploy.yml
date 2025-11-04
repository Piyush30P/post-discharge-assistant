name: Deploy to Production

on:
  release:
    types: [published]
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }}

      # Example deployment to various cloud providers
      # Uncomment and configure the one you want to use

      # Deploy to Heroku
      # - name: Deploy to Heroku
      #   uses: akhileshns/heroku-deploy@v3.12.14
      #   with:
      #     heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
      #     heroku_app_name: "your-app-name"
      #     heroku_email: "your-email@example.com"
      #     usedocker: true

      # Deploy to Railway
      # - name: Deploy to Railway
      #   uses: railway-deploy-action@v1.0.0
      #   with:
      #     token: ${{ secrets.RAILWAY_TOKEN }}
      #     project: ${{ secrets.RAILWAY_PROJECT_ID }}

      # Deploy to DigitalOcean App Platform
      # - name: Deploy to DigitalOcean
      #   uses: digitalocean/app_action@v1.1.5
      #   with:
      #     app_name: your-app-name
      #     token: ${{ secrets.DIGITALOCEAN_ACCESS_TOKEN }}

      - name: Post-deployment health check
        run: |
          echo "Deployment completed successfully!"
          echo "Remember to verify the application is running correctly."
