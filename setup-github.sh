#!/bin/bash

echo "Setting up GitHub repository for Post-Discharge Assistant..."
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    echo "Visit: https://git-scm.com/downloads"
    exit 1
fi

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
fi

# Check if we have any commits
if ! git log --oneline -1 &> /dev/null; then
    echo "Adding files to Git..."
    git add .
    echo
    echo "Committing files..."
    git commit -m "Initial commit: Post-Discharge Medical AI Assistant with Docker"
fi

echo
echo "============================================"
echo "   GITHUB SETUP INSTRUCTIONS"
echo "============================================"
echo
echo "1. Go to GitHub.com and create a new repository"
echo "2. Name it: post-discharge-assistant"
echo "3. Make it public or private (your choice)"
echo "4. DO NOT initialize with README, .gitignore, or license"
echo
echo "5. Copy the repository URL (it will look like):"
echo "   https://github.com/YOUR_USERNAME/post-discharge-assistant.git"
echo

read -p "Enter your GitHub repository URL: " repo_url

if [ -z "$repo_url" ]; then
    echo "Error: No repository URL provided."
    exit 1
fi

echo
echo "Adding remote origin..."
git remote remove origin 2>/dev/null || true
git remote add origin "$repo_url"

echo
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo
    echo "✅ SUCCESS! Your project is now on GitHub!"
    echo
    echo "Next steps:"
    echo "1. Visit your repository: $repo_url"
    echo "2. Set up repository secrets for API keys (if deploying)"
    echo "3. Check out DEPLOYMENT.md for deployment options"
    echo
else
    echo
    echo "❌ Push failed. Please check:"
    echo "1. Repository URL is correct"
    echo "2. You have access to the repository"
    echo "3. You're logged into Git (git config user.name/user.email)"
    echo
fi