#!/bin/bash
################################################################################
# Advanced RAG Testing Script
#
# This script will guide you through testing the Advanced RAG system.
# Run this after you've set up your .env file with API keys.
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Main script
print_header "Advanced RAG System Testing Guide"

echo "This script will guide you through testing the Advanced RAG system."
echo "Follow the steps below:"
echo ""

################################################################################
# Step 1: Check environment variables
################################################################################
print_header "STEP 1: Checking Environment Variables"

if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo ""
    echo "Please create a .env file first:"
    echo "  1. Copy the example: cp .env.example .env"
    echo "  2. Edit .env and add your API keys"
    echo "  3. Run this script again"
    echo ""
    exit 1
fi

print_success ".env file found"

# Source the .env file
set -a
source .env
set +a

# Check required variables
missing_vars=0

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_gemini_api_key_here" ]; then
    print_error "GOOGLE_API_KEY not set or still has placeholder value"
    missing_vars=1
else
    print_success "GOOGLE_API_KEY is set"
fi

if [ -z "$PINECONE_API_KEY" ] || [ "$PINECONE_API_KEY" = "your_pinecone_api_key_here" ]; then
    print_error "PINECONE_API_KEY not set or still has placeholder value"
    missing_vars=1
else
    print_success "PINECONE_API_KEY is set"
fi

if [ -z "$PINECONE_INDEX_NAME" ]; then
    print_error "PINECONE_INDEX_NAME not set"
    missing_vars=1
else
    print_success "PINECONE_INDEX_NAME is set to: $PINECONE_INDEX_NAME"
fi

if [ $missing_vars -eq 1 ]; then
    print_error "Please update your .env file with valid API keys"
    echo ""
    echo "Get your API keys from:"
    echo "  - Google Gemini: https://makersuite.google.com/app/apikey"
    echo "  - Pinecone: https://app.pinecone.io/"
    echo ""
    exit 1
fi

################################################################################
# Step 2: Check Python dependencies
################################################################################
print_header "STEP 2: Checking Python Dependencies"

print_info "Checking if required packages are installed..."

# Check for python3
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3.8+"
    exit 1
fi

print_success "Python found: $(python3 --version)"

# Try importing key packages
python3 << 'EOF'
import sys

packages = {
    "langchain": "langchain",
    "langchain_google_genai": "langchain-google-genai",
    "pinecone": "pinecone-client",
    "sentence_transformers": "sentence-transformers",
    "numpy": "numpy",
    "sklearn": "scikit-learn"
}

missing = []
for module, package in packages.items():
    try:
        __import__(module)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - NOT INSTALLED")
        missing.append(package)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("\nTo install missing packages, run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✅ All required packages are installed!")
    sys.exit(0)
EOF

if [ $? -ne 0 ]; then
    echo ""
    read -p "Install missing packages now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing dependencies..."
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "Please install dependencies before continuing"
        exit 1
    fi
fi

################################################################################
# Step 3: Check if Pinecone has data
################################################################################
print_header "STEP 3: Checking Pinecone Vector Store"

print_info "Verifying Pinecone has data..."

python3 << 'EOF'
import sys
import os
sys.path.insert(0, 'src')

try:
    from pinecone_manager import PineconeManager

    pm = PineconeManager()
    stats = pm.index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)

    if total_vectors == 0:
        print("❌ Pinecone index is empty!")
        print("\nYou need to populate Pinecone first:")
        print("  python setup_phase2.py")
        sys.exit(1)
    else:
        print(f"✅ Pinecone has {total_vectors} vectors")
        sys.exit(0)

except Exception as e:
    print(f"❌ Failed to connect to Pinecone: {e}")
    print("\nPlease check:")
    print("  1. Your PINECONE_API_KEY is correct")
    print("  2. Your PINECONE_INDEX_NAME exists")
    print("  3. You have run: python setup_phase2.py")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    read -p "Run setup_phase2.py to populate Pinecone? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Running setup_phase2.py..."
        python3 setup_phase2.py
    else
        print_error "Pinecone must have data before building Advanced RAG indices"
        exit 1
    fi
fi

################################################################################
# Step 4: Build Advanced RAG Indices
################################################################################
print_header "STEP 4: Building Advanced RAG Indices"

if [ -f "data/bm25_index.pkl" ] && [ -f "data/summary_index.pkl" ]; then
    print_success "Advanced RAG indices already exist!"
    echo ""
    read -p "Rebuild indices? This will take 10-15 minutes. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping index building"
    else
        print_info "Building Advanced RAG indices..."
        print_warning "This will take 10-15 minutes..."
        python3 setup_advanced_rag.py

        if [ $? -eq 0 ]; then
            print_success "Advanced RAG indices built successfully!"
        else
            print_error "Failed to build indices"
            exit 1
        fi
    fi
else
    print_info "Building Advanced RAG indices for the first time..."
    print_warning "This will take 10-15 minutes..."
    echo ""

    python3 setup_advanced_rag.py

    if [ $? -eq 0 ]; then
        print_success "Advanced RAG indices built successfully!"
    else
        print_error "Failed to build indices"
        exit 1
    fi
fi

################################################################################
# Step 5: Run Tests
################################################################################
print_header "STEP 5: Testing Advanced RAG System"

echo ""
echo "Choose a test to run:"
echo "  1) Quick Import Test (5 seconds)"
echo "  2) Comprehensive Evaluation (5-10 minutes)"
echo "  3) Both"
echo "  4) Skip tests"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        print_info "Running quick import test..."
        python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

print("\nTesting imports...")
from advanced_rag import AdvancedRAG, BM25Retriever
from summary_index import SummaryIndex
from agents.clinical_agent_advanced import ClinicalAgentAdvanced
print("✅ All imports successful!\n")

print("Testing BM25 retriever...")
docs = ['chronic kidney disease symptoms', 'acute renal failure treatment', 'dialysis procedure']
bm25 = BM25Retriever(docs)
results = bm25.search('kidney disease', top_k=2)
print(f"✅ BM25 search works! Found {len(results)} results\n")

print("✅ Quick test passed!")
EOF
        ;;
    2)
        print_info "Running comprehensive evaluation..."
        python3 test_advanced_rag.py
        ;;
    3)
        print_info "Running quick test first..."
        python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from advanced_rag import BM25Retriever
docs = ['chronic kidney disease', 'acute renal failure', 'dialysis']
bm25 = BM25Retriever(docs)
results = bm25.search('kidney', top_k=2)
print(f"✅ Quick test passed! Found {len(results)} results")
EOF

        echo ""
        print_info "Running comprehensive evaluation..."
        python3 test_advanced_rag.py
        ;;
    4)
        print_info "Skipping tests"
        ;;
    *)
        print_warning "Invalid choice, skipping tests"
        ;;
esac

################################################################################
# Step 6: Try a sample query
################################################################################
print_header "STEP 6: Try a Sample Query"

echo ""
read -p "Test Advanced RAG with a sample query? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running sample query..."

    python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

from agents.clinical_agent_advanced import ClinicalAgentAdvanced

print("\nInitializing Advanced Clinical Agent...")
agent = ClinicalAgentAdvanced(use_advanced_rag=True)

print("\nProcessing query: 'What are the symptoms of chronic kidney disease?'\n")
print("-" * 80)

response = agent.process(
    message="What are the symptoms of chronic kidney disease?",
    patient_context=None
)

print("\n" + "=" * 80)
print("RESPONSE:")
print("=" * 80)
print(response['message'][:500] + "..." if len(response['message']) > 500 else response['message'])
print("\n" + "=" * 80)

print(f"\nAgent used: {response.get('agent', 'unknown')}")
print(f"Response length: {len(response['message'])} characters")

if 'error' not in response:
    print("\n✅ Sample query successful!")
else:
    print(f"\n⚠️  Query had an error: {response.get('error')}")
EOF
fi

################################################################################
# Final Summary
################################################################################
print_header "Testing Complete!"

echo "Next steps:"
echo ""
echo "1. 📚 Read the documentation:"
echo "   - Quick Start: cat ADVANCED_RAG_QUICKSTART.md"
echo "   - Full Guide: cat ADVANCED_RAG_GUIDE.md"
echo ""
echo "2. 🧪 View test results:"
echo "   - cat data/advanced_rag_evaluation_results.json | jq"
echo ""
echo "3. 🚀 Use in your code:"
echo "   from src.agents.clinical_agent_advanced import ClinicalAgentAdvanced"
echo "   agent = ClinicalAgentAdvanced(use_advanced_rag=True)"
echo ""
echo "4. 💻 Start the app:"
echo "   streamlit run src/app.py"
echo ""

print_success "Advanced RAG system is ready to use!"
