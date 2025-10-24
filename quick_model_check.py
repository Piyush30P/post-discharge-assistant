"""
Quick Model Check - Tests Common Gemini Models
This is faster than the comprehensive check
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GOOGLE_API_KEY

print("="*80)
print(" QUICK MODEL CHECK")
print("="*80)

# Most common models to test
test_models = [
    "gemini-pro",
    "gemini-1.5-pro", 
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "models/gemini-pro",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash",
]

print(f"\nTesting {len(test_models)} common models...\n")
print("(This may take 30-60 seconds)\n")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    working_models = []
    
    for model_name in test_models:
        try:
            print(f"Testing: {model_name:30} ", end="", flush=True)
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                timeout=10,
            )
            
            # Quick test
            response = llm.invoke("hi")
            
            print("✅ WORKS")
            working_models.append(model_name)
            
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                print("❌ Not found")
            elif "timeout" in str(e).lower():
                print("⏱️  Timeout (might work, try again)")
            else:
                print(f"❌ {str(e)[:30]}")
    
    print("\n" + "="*80)
    print(" RESULTS")
    print("="*80)
    
    if working_models:
        print(f"\n✅ {len(working_models)} working model(s) found:\n")
        
        for i, model in enumerate(working_models, 1):
            print(f"  {i}. {model}")
        
        print("\n" + "-"*80)
        print(" RECOMMENDATION")
        print("-"*80)
        
        # Choose best model
        if "gemini-pro" in working_models:
            recommended = "gemini-pro"
        elif "gemini-1.5-flash" in working_models:
            recommended = "gemini-1.5-flash"
        elif "gemini-1.5-pro" in working_models:
            recommended = "gemini-1.5-pro"
        else:
            recommended = working_models[0]
        
        print(f"\n✅ Use this model: {recommended}\n")
        
        print("To update your code, run:")
        print(f"  python replace_model_name.py {recommended}")
        
    else:
        print("\n❌ NO WORKING MODELS FOUND")
        print("\nTroubleshooting:")
        print("  1. Check GOOGLE_API_KEY in .env file")
        print("  2. Verify API key at: https://aistudio.google.com/app/apikey")
        print("  3. Try running: python check_available_models.py")
        print("     for a more comprehensive check")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nInstall required packages:")
    print("  pip install langchain-google-genai")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
