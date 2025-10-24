"""
Check Available Gemini Models
This script lists all available Gemini models and tests which ones work
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import GOOGLE_API_KEY

print("="*80)
print(" CHECKING AVAILABLE GEMINI MODELS")
print("="*80)

# Method 1: Using google.generativeai directly
print("\n" + "="*80)
print(" METHOD 1: Using google.generativeai")
print("="*80)

try:
    import google.generativeai as genai
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    print("\n✅ Successfully configured Google AI")
    print("\nAvailable models that support generateContent:\n")
    
    models_list = []
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            model_name = model.name
            # Extract just the model name without 'models/' prefix
            short_name = model_name.replace('models/', '')
            
            print(f"  ✅ {model_name}")
            print(f"     Short name: {short_name}")
            print(f"     Display name: {model.display_name}")
            print(f"     Description: {model.description[:100]}...")
            print()
            
            models_list.append({
                'full': model_name,
                'short': short_name,
                'display': model.display_name
            })
    
    print(f"\nTotal models found: {len(models_list)}")
    
except Exception as e:
    print(f"❌ Error with google.generativeai: {e}")
    models_list = []

# Method 2: Test with LangChain
print("\n" + "="*80)
print(" METHOD 2: Testing with LangChain ChatGoogleGenerativeAI")
print("="*80)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Common model names to test
    test_models = [
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
        "models/gemini-pro",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
    ]
    
    # Add models from Method 1
    if models_list:
        for model_info in models_list:
            if model_info['short'] not in test_models:
                test_models.append(model_info['short'])
            if model_info['full'] not in test_models:
                test_models.append(model_info['full'])
    
    print(f"\nTesting {len(test_models)} model names with LangChain...\n")
    
    working_models = []
    
    for model_name in test_models:
        try:
            print(f"Testing: {model_name}... ", end="", flush=True)
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
            )
            
            # Try a simple invocation
            response = llm.invoke("Say 'test' and nothing else")
            
            print(f"✅ WORKS!")
            print(f"   Response: {response.content[:50]}")
            
            working_models.append(model_name)
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print("❌ Not found (404)")
            elif "not supported" in error_msg.lower():
                print("❌ Not supported")
            else:
                print(f"❌ Error: {error_msg[:50]}")
    
    print("\n" + "="*80)
    print(" RESULTS SUMMARY")
    print("="*80)
    
    if working_models:
        print(f"\n✅ {len(working_models)} WORKING MODEL(S) FOUND:\n")
        
        for i, model in enumerate(working_models, 1):
            print(f"  {i}. {model}")
        
        print("\n" + "-"*80)
        print(" RECOMMENDED MODEL FOR YOUR CODE:")
        print("-"*80)
        
        # Prefer simpler names
        recommended = working_models[0]
        for model in working_models:
            if model == "gemini-pro":
                recommended = "gemini-pro"
                break
            elif "gemini-1.5" in model and "models/" not in model:
                recommended = model
                break
        
        print(f"\n  Use: {recommended}")
        
        print("\n" + "-"*80)
        print(" HOW TO UPDATE YOUR CODE:")
        print("-"*80)
        
        print(f"""
1. Open: src/agents/receptionist_agent.py
   Line ~20, change to:
   model="{recommended}"

2. Open: src/agents/clinical_agent.py
   Line ~22, change to:
   model="{recommended}"

3. Save both files

4. Run: streamlit run src/app.py
""")
        
    else:
        print("\n❌ NO WORKING MODELS FOUND!")
        print("\nPossible issues:")
        print("  1. Invalid GOOGLE_API_KEY in .env")
        print("  2. API key doesn't have access to Gemini models")
        print("  3. Network/firewall blocking Google AI API")
        print("\nPlease check:")
        print("  - Your API key at: https://makersuite.google.com/app/apikey")
        print("  - That Gemini API is enabled for your project")

except Exception as e:
    print(f"\n❌ Error testing with LangChain: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print(" CHECK COMPLETE")
print("="*80)