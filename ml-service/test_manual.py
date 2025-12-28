"""
Manual test script for Ollama integration
Quick test of the model_loader_ollama module
"""

print("="*70)
print("MANUAL TEST - Ollama Integration")
print("="*70)

# Test 1: Import and initialization
print("\n[1/4] Testing import and initialization...")
try:
    from model_loader_ollama import get_generator
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Get generator instance
print("\n[2/4] Creating generator instance...")
try:
    gen = get_generator()
    print(f"✅ Generator created")
    print(f"   Device: {gen.device}")
    print(f"   Model: {gen.model}")
except Exception as e:
    print(f"❌ Generator creation failed: {e}")
    exit(1)

# Test 3: Generate text for executive summary
print("\n[3/4] Testing text generation for executiveSummary...")
try:
    text = gen.suggest_for_field("executiveSummary", "", 200)
    print(f"✅ Generation successful")
    print(f"\n📄 Generated Text ({len(text)} chars):")
    print("-"*70)
    print(text)
    print("-"*70)
except Exception as e:
    print(f"❌ Generation failed: {e}")
    exit(1)

# Test 4: Test with partial text
print("\n[4/4] Testing with partial text...")
try:
    partial = "The BIM objectives for this project are to"
    text = gen.suggest_for_field("bimObjectives", partial, 200)
    print(f"✅ Generation successful")
    print(f"\n📄 Input: {partial}")
    print(f"📄 Continuation ({len(text)} chars):")
    print("-"*70)
    print(text)
    print("-"*70)
except Exception as e:
    print(f"❌ Generation with partial text failed: {e}")
    exit(1)

print("\n" + "="*70)
print("🎉 ALL MANUAL TESTS PASSED!")
print("="*70)
print("\nNext steps:")
print("  1. Test API: python -m uvicorn api:app --reload --port 8000")
print("  2. Test with curl (see docs/INTEGRATION_CHANGES.md)")
print("  3. Test in browser: npm start")
print("="*70)
