from analyzer import analyze_text
import json

# Test texts covering different domains
TEST_TEXTS = {
    "news": """
    The Federal Reserve raised interest rates by 0.25% yesterday. 
    This move is expected to curb inflation, which has been running at 6.5% annually. 
    Economists suggest the rate hike might slow economic growth in the coming months.
    """,
    
    "product_review": """
    This laptop has a 15-inch display and weighs 3.2 pounds. 
    The battery life appears to last around 8 hours under normal use.
    Performance seems excellent for most tasks, though it might struggle with heavy video editing.
    The build quality could be better given the price point.
    """,
    
    "research": """
    The study found a correlation between sleep duration and cognitive performance.
    Participants who slept 7-8 hours scored 15% higher on memory tests.
    The researchers suggest that sleep quality may be more important than duration.
    Further research is needed to establish causation.
    """,
    
    "short_valid": "Python is a programming language. It was created by Guido van Rossum.",
    
    "too_short": "Hi there",
    
    "no_claims": "What do you think? Maybe? I wonder...",
}


def run_test(name: str, text: str, max_claims: int = 10):
    """Run analyzer on a test text and print results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Input text ({len(text)} chars):")
    print(f"{text[:100]}..." if len(text) > 100 else text)
    print(f"\n{'─'*60}")
    
    result = analyze_text(text, max_claims)
    
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        print(f"Error type: {result['error_type']}")
    else:
        print(f"✓ Analysis successful")
        print(f"\nClaims found: {result['metadata']['total_claims_found']}")
        print(f"Tokens used: {result['metadata']['tokens_used']}")
        
        if result['claims']:
            print(f"\nExtracted claims:")
            for i, claim in enumerate(result['claims'], 1):
                print(f"\n{i}. [{claim['uncertainty'].upper()}]")
                print(f"   Claim: {claim['claim']}")
                if claim['context']:
                    print(f"   Context: {claim['context']}")
        else:
            print("\nNo claims found in text")
    
    print(f"\n{'='*60}\n")


def main():
    """Run all tests."""
    print("\n🔍 CLAIM ANALYZER - TEST SUITE")
    print("Testing across different text types...\n")
    
    # Test valid cases
    run_test("News Article", TEST_TEXTS["news"], max_claims=5)
    run_test("Product Review", TEST_TEXTS["product_review"], max_claims=5)
    run_test("Research Abstract", TEST_TEXTS["research"], max_claims=5)
    run_test("Short Valid Text", TEST_TEXTS["short_valid"], max_claims=5)
    
    # Test edge cases
    run_test("Text Too Short", TEST_TEXTS["too_short"])
    run_test("No Claims Present", TEST_TEXTS["no_claims"])
    
    print("\n✅ Test suite complete!")


if __name__ == "__main__":
    main()