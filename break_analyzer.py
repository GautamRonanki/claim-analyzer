from analyzer import analyze_text
import json

# Adversarial test cases designed to break the system
BREAK_TESTS = {
    "at_max_length": "A" * 10000,  # Exactly at limit
    
    "over_max_length": "B" * 10001,  # Just over limit
    
    "special_chars_only": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    
    "numbers_only": "123 456 789 000 111 222 333 444 555 666 777 888 999",
    
    "extremely_ambiguous": "It happened. Things changed. Results vary. Nobody knows. Something occurred.",
    
    "contradictions": """
    The sky is blue. The sky is red. 
    Water is wet. Water is dry.
    The meeting is on Monday. The meeting is on Friday.
    """,
    
    "mixed_languages": """
    The company está creciendo rapidly. 
    Das ist sehr gut for business.
    我们的产品 is the best in the market.
    """,
    
    "repeated_words": "The the the the the cat cat cat cat sat sat sat sat on on on the the mat mat mat.",
    
    "no_punctuation": "the federal reserve raised interest rates yesterday this move is expected to curb inflation which has been running high",
    
    "all_caps_aggressive": "THE COMPANY MADE $5 BILLION LAST QUARTER!!! THIS IS AMAZING!!! PROFITS ARE UP 500%!!!",
    
    "urls_and_emails": "Visit https://example.com or email contact@example.com for more info. Check out www.test.org too.",
    
    "html_tags": "<p>The study found <strong>significant results</strong> in the <em>control group</em>.</p>",
    
    "unicode_emojis": "The company 🚀 grew revenue by 50% 📈. Employees are happy 😊 and productivity is up 💪.",
    
    "json_like_text": '{"claim": "This looks like JSON", "value": 123, "nested": {"key": "value"}}',
}


def run_break_test(name: str, text: str):
    """Run analyzer on adversarial input and observe behavior."""
    print(f"\n{'='*70}")
    print(f"BREAK TEST: {name}")
    print(f"{'='*70}")
    
    # Show preview of input
    text_preview = text[:100] + "..." if len(text) > 100 else text
    print(f"Input ({len(text)} chars): {text_preview}")
    print(f"\n{'─'*70}")
    
    result = analyze_text(text, max_claims=5)
    
    if "error" in result:
        print(f"❌ ERROR (expected): {result['error']}")
        print(f"   Error type: {result['error_type']}")
    else:
        print(f"✓ Analysis completed (no error)")
        print(f"   Claims found: {result['metadata']['total_claims_found']}")
        print(f"   Tokens used: {result['metadata']['tokens_used']}")
        
        if result['claims']:
            print(f"\n   Extracted claims:")
            for i, claim in enumerate(result['claims'][:3], 1):  # Show first 3
                print(f"   {i}. [{claim['uncertainty'].upper()}] {claim['claim'][:60]}...")
            if len(result['claims']) > 3:
                print(f"   ... and {len(result['claims']) - 3} more")
    
    print(f"{'='*70}\n")
    
    return result


def main():
    """Run all break tests and summarize findings."""
    print("\n💥 CLAIM ANALYZER - ADVERSARIAL TEST SUITE")
    print("Attempting to break the system with edge cases...\n")
    
    results = {}
    
    # Run all tests
    for test_name, test_text in BREAK_TESTS.items():
        results[test_name] = run_break_test(test_name, test_text)
    
    # Summarize findings
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    errors = [name for name, result in results.items() if "error" in result]
    successes = [name for name, result in results.items() if "error" not in result]
    
    print(f"\n✓ Handled gracefully: {len(successes)}/{len(results)}")
    for name in successes:
        claims_count = results[name]['metadata']['total_claims_found']
        print(f"  - {name}: {claims_count} claims extracted")
    
    print(f"\n❌ Triggered errors: {len(errors)}/{len(results)}")
    for name in errors:
        print(f"  - {name}: {results[name]['error_type']}")
    
    print("\n" + "="*70)
    print("💡 Key insights:")
    print("- Observe which edge cases the system handles vs rejects")
    print("- Note unexpected successful extractions")
    print("- Identify patterns in failures")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()