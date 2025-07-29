"""
Test runner for AI Financial Portfolio Advisor

This script runs all test suites and provides a summary of results.
Suitable for continuous integration and development workflows.
"""

import unittest
import sys
import os
from io import StringIO

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """
    Run all test suites and return results summary
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    print("🧪 Running AI Financial Portfolio Advisor Test Suite")
    print("=" * 60)
    
    # Run tests
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Overall result
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("🚀 Application is ready for deployment")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Please fix the issues before proceeding")
    
    print("=" * 60)
    
    return success

def run_specific_test(test_module):
    """
    Run a specific test module
    
    Args:
        test_module (str): Name of the test module (without .py extension)
    """
    try:
        suite = unittest.TestLoader().loadTestsFromName(test_module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    except Exception as e:
        print(f"Error running test module {test_module}: {e}")
        return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) > 1:
        # Run specific test
        test_module = sys.argv[1]
        print(f"Running specific test: {test_module}")
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 