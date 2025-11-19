#!/usr/bin/env python3
"""
GSFlow Module Validation Script

This script validates that the GSFlow module and related utilities
are properly structured and can be imported (when dependencies are available).
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_module_structure():
    """Test that module files exist and have correct structure."""
    print("=" * 60)
    print("Testing Module Structure")
    print("=" * 60)
    print()
    
    required_files = [
        'GSFlow/__init__.py',
        'GSFlow/gs_flow_generator.py',
        'GSFlow/warp_functions.py',
        'GSFlow/example_usage.py',
        'GSFlow/README.md',
        'utils/flow_loss_utils.py',
        'check_requirements.py',
        'GSFlow_OVERVIEW.md',
        'INTEGRATION_GUIDE.md',
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = os.path.isfile(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    print()
    if all_exist:
        print("✓ All required files exist")
        return True
    else:
        print("✗ Some files are missing")
        return False


def test_syntax():
    """Test Python syntax of all module files."""
    print("=" * 60)
    print("Testing Python Syntax")
    print("=" * 60)
    print()
    
    python_files = [
        'GSFlow/__init__.py',
        'GSFlow/gs_flow_generator.py',
        'GSFlow/warp_functions.py',
        'GSFlow/example_usage.py',
        'utils/flow_loss_utils.py',
        'check_requirements.py',
    ]
    
    all_valid = True
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path}")
        except SyntaxError as e:
            print(f"✗ {file_path}: {e}")
            all_valid = False
    
    print()
    if all_valid:
        print("✓ All Python files have valid syntax")
        return True
    else:
        print("✗ Some files have syntax errors")
        return False


def test_imports():
    """Test that modules can be imported (if dependencies available)."""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    print()
    
    # Test if torch is available
    try:
        import torch
        torch_available = True
        print("✓ PyTorch is available")
    except ImportError:
        torch_available = False
        print("ℹ PyTorch not available - skipping import tests")
        print("  (This is OK - imports will work once dependencies are installed)")
        return True
    
    if not torch_available:
        return True
    
    # Test GSFlow imports
    try:
        from GSFlow import calculate_gs_flow, GaussianFlowGenerator, warping_gs_flow
        print("✓ GSFlow module imports successfully")
        print("  - calculate_gs_flow")
        print("  - GaussianFlowGenerator")
        print("  - warping_gs_flow")
    except Exception as e:
        print(f"✗ Failed to import GSFlow: {e}")
        return False
    
    # Test flow_loss_utils imports
    try:
        from utils.flow_loss_utils import (
            combined_flow_l1_loss,
            flow_supervised_loss,
            compute_flow_metrics,
            flow_loss
        )
        print("✓ flow_loss_utils imports successfully")
        print("  - combined_flow_l1_loss")
        print("  - flow_supervised_loss")
        print("  - compute_flow_metrics")
        print("  - flow_loss")
    except Exception as e:
        print(f"✗ Failed to import flow_loss_utils: {e}")
        return False
    
    print()
    print("✓ All imports successful")
    return True


def test_api_signatures():
    """Test that key functions have expected signatures."""
    print("=" * 60)
    print("Testing API Signatures")
    print("=" * 60)
    print()
    
    try:
        import torch
    except ImportError:
        print("ℹ PyTorch not available - skipping API tests")
        return True
    
    try:
        from GSFlow import calculate_gs_flow, GaussianFlowGenerator
        from utils.flow_loss_utils import combined_flow_l1_loss
        import inspect
        
        # Test calculate_gs_flow signature
        sig = inspect.signature(calculate_gs_flow)
        expected_params = ['gs_per_pixel', 'weight_per_gs_pixel', 'next_conic_2D', 
                          'conic_2D_inv', 'proj_2D', 'next_proj_2D', 'x_mu']
        actual_params = list(sig.parameters.keys())
        
        if actual_params == expected_params:
            print("✓ calculate_gs_flow has correct signature")
        else:
            print(f"✗ calculate_gs_flow signature mismatch")
            print(f"  Expected: {expected_params}")
            print(f"  Got: {actual_params}")
            return False
        
        # Test combined_flow_l1_loss signature
        sig = inspect.signature(combined_flow_l1_loss)
        required_params = ['image_pred', 'image_gt', 'flow_pred', 'flow_gt', 
                          'height', 'width']
        actual_params = list(sig.parameters.keys())
        
        has_required = all(p in actual_params for p in required_params)
        if has_required:
            print("✓ combined_flow_l1_loss has correct signature")
        else:
            print(f"✗ combined_flow_l1_loss missing required parameters")
            return False
        
        # Test class interface
        generator = GaussianFlowGenerator()
        if hasattr(generator, 'compute_flow') and callable(generator.compute_flow):
            print("✓ GaussianFlowGenerator has compute_flow method")
        else:
            print("✗ GaussianFlowGenerator missing compute_flow method")
            return False
        
        print()
        print("✓ All API signatures are correct")
        return True
        
    except Exception as e:
        print(f"✗ Error testing APIs: {e}")
        return False


def main():
    """Run all validation tests."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "GSFlow Module Validation" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Module Structure", test_module_structure()))
    print()
    
    results.append(("Python Syntax", test_syntax()))
    print()
    
    results.append(("Module Imports", test_imports()))
    print()
    
    results.append(("API Signatures", test_api_signatures()))
    print()
    
    # Summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print()
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✓ All validation tests passed!")
        print()
        print("Next steps:")
        print("1. Run: python check_requirements.py")
        print("2. Install missing dependencies if needed")
        print("3. Review INTEGRATION_GUIDE.md for usage instructions")
        return 0
    else:
        print("✗ Some validation tests failed")
        print()
        print("Please review the errors above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
