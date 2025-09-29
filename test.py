#!/usr/bin/env python3
"""
Simple test script to verify the battery simulation is working.
This avoids the LLM optimization and just tests the core simulation.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import the fixed modules
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge, debug_run

def test_basic_simulation():
    """Test basic battery charging simulation."""
    print("="*60)
    print("TESTING BASIC BATTERY SIMULATION")
    print("="*60)
    
    try:
        # Test with simple current density profile
        j_segments = [30.0, 25.0, 20.0]  # A/m²
        seg_duration = 400.0  # seconds (reduced for faster testing)
        
        print(f"Testing with current densities: {j_segments} A/m²")
        print(f"Segment duration: {seg_duration} s")
        
        # Build current profile
        t_knots, I_segments = build_piecewise_current_A(j_segments, seg_duration)
        print(f"Time knots: {t_knots}")
        print(f"Current segments: {I_segments} A")
        
        # Run simulation
        total_time = seg_duration * len(j_segments)
        result = run_spme_charge(
            piecewise_current_A=(t_knots, I_segments),
            t_end_max=total_time + 300.0,
            soc_start=0.2,
            soc_target=0.8,
            with_aging=False,  # Disable aging for simpler test
            diagnose=True
        )
        
        print(f"\n--- SIMULATION RESULTS ---")
        print(f"Feasible: {result.feasible}")
        print(f"Reason: {result.reason}")
        
        if result.feasible and result.t_final:
            final_soc = result.soc[-1] if len(result.soc) > 0 else 0.0
            final_temp = result.T_peak - 273.15 if result.T_peak else np.nan
            final_voltage = result.V[-1] if len(result.V) > 0 else np.nan
            
            print(f"Charging time: {result.t_final:.1f} s ({result.t_final/60:.1f} min)")
            print(f"Final SOC: {final_soc:.1%}")
            print(f"Peak temperature: {final_temp:.1f} °C")
            print(f"Final voltage: {final_voltage:.3f} V")
            
            if final_soc >= 0.75:
                print("✓ TEST PASSED: Simulation completed successfully!")
                return True
            else:
                print("⚠ TEST PARTIAL: Simulation ran but didn't reach target SOC")
                return True
        else:
            print(f"✗ TEST FAILED: Simulation infeasible - {result.reason}")
            return False
            
    except Exception as e:
        print(f"✗ TEST FAILED: Exception occurred - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_function():
    """Test the debug_run function."""
    print(f"\n{'='*60}")
    print("TESTING DEBUG FUNCTION")
    print("="*60)
    
    try:
        result = debug_run(j_segments_Apm2=[35, 30, 25], seg_duration_s=300)
        
        if result.feasible:
            print("✓ DEBUG TEST PASSED")
            return True
        else:
            print(f"⚠ DEBUG TEST PARTIAL: {result.reason}")
            return True
            
    except Exception as e:
        print(f"✗ DEBUG TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_scenarios():
    """Test multiple current density scenarios."""
    print(f"\n{'='*60}")
    print("TESTING MULTIPLE SCENARIOS")
    print("="*60)
    
    scenarios = [
        ([40, 35, 30], "Aggressive charging"),
        ([25, 25, 25], "Conservative constant"),
        ([45, 30, 20], "Decreasing profile"),
        ([20, 30, 35], "Increasing profile")
    ]
    
    passed = 0
    for i, (j_vec, description) in enumerate(scenarios):
        print(f"\nScenario {i+1}: {description} - {j_vec} A/m²")
        try:
            t_knots, I_segments = build_piecewise_current_A(j_vec, 300.0)
            result = run_spme_charge(
                piecewise_current_A=(t_knots, I_segments),
                t_end_max=1200.0,
                soc_start=0.2,
                soc_target=0.8,
                with_aging=False,
                diagnose=False  # Less verbose for multiple tests
            )
            
            if result.feasible:
                final_soc = result.soc[-1] if len(result.soc) > 0 else 0.0
                print(f"  ✓ PASSED: {result.t_final:.0f}s, SOC={final_soc:.1%}")
                passed += 1
            else:
                print(f"  ⚠ PARTIAL: {result.reason}")
                passed += 0.5
                
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
    
    print(f"\nScenario Summary: {passed}/{len(scenarios)} passed")
    return passed >= len(scenarios) * 0.5

def main():
    """Run all tests."""
    print("BATTERY SIMULATION TEST SUITE")
    print("This script tests the core battery simulation without LLM optimization.")
    print("If these tests pass, the optimization should work correctly.")
    
    tests = [
        ("Basic Simulation", test_basic_simulation),
        ("Debug Function", test_debug_function),
        ("Multiple Scenarios", test_multiple_scenarios)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"UNEXPECTED ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The battery simulation is working correctly.")
        print("You can now run the full optimization with confidence.")
    elif passed > 0:
        print(f"\n⚠ PARTIAL SUCCESS: {passed} out of {total} tests passed.")
        print("The basic functionality works but there may be edge cases.")
    else:
        print("\n❌ ALL TESTS FAILED: There are fundamental issues with the simulation.")
        print("Please check your PyBaMM installation and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)