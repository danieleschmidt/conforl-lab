#!/usr/bin/env python3
"""
Simple ConfoRL CLI Demo
A minimal command-line interface demonstrating core ConfoRL functionality.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conforl

def demo_conformal_prediction(coverage=0.95):
    """Demonstrate conformal prediction functionality."""
    print(f"🔮 Conformal Prediction Demo (coverage: {coverage})")
    
    from conforl.core.conformal import SplitConformalPredictor
    
    predictor = SplitConformalPredictor(coverage=coverage)
    print(f"   ✅ Predictor created with {coverage} coverage")
    
    # Simulate calibration data
    calibration_scores = [0.1, 0.2, 0.15, 0.25, 0.18, 0.22, 0.12, 0.28, 0.19, 0.16]
    predictor.calibrate(calibration_scores)
    print(f"   📊 Calibrated on {len(calibration_scores)} samples")
    
    # Get prediction interval
    quantile = predictor.quantile
    print(f"   🎯 Quantile threshold: {quantile:.3f}")
    
    # Test prediction interval
    test_pred = 0.5
    lower, upper = predictor.get_prediction_interval([test_pred])
    print(f"   📏 Prediction interval for {test_pred}: [{lower[0]:.3f}, {upper[0]:.3f}]")
    
    return quantile

def demo_risk_control(target_risk=0.05):
    """Demonstrate adaptive risk control."""
    print(f"⚡ Risk Control Demo (target: {target_risk})")
    
    # Create components
    controller = conforl.AdaptiveRiskController(target_risk=target_risk)
    risk_measure = conforl.SafetyViolationRisk(violation_threshold=0.1)
    
    print(f"   ✅ Controller created with {target_risk} target risk")
    
    # Create sample trajectory
    trajectory = conforl.TrajectoryData(
        states=[[i*0.1, (i+1)*0.1] for i in range(5)],
        actions=[i % 2 for i in range(5)],
        rewards=[0.1 * (-1)**i for i in range(5)],
        dones=[False, False, False, False, True],
        infos=[{} for _ in range(5)]
    )
    
    # Update controller
    controller.update(trajectory, risk_measure)
    risk_bound = controller.get_risk_bound()
    certificate = controller.get_certificate()
    
    print(f"   📊 Risk bound: {risk_bound:.3f}")
    print(f"   🏆 Certificate confidence: {certificate.confidence:.3f}")
    
    return risk_bound

def demo_safety_certificate():
    """Demonstrate safety certificate generation."""
    print("🔒 Safety Certificate Demo")
    
    certificate = conforl.RiskCertificate(
        risk_bound=0.042,
        confidence=0.95,
        coverage_guarantee=0.947,
        method="adaptive_conformal",
        sample_size=150,
        metadata={
            "algorithm": "demo_sac",
            "environment": "simple_demo",
            "safety_verified": True
        }
    )
    
    print(f"   ✅ Certificate generated:")
    print(f"      🎯 Risk Bound: {certificate.risk_bound:.3f}")
    print(f"      📈 Confidence: {certificate.confidence:.3f}") 
    print(f"      ⚙️  Method: {certificate.method}")
    print(f"      📊 Sample Size: {certificate.sample_size}")
    
    # Safety check
    is_safe = certificate.risk_bound < 0.05
    print(f"   {'🟢 SAFE' if is_safe else '🔴 UNSAFE'}: Risk bound {'below' if is_safe else 'above'} 5% threshold")
    
    return certificate

def main():
    parser = argparse.ArgumentParser(description="ConfoRL Simple CLI Demo")
    parser.add_argument("--coverage", type=float, default=0.95, help="Coverage level for conformal prediction")
    parser.add_argument("--target-risk", type=float, default=0.05, help="Target risk level")
    parser.add_argument("--demo", choices=["all", "conformal", "risk", "certificate"], 
                       default="all", help="Which demo to run")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 ConfoRL Simple CLI Demo")
    print("=" * 50)
    print(f"📦 ConfoRL v{conforl.__version__}")
    print(f"🧰 Available: {', '.join(conforl.__all__)}")
    print()
    
    results = {}
    
    if args.demo in ["all", "conformal"]:
        results["conformal"] = demo_conformal_prediction(args.coverage)
        print()
    
    if args.demo in ["all", "risk"]:
        results["risk"] = demo_risk_control(args.target_risk)
        print()
    
    if args.demo in ["all", "certificate"]:
        results["certificate"] = demo_safety_certificate()
        print()
    
    print("=" * 50)
    print("✅ Demo Complete!")
    
    if args.demo == "all":
        print("📋 Summary:")
        print(f"   🔮 Conformal quantile: {results.get('conformal', 'N/A'):.3f}")
        print(f"   ⚡ Risk bound: {results.get('risk', 'N/A'):.3f}")
        print(f"   🔒 Certificate risk: {results.get('certificate', {}).risk_bound:.3f}")
    
    print("🎯 ConfoRL is ready for production deployment!")
    print("=" * 50)

if __name__ == "__main__":
    main()