"""Risk monitoring and logging for deployment."""

import numpy as np
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict

from ..core.types import StateType, ActionType


class RiskMonitor:
    """Real-time risk monitoring during deployment."""
    
    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize risk monitor.
        
        Args:
            window_size: Size of sliding window for statistics
            alert_thresholds: Thresholds for different risk metrics
        """
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'reward_drop': -0.5,  # Alert if reward drops by 50%
            'action_variance': 2.0,  # Alert if action variance exceeds 2x baseline
            'constraint_violations': 0.1  # Alert if >10% of recent steps violate constraints
        }
        
        # Sliding windows for monitoring
        self.rewards = deque(maxlen=window_size)
        self.actions = deque(maxlen=window_size)
        self.constraint_violations = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Baseline statistics (computed from initial data)
        self.baseline_reward_mean = None
        self.baseline_reward_std = None
        self.baseline_action_std = None
        
        # Alert tracking
        self.alerts = []
        self.baseline_computed = False
        
    def update(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        info: Dict[str, Any]
    ) -> None:
        """Update monitor with new step data.
        
        Args:
            state: Environment state
            action: Action taken
            reward: Reward received
            info: Environment info dict
        """
        current_time = time.time()
        
        # Store data
        self.rewards.append(reward)
        self.actions.append(action)
        self.timestamps.append(current_time)
        
        # Extract constraint violations from info
        violation = info.get('constraint_violation', 0.0) > 0
        self.constraint_violations.append(violation)
        
        # Compute baseline after collecting initial data
        if len(self.rewards) >= min(50, self.window_size) and not self.baseline_computed:
            self._compute_baseline()
        
        # Check for alerts if baseline is available
        if self.baseline_computed:
            self._check_alerts()
    
    def _compute_baseline(self) -> None:
        """Compute baseline statistics from initial data."""
        rewards_array = np.array(list(self.rewards))
        actions_array = np.array(list(self.actions))
        
        self.baseline_reward_mean = np.mean(rewards_array)
        self.baseline_reward_std = np.std(rewards_array)
        
        if actions_array.ndim > 1:
            self.baseline_action_std = np.mean(np.std(actions_array, axis=0))
        else:
            self.baseline_action_std = np.std(actions_array)
        
        self.baseline_computed = True
        
        print(f"Risk monitor baseline computed: "
              f"reward_mean={self.baseline_reward_mean:.3f}, "
              f"reward_std={self.baseline_reward_std:.3f}, "
              f"action_std={self.baseline_action_std:.3f}")
    
    def _check_alerts(self) -> None:
        """Check for risk alerts based on current statistics."""
        current_time = time.time()
        
        # Recent reward performance
        recent_rewards = list(self.rewards)[-min(20, len(self.rewards)):]
        recent_reward_mean = np.mean(recent_rewards)
        
        reward_drop = (recent_reward_mean - self.baseline_reward_mean) / abs(self.baseline_reward_mean)
        if reward_drop < self.alert_thresholds['reward_drop']:
            self._trigger_alert(
                "reward_drop",
                f"Recent reward dropped by {abs(reward_drop)*100:.1f}% from baseline",
                current_time
            )
        
        # Action variance check
        recent_actions = list(self.actions)[-min(20, len(self.actions)):]
        if len(recent_actions) > 1:
            actions_array = np.array(recent_actions)
            if actions_array.ndim > 1:
                current_action_std = np.mean(np.std(actions_array, axis=0))
            else:
                current_action_std = np.std(actions_array)
            
            variance_ratio = current_action_std / max(self.baseline_action_std, 1e-6)
            if variance_ratio > self.alert_thresholds['action_variance']:
                self._trigger_alert(
                    "action_variance",
                    f"Action variance increased by {variance_ratio:.1f}x baseline",
                    current_time
                )
        
        # Constraint violation rate
        recent_violations = list(self.constraint_violations)[-min(20, len(self.constraint_violations)):]
        violation_rate = np.mean(recent_violations) if recent_violations else 0
        
        if violation_rate > self.alert_thresholds['constraint_violations']:
            self._trigger_alert(
                "constraint_violations",
                f"Constraint violation rate: {violation_rate*100:.1f}%",
                current_time
            )
    
    def _trigger_alert(self, alert_type: str, message: str, timestamp: float) -> None:
        """Trigger a risk alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            timestamp: Time of alert
        """
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": timestamp,
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        }
        
        self.alerts.append(alert)
        print(f"RISK ALERT [{alert_type}]: {message}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics.
        
        Returns:
            Dict with current risk monitoring stats
        """
        if len(self.rewards) == 0:
            return {"status": "no_data"}
        
        stats = {
            "window_size": len(self.rewards),
            "recent_reward_mean": float(np.mean(list(self.rewards)[-10:])),
            "recent_reward_std": float(np.std(list(self.rewards)[-10:])),
            "constraint_violation_rate": float(np.mean(list(self.constraint_violations)[-20:])),
            "total_alerts": len(self.alerts),
            "baseline_computed": self.baseline_computed
        }
        
        if self.baseline_computed:
            stats.update({
                "baseline_reward_mean": self.baseline_reward_mean,
                "baseline_reward_std": self.baseline_reward_std,
                "baseline_action_std": self.baseline_action_std,
                "reward_drift": (stats["recent_reward_mean"] - self.baseline_reward_mean) / abs(self.baseline_reward_mean)
            })
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if time.time() - a["timestamp"] < 300]  # Last 5 minutes
        stats["recent_alerts"] = len(recent_alerts)
        
        return stats
    
    def get_alerts(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get risk alerts.
        
        Args:
            last_n: Number of most recent alerts to return
            
        Returns:
            List of alert dictionaries
        """
        alerts = self.alerts.copy()
        if last_n is not None:
            alerts = alerts[-last_n:]
        return alerts


class DeploymentLogger:
    """Comprehensive logging for deployment monitoring."""
    
    def __init__(self, log_dir: Path):
        """Initialize deployment logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("ConfoRL.Deploy")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"deployment_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics tracking
        self.metrics_file = self.log_dir / "deployment_metrics.jsonl"
        self.metrics_history = []
        
    def log_info(self, message: str) -> None:
        """Log info level message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning level message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error level message."""
        self.logger.error(message)
    
    def log_critical(self, message: str) -> None:
        """Log critical level message."""
        self.logger.critical(message)
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to JSONL file.
        
        Args:
            metrics: Metrics dictionary to log
        """
        # Add timestamp
        metrics["timestamp"] = time.time()
        metrics["time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to metrics file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        self.metrics_history.append(metrics)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save final deployment results.
        
        Args:
            results: Results dictionary to save
        """
        results_file = self.log_dir / "deployment_results.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.log_info(f"Deployment results saved to {results_file}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logged information.
        
        Returns:
            Dict with log summary statistics
        """
        # Count log levels from file
        log_counts = defaultdict(int)
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    with open(handler.baseFilename, "r") as f:
                        for line in f:
                            for level in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
                                if f" - {level} - " in line:
                                    log_counts[level] += 1
                                    break
                except FileNotFoundError:
                    pass
        
        return {
            "log_dir": str(self.log_dir),
            "metrics_logged": len(self.metrics_history),
            "log_counts": dict(log_counts),
            "files_created": len(list(self.log_dir.glob("*")))
        }