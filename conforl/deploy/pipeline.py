"""Safe deployment pipeline with risk monitoring and fallback policies."""

import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import json

from ..algorithms.base import ConformalRLAgent
from ..core.types import RiskCertificate, TrajectoryData
from .monitor import RiskMonitor, DeploymentLogger


class SafeDeploymentPipeline:
    """Production deployment pipeline with safety guarantees."""
    
    def __init__(
        self,
        agent: ConformalRLAgent,
        fallback_policy: Optional[Callable] = None,
        risk_monitor: bool = True,
        alert_threshold: float = 0.8,
        max_risk_violations: int = 5,
        log_dir: str = "./deploy_logs"
    ):
        """Initialize safe deployment pipeline.
        
        Args:
            agent: Trained conformal RL agent
            fallback_policy: Safe fallback policy function
            risk_monitor: Whether to enable risk monitoring
            alert_threshold: Risk threshold for alerts (fraction of target)
            max_risk_violations: Max violations before fallback activation
            log_dir: Directory for deployment logs
        """
        self.agent = agent
        self.fallback_policy = fallback_policy or self._default_fallback_policy
        self.alert_threshold = alert_threshold
        self.max_risk_violations = max_risk_violations
        
        # Setup monitoring
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        self.risk_monitor = RiskMonitor() if risk_monitor else None
        self.logger = DeploymentLogger(log_path)
        
        # Deployment state
        self.is_deployed = False
        self.fallback_active = False
        self.risk_violations = 0
        self.deployment_start_time = None
        
        # Statistics tracking
        self.episode_count = 0
        self.total_steps = 0
        self.safety_interventions = 0
        
        self.logger.log_info("SafeDeploymentPipeline initialized")
    
    def _default_fallback_policy(self, state, env):
        """Default safe fallback policy (no-op or environment default)."""
        if hasattr(env.action_space, 'sample'):
            # Random action as simple fallback
            return env.action_space.sample() * 0.1  # Conservative scaling
        return 0  # No-op for discrete actions
    
    def deploy(
        self,
        env,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        eval_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Deploy agent in production environment.
        
        Args:
            env: Production environment
            num_episodes: Number of episodes to run
            max_steps_per_episode: Max steps per episode
            eval_callback: Optional evaluation callback
            
        Returns:
            Deployment statistics and results
        """
        if self.is_deployed:
            raise RuntimeError("Pipeline is already deployed")
        
        self.is_deployed = True
        self.deployment_start_time = time.time()
        
        self.logger.log_info(f"Starting deployment: {num_episodes} episodes")
        
        try:
            results = self._run_deployment(
                env, num_episodes, max_steps_per_episode, eval_callback
            )
            
            self.logger.log_info("Deployment completed successfully")
            return results
            
        except Exception as e:
            self.logger.log_error(f"Deployment failed: {str(e)}")
            raise
        finally:
            self.is_deployed = False
    
    def _run_deployment(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int,
        eval_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Run the actual deployment process."""
        episode_returns = []
        episode_risks = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            episode_return, episode_risk, episode_length = self._run_episode(
                env, max_steps_per_episode, episode
            )
            
            episode_returns.append(episode_return)
            episode_risks.append(episode_risk)
            episode_lengths.append(episode_length)
            
            # Check for risk violations
            target_risk = self.agent.risk_controller.target_risk
            if episode_risk > target_risk * self.alert_threshold:
                self._handle_risk_alert(episode, episode_risk, target_risk)
            
            # Evaluation callback
            if eval_callback is not None:
                eval_callback(episode, episode_return, episode_risk)
            
            # Progress logging
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_risk = np.mean(episode_risks[-10:])
                self.logger.log_info(
                    f"Episode {episode + 1}: avg_return={avg_return:.2f}, "
                    f"avg_risk={avg_risk:.4f}, fallback_active={self.fallback_active}"
                )
        
        # Compute final statistics
        deployment_time = time.time() - self.deployment_start_time
        
        results = {
            "num_episodes": num_episodes,
            "total_steps": self.total_steps,
            "deployment_time": deployment_time,
            "avg_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "avg_risk": float(np.mean(episode_risks)),
            "max_risk": float(np.max(episode_risks)),
            "avg_episode_length": float(np.mean(episode_lengths)),
            "safety_interventions": self.safety_interventions,
            "risk_violations": self.risk_violations,
            "fallback_activation_rate": self.safety_interventions / max(1, self.total_steps),
            "episode_returns": episode_returns,
            "episode_risks": episode_risks
        }
        
        # Save results
        self.logger.save_results(results)
        
        return results
    
    def _run_episode(
        self,
        env,
        max_steps: int,
        episode_num: int
    ) -> tuple[float, float, int]:
        """Run a single episode with safety monitoring.
        
        Returns:
            (episode_return, episode_risk, episode_length)
        """
        state, info = env.reset()
        episode_return = 0.0
        step_count = 0
        
        # Track trajectory for risk computation
        trajectory_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        for step in range(max_steps):
            # Get action with risk monitoring
            if self.fallback_active:
                action = self.fallback_policy(state, env)
                self.safety_interventions += 1
            else:
                action, risk_cert = self.agent.predict(
                    state, deterministic=True, return_risk_certificate=True
                )
                
                # Check if we should activate fallback
                if self._should_activate_fallback(risk_cert):
                    self.fallback_active = True
                    action = self.fallback_policy(state, env)
                    self.safety_interventions += 1
                    self.logger.log_warning(
                        f"Fallback activated at episode {episode_num}, step {step}"
                    )
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store trajectory data
            trajectory_data["states"].append(state)
            trajectory_data["actions"].append(action)
            trajectory_data["rewards"].append(reward)
            trajectory_data["dones"].append(done or truncated)
            trajectory_data["infos"].append(info)
            
            episode_return += reward
            step_count += 1
            self.total_steps += 1
            
            # Risk monitoring
            if self.risk_monitor is not None:
                self.risk_monitor.update(state, action, reward, info)
            
            if done or truncated:
                break
            
            state = next_state
        
        # Compute episode risk
        trajectory = TrajectoryData(
            states=np.array(trajectory_data["states"]),
            actions=np.array(trajectory_data["actions"]),
            rewards=np.array(trajectory_data["rewards"]),
            dones=np.array(trajectory_data["dones"]),
            infos=trajectory_data["infos"]
        )
        
        episode_risk = self.agent.risk_measure.compute(trajectory)
        
        # Update agent's risk controller
        self.agent.risk_controller.update(trajectory, self.agent.risk_measure)
        
        self.episode_count += 1
        
        # Reset fallback if risk is acceptable
        if self.fallback_active and episode_risk < self.agent.risk_controller.target_risk * 0.5:
            self.fallback_active = False
            self.logger.log_info(f"Fallback deactivated after episode {episode_num}")
        
        return episode_return, episode_risk, step_count
    
    def _should_activate_fallback(self, risk_cert: RiskCertificate) -> bool:
        """Check if fallback policy should be activated.
        
        Args:
            risk_cert: Current risk certificate
            
        Returns:
            True if fallback should be activated
        """
        target_risk = self.agent.risk_controller.target_risk
        
        # Activate if risk bound exceeds threshold
        if risk_cert.risk_bound > target_risk * (1 + self.alert_threshold):
            return True
        
        # Activate if too many violations occurred
        if self.risk_violations >= self.max_risk_violations:
            return True
        
        return False
    
    def _handle_risk_alert(self, episode: int, risk: float, target_risk: float) -> None:
        """Handle risk alert by logging and updating violation count.
        
        Args:
            episode: Current episode number
            risk: Observed risk level
            target_risk: Target risk level
        """
        self.risk_violations += 1
        
        alert_msg = (
            f"Risk alert at episode {episode}: "
            f"observed_risk={risk:.4f} > {self.alert_threshold * target_risk:.4f} "
            f"(violations: {self.risk_violations}/{self.max_risk_violations})"
        )
        
        self.logger.log_warning(alert_msg)
        
        if self.risk_violations >= self.max_risk_violations:
            self.logger.log_critical(
                f"Max risk violations reached. Activating fallback policy."
            )
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status.
        
        Returns:
            Dict with deployment status information
        """
        current_time = time.time()
        runtime = current_time - self.deployment_start_time if self.deployment_start_time else 0
        
        status = {
            "is_deployed": self.is_deployed,
            "fallback_active": self.fallback_active,
            "runtime_seconds": runtime,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "risk_violations": self.risk_violations,
            "safety_interventions": self.safety_interventions,
            "intervention_rate": self.safety_interventions / max(1, self.total_steps)
        }
        
        if self.is_deployed and self.agent.risk_controller:
            cert = self.agent.risk_controller.get_certificate()
            status.update({
                "current_risk_bound": cert.risk_bound,
                "target_risk": self.agent.risk_controller.target_risk,
                "risk_margin": cert.risk_bound / self.agent.risk_controller.target_risk
            })
        
        return status
    
    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Emergency stop of deployment.
        
        Args:
            reason: Reason for emergency stop
        """
        self.fallback_active = True
        self.is_deployed = False
        
        self.logger.log_critical(f"EMERGENCY STOP: {reason}")
        
        print(f"EMERGENCY STOP ACTIVATED: {reason}")
        print("All actions will now use fallback policy.")