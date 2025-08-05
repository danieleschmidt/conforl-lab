"""Command-line interface for ConfoRL."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .utils.logging import setup_logging, get_logger
from .utils.validation import validate_config
from .utils.errors import ConfoRLError, ConfigurationError
from .algorithms import ConformaSAC, ConformaPPO, ConformaTD3, ConformaCQL


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ConfoRL: Adaptive Conformal Risk Control for Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conforl train --algorithm sac --env CartPole-v1 --config config.json
  conforl evaluate --model ./models/agent --env CartPole-v1
  conforl deploy --model ./models/agent --env CartPole-v1 --monitor
        """
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a conformal RL agent")
    train_parser.add_argument(
        "--algorithm", "-a",
        choices=["sac", "ppo", "td3", "cql"],
        required=True,
        help="RL algorithm to use"
    )
    train_parser.add_argument(
        "--env", "-e",
        type=str,
        required=True,
        help="Environment name or path"
    )
    train_parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    train_parser.add_argument(
        "--save-path", "-s",
        type=str,
        default="./models/agent",
        help="Path to save trained model"
    )
    train_parser.add_argument(
        "--target-risk",
        type=float,
        default=0.05,
        help="Target risk level (0-1)"
    )
    train_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (0-1)"
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset path for offline RL (CQL)"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model"
    )
    eval_parser.add_argument(
        "--env", "-e",
        type=str,
        required=True,
        help="Environment name"
    )
    eval_parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy agent in production")
    deploy_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model"
    )
    deploy_parser.add_argument(
        "--env", "-e",
        type=str,
        required=True,
        help="Environment name"
    )
    deploy_parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=100,
        help="Number of deployment episodes"
    )
    deploy_parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable risk monitoring"
    )
    deploy_parser.add_argument(
        "--fallback-policy",
        type=str,
        help="Path to fallback policy"
    )
    
    # Certificate command
    cert_parser = subparsers.add_parser("certificate", help="Generate risk certificate")
    cert_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model"
    )
    cert_parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Desired coverage level"
    )
    
    return parser


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path:
        return {}
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            else:
                raise ConfigurationError(f"Unsupported config format: {config_path}")
        
        return validate_config(config)
    
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config: {e}")


def setup_environment(env_name: str):
    """Setup and validate environment.
    
    Args:
        env_name: Environment name or path
        
    Returns:
        Gymnasium environment
    """
    try:
        import gymnasium as gym
        env = gym.make(env_name)
        return env
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to create environment '{env_name}': {e}")
        raise


def train_command(args, config: Dict[str, Any]):
    """Execute train command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary
    """
    logger = get_logger(__name__)
    logger.info(f"Training {args.algorithm} on {args.env}")
    
    # Setup environment
    env = setup_environment(args.env)
    
    # Algorithm-specific setup
    algorithm_config = {
        'learning_rate': config.get('learning_rate', 3e-4),
        'buffer_size': config.get('buffer_size', 1000000),
        'device': config.get('device', 'auto')
    }
    
    # Risk control configuration
    from .risk.controllers import AdaptiveRiskController
    risk_controller = AdaptiveRiskController(
        target_risk=args.target_risk,
        confidence=args.confidence
    )
    
    # Create agent
    if args.algorithm == "sac":
        agent = ConformaSAC(env=env, risk_controller=risk_controller, **algorithm_config)
    elif args.algorithm == "ppo":
        agent = ConformaPPO(env=env, risk_controller=risk_controller, **algorithm_config)
    elif args.algorithm == "td3":
        agent = ConformaTD3(env=env, risk_controller=risk_controller, **algorithm_config)
    elif args.algorithm == "cql":
        # Load dataset for offline RL
        if not args.dataset:
            raise ConfigurationError("Dataset required for CQL algorithm")
        
        # For now, create empty dataset - in practice would load from file
        dataset = {'observations': [], 'actions': [], 'rewards': []}
        agent = ConformaCQL(
            env=env, 
            dataset=dataset,
            risk_controller=risk_controller, 
            **algorithm_config
        )
    else:
        raise ConfigurationError(f"Unknown algorithm: {args.algorithm}")
    
    # Train agent
    logger.info(f"Starting training for {args.timesteps} timesteps")
    agent.train(
        total_timesteps=args.timesteps,
        eval_freq=10000,
        save_path=args.save_path
    )
    
    logger.info(f"Training completed. Model saved to {args.save_path}")


def evaluate_command(args, config: Dict[str, Any]):
    """Execute evaluate command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary
    """
    logger = get_logger(__name__)
    logger.info(f"Evaluating model {args.model} on {args.env}")
    
    # Setup environment
    env = setup_environment(args.env)
    
    # Load agent (simplified - would load from saved file)
    logger.warning("Model loading not fully implemented - using random agent")
    
    # Run evaluation
    total_reward = 0
    for episode in range(args.episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(1000):  # Max steps per episode
            action = env.action_space.sample()  # Random action for now
            
            if args.render:
                env.render()
            
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_reward += episode_reward
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = total_reward / args.episodes
    logger.info(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")


def deploy_command(args, config: Dict[str, Any]):
    """Execute deploy command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary
    """
    logger = get_logger(__name__)
    logger.info(f"Deploying model {args.model} on {args.env}")
    
    # Setup environment
    env = setup_environment(args.env)
    
    # Load agent (simplified)
    logger.warning("Model loading not fully implemented")
    
    # Setup deployment pipeline
    from .deploy import SafeDeploymentPipeline
    
    # Create dummy agent for demonstration
    from .risk.controllers import AdaptiveRiskController
    risk_controller = AdaptiveRiskController()
    agent = ConformaSAC(env=env, risk_controller=risk_controller)
    
    pipeline = SafeDeploymentPipeline(
        agent=agent,
        risk_monitor=args.monitor,
        log_dir="./deploy_logs"
    )
    
    # Deploy
    results = pipeline.deploy(
        env=env,
        num_episodes=args.episodes
    )
    
    logger.info(f"Deployment completed: {results}")


def certificate_command(args, config: Dict[str, Any]):
    """Execute certificate command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary
    """
    logger = get_logger(__name__)
    logger.info(f"Generating risk certificate for model {args.model}")
    
    # Load agent (simplified)
    logger.warning("Model loading not fully implemented")
    
    # Generate certificate
    logger.info(f"Risk certificate with {args.coverage} coverage would be generated here")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose >= 2 else "INFO" if args.verbose >= 1 else "WARNING"
    setup_logging(level=log_level, log_dir=args.log_dir, include_console=True)
    
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Execute command
        if args.command == "train":
            train_command(args, config)
        elif args.command == "evaluate":
            evaluate_command(args, config)
        elif args.command == "deploy":
            deploy_command(args, config)
        elif args.command == "certificate":
            certificate_command(args, config)
        else:
            parser.print_help()
            sys.exit(1)
    
    except ConfoRLError as e:
        logger.error(f"ConfoRL Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()