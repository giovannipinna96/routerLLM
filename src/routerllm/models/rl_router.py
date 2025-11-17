"""
Reinforcement Learning-based Router for RouterLLM
Implements TODO #3: RL-based router with multi-objective optimization
Includes advanced carbon tracking and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, namedtuple
from dataclasses import dataclass
import random
from transformers import AutoTokenizer, AutoModel


# Experience replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])


@dataclass
class RLConfig:
    """Configuration for RL Router"""
    # Model parameters
    state_dim: int = 768  # BERT embedding dimension
    action_dim: int = 4   # Number of LLM experts
    hidden_dim: int = 256
    
    # RL hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    tau: float = 0.001  # Soft update parameter
    
    # Multi-objective weights
    accuracy_weight: float = 0.4
    carbon_weight: float = 0.3
    cost_weight: float = 0.2
    latency_weight: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    buffer_size: int = 10000
    update_every: int = 4
    
    # Carbon optimization
    carbon_penalty_factor: float = 0.1  # Penalty for high carbon models
    carbon_budget_per_request: float = 0.001  # kg CO2 budget per request
    enable_carbon_aware_exploration: bool = True


class PolicyNetwork(nn.Module):
    """
    Deep Q-Network for policy learning
    Outputs Q-values for each expert/action
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Dueling DQN architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: State representation [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        x = F.relu(self.fc1(state))
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # Compute value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine to get Q-values (Dueling DQN formula)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class CarbonAwareReplayBuffer:
    """
    Experience replay buffer with carbon-aware prioritization
    Prioritizes experiences that lead to low-carbon solutions
    """
    
    def __init__(self, capacity: int, carbon_priority_alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.carbon_priority_alpha = carbon_priority_alpha
        
    def push(self, experience: Experience):
        """Add experience to buffer with carbon-based priority"""
        # Calculate priority based on carbon efficiency
        carbon_used = experience.info.get('carbon_emissions', 1.0)
        accuracy = experience.info.get('accuracy', 0.0)
        
        # Priority: high accuracy with low carbon gets higher priority
        carbon_efficiency = accuracy / (carbon_used + 1e-6)
        priority = carbon_efficiency ** self.carbon_priority_alpha
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch with carbon-aware prioritization"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.buffer), 
            batch_size, 
            p=probabilities,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]
        
    def __len__(self):
        return len(self.buffer)


class ReinforcementLearningRouter(nn.Module):
    """
    RL-based router for dynamic LLM selection
    Implements TODO #3: Reinforcement Learning-based routing
    Optimizes for accuracy, carbon footprint, cost, and latency
    """
    
    def __init__(
        self,
        config: Optional[RLConfig] = None,
        encoder_model: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RL Router
        
        Args:
            config: RL configuration
            encoder_model: Pre-trained encoder for state representation
            device: Device to run on
            logger: Logger instance
        """
        super(ReinforcementLearningRouter, self).__init__()
        
        self.config = config or RLConfig()
        self.encoder_model_name = encoder_model
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize encoder for state representation
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder.to(self.device)
        self.encoder.eval()  # Freeze encoder
        
        # Initialize Q-networks (Double DQN)
        self.q_network = PolicyNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        self.target_network = PolicyNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Experience replay buffer with carbon awareness
        self.memory = CarbonAwareReplayBuffer(self.config.buffer_size)
        
        # Training state
        self.epsilon = self.config.epsilon_start
        self.steps_done = 0
        self.episodes_done = 0
        
        # Expert metadata with detailed carbon footprint
        self.expert_metadata = {
            0: {
                "name": "llama3_70b",
                "params": 70e9,
                "carbon_per_token": 0.000050,
                "cost_per_token": 0.0002,
                "avg_latency_ms": 150,
                "quality_score": 0.95,
                "energy_consumption_kwh": 0.5
            },
            1: {
                "name": "codellama_34b",
                "params": 34e9,
                "carbon_per_token": 0.000020,
                "cost_per_token": 0.0001,
                "avg_latency_ms": 80,
                "quality_score": 0.85,
                "energy_consumption_kwh": 0.25
            },
            2: {
                "name": "codellama_13b",
                "params": 13e9,
                "carbon_per_token": 0.000008,
                "cost_per_token": 0.00005,
                "avg_latency_ms": 40,
                "quality_score": 0.75,
                "energy_consumption_kwh": 0.1
            },
            3: {
                "name": "deepseek_7b",
                "params": 7e9,
                "carbon_per_token": 0.000003,
                "cost_per_token": 0.00002,
                "avg_latency_ms": 20,
                "quality_score": 0.65,
                "energy_consumption_kwh": 0.05
            }
        }
        
        # Carbon tracking statistics
        self.carbon_stats = {
            "total_emissions": 0.0,
            "emissions_by_expert": {i: 0.0 for i in range(self.config.action_dim)},
            "requests_count": 0,
            "carbon_saved": 0.0,
            "carbon_budget_used": 0.0
        }
        
        self.logger.info(f"ReinforcementLearningRouter initialized on {self.device}")
        self.logger.info(f"Multi-objective weights - Accuracy: {self.config.accuracy_weight:.2f}, "
                        f"Carbon: {self.config.carbon_weight:.2f}, Cost: {self.config.cost_weight:.2f}, "
                        f"Latency: {self.config.latency_weight:.2f}")
        
    def encode_state(self, text: str) -> torch.Tensor:
        """
        Encode text to state representation
        
        Args:
            text: Input text
            
        Returns:
            State tensor [1, state_dim]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token embedding as state
            state = outputs.last_hidden_state[:, 0, :]
            
        return state
    
    def select_action(
        self, 
        state: torch.Tensor, 
        training: bool = False,
        carbon_budget_remaining: Optional[float] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action (expert) using epsilon-greedy with carbon awareness
        
        Args:
            state: State representation
            training: Whether in training mode
            carbon_budget_remaining: Remaining carbon budget for this session
            
        Returns:
            Selected action and metadata
        """
        # Carbon-aware exploration: prefer low-carbon models during exploration
        if training and random.random() < self.epsilon:
            if self.config.enable_carbon_aware_exploration and carbon_budget_remaining is not None:
                # Weighted random selection based on carbon efficiency
                weights = []
                for i in range(self.config.action_dim):
                    carbon = self.expert_metadata[i]["carbon_per_token"]
                    if carbon_budget_remaining > 0:
                        # Prefer low-carbon models when budget is tight
                        weight = (carbon_budget_remaining / carbon) ** 2
                    else:
                        weight = 1.0 / (carbon + 1e-6)
                    weights.append(weight)
                    
                weights = np.array(weights)
                weights = weights / weights.sum()
                action = np.random.choice(self.config.action_dim, p=weights)
            else:
                # Standard random exploration
                action = random.randrange(self.config.action_dim)
        else:
            # Exploitation: choose best action based on Q-values
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # Apply carbon penalty if budget is tight
                if carbon_budget_remaining is not None and carbon_budget_remaining < self.config.carbon_budget_per_request:
                    for i in range(self.config.action_dim):
                        carbon_cost = self.expert_metadata[i]["carbon_per_token"]
                        penalty = self.config.carbon_penalty_factor * (carbon_cost / self.config.carbon_budget_per_request)
                        q_values[0, i] -= penalty
                        
                action = q_values.max(1)[1].item()
        
        # Get metadata for selected action
        expert_meta = self.expert_metadata[action]
        metadata = {
            "action": action,
            "expert_name": expert_meta["name"],
            "expected_carbon": expert_meta["carbon_per_token"],
            "expected_cost": expert_meta["cost_per_token"],
            "expected_latency": expert_meta["avg_latency_ms"],
            "quality_score": expert_meta["quality_score"],
            "epsilon": self.epsilon if training else 0.0
        }
        
        return action, metadata
    
    def compute_reward(
        self,
        action: int,
        accuracy: float,
        carbon_used: float,
        cost_incurred: float,
        latency_ms: float,
        carbon_budget: Optional[float] = None
    ) -> float:
        """
        Compute multi-objective reward
        
        Args:
            action: Selected expert
            accuracy: Task accuracy (0-1)
            carbon_used: Carbon emissions in kg CO2
            cost_incurred: Monetary cost
            latency_ms: Inference latency in milliseconds
            carbon_budget: Carbon budget for the request
            
        Returns:
            Composite reward signal
        """
        # Normalize metrics
        expert_meta = self.expert_metadata[action]
        
        # Accuracy component (0 to 1)
        accuracy_reward = accuracy
        
        # Carbon efficiency component (lower is better)
        expected_carbon = expert_meta["carbon_per_token"]
        carbon_efficiency = 1.0 - min(carbon_used / (expected_carbon * 2), 1.0)
        
        # Add bonus for staying under carbon budget
        carbon_bonus = 0.0
        if carbon_budget and carbon_used < carbon_budget:
            carbon_bonus = 0.2 * (1.0 - carbon_used / carbon_budget)
            
        carbon_reward = carbon_efficiency + carbon_bonus
        
        # Cost efficiency component
        expected_cost = expert_meta["cost_per_token"]
        cost_efficiency = 1.0 - min(cost_incurred / (expected_cost * 2), 1.0)
        
        # Latency component (faster is better)
        expected_latency = expert_meta["avg_latency_ms"]
        latency_efficiency = 1.0 - min(latency_ms / (expected_latency * 2), 1.0)
        
        # Composite reward
        reward = (
            self.config.accuracy_weight * accuracy_reward +
            self.config.carbon_weight * carbon_reward +
            self.config.cost_weight * cost_efficiency +
            self.config.latency_weight * latency_efficiency
        )
        
        # Penalty for exceeding carbon budget
        if carbon_budget and carbon_used > carbon_budget:
            penalty = self.config.carbon_penalty_factor * (carbon_used / carbon_budget - 1.0)
            reward -= penalty
            
        return reward
    
    def update_carbon_stats(self, action: int, carbon_emissions: float):
        """Update carbon tracking statistics"""
        self.carbon_stats["total_emissions"] += carbon_emissions
        self.carbon_stats["emissions_by_expert"][action] += carbon_emissions
        self.carbon_stats["requests_count"] += 1
        
        # Calculate carbon saved compared to always using largest model
        largest_model_carbon = self.expert_metadata[0]["carbon_per_token"]
        carbon_saved = max(0, largest_model_carbon - carbon_emissions)
        self.carbon_stats["carbon_saved"] += carbon_saved
        
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config.batch_size:
            return None
            
        # Sample batch from memory with carbon-aware prioritization
        experiences = self.memory.sample(self.config.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            # Action selection from main network
            next_actions = self.q_network(next_state_batch).max(1)[1].unsqueeze(1)
            # Q-value evaluation from target network
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze()
            
            # Compute targets
            target_q_values = reward_batch + (self.config.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        # Update target network (soft update)
        if self.steps_done % self.config.update_every == 0:
            self.soft_update_target_network()
            
        self.steps_done += 1
        
        return loss.item()
    
    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * local_param.data + 
                (1.0 - self.config.tau) * target_param.data
            )
    
    def forward(
        self,
        text: str,
        training: bool = False,
        carbon_budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Main routing decision with RL policy
        
        Args:
            text: Input text
            training: Whether in training mode
            carbon_budget: Carbon budget for this request
            
        Returns:
            Routing decision with metadata
        """
        # Encode state
        state = self.encode_state(text)
        
        # Calculate remaining carbon budget
        if carbon_budget:
            carbon_budget_remaining = carbon_budget - (
                self.carbon_stats["total_emissions"] / max(self.carbon_stats["requests_count"], 1)
            )
        else:
            carbon_budget_remaining = self.config.carbon_budget_per_request
            
        # Select action
        action, metadata = self.select_action(
            state, 
            training=training,
            carbon_budget_remaining=carbon_budget_remaining
        )
        
        # Get Q-values for analysis
        with torch.no_grad():
            q_values = self.q_network(state).cpu().numpy()[0]
            
        # Prepare response
        result = {
            "selected_expert": action,
            "expert_name": metadata["expert_name"],
            "q_values": q_values.tolist(),
            "confidence": float(np.exp(q_values[action]) / np.exp(q_values).sum()),  # Softmax
            "expected_carbon": metadata["expected_carbon"],
            "expected_cost": metadata["expected_cost"],
            "expected_latency": metadata["expected_latency"],
            "quality_score": metadata["quality_score"],
            "epsilon": metadata["epsilon"],
            "carbon_budget_remaining": carbon_budget_remaining,
            "state": state,  # Keep for training
            "training": training
        }
        
        # Update carbon tracking
        self.update_carbon_stats(action, metadata["expected_carbon"])
        
        return result
    
    def add_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        info: Dict[str, Any]
    ):
        """Add experience to replay buffer"""
        experience = Experience(state, action, reward, next_state, done, info)
        self.memory.push(experience)
        
    def get_carbon_report(self) -> Dict[str, Any]:
        """Generate comprehensive carbon footprint report"""
        report = {
            "total_emissions_kg": self.carbon_stats["total_emissions"],
            "total_requests": self.carbon_stats["requests_count"],
            "average_emissions_per_request": (
                self.carbon_stats["total_emissions"] / max(self.carbon_stats["requests_count"], 1)
            ),
            "carbon_saved_kg": self.carbon_stats["carbon_saved"],
            "emissions_by_expert": self.carbon_stats["emissions_by_expert"],
            "carbon_efficiency_score": (
                self.carbon_stats["carbon_saved"] / 
                (self.carbon_stats["total_emissions"] + self.carbon_stats["carbon_saved"] + 1e-6)
            ),
            "recommendations": self._generate_carbon_recommendations()
        }
        
        return report
    
    def _generate_carbon_recommendations(self) -> List[str]:
        """Generate recommendations for carbon reduction"""
        recommendations = []
        
        # Analyze expert usage
        total_emissions = self.carbon_stats["total_emissions"]
        if total_emissions > 0:
            for expert_id, emissions in self.carbon_stats["emissions_by_expert"].items():
                percentage = (emissions / total_emissions) * 100
                if percentage > 50 and expert_id == 0:  # Largest model
                    recommendations.append(
                        f"Consider using smaller models more frequently. "
                        f"{self.expert_metadata[expert_id]['name']} accounts for {percentage:.1f}% of emissions."
                    )
                    
        # Check carbon efficiency
        avg_emissions = self.carbon_stats["total_emissions"] / max(self.carbon_stats["requests_count"], 1)
        if avg_emissions > self.config.carbon_budget_per_request:
            recommendations.append(
                f"Average emissions ({avg_emissions:.6f} kg) exceed budget "
                f"({self.config.carbon_budget_per_request:.6f} kg). Consider stricter routing policies."
            )
            
        # Suggest optimization strategies
        if self.epsilon > 0.5:
            recommendations.append(
                "Model is still exploring heavily. Continue training for better carbon optimization."
            )
            
        return recommendations
    
    def save_model(self, path: str):
        """Save the RL router model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'carbon_stats': self.carbon_stats,
            'expert_metadata': self.expert_metadata
        }, path)
        self.logger.info(f"RL Router saved to {path}")
        
    def load_model(self, path: str):
        """Load a trained RL router model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.carbon_stats = checkpoint['carbon_stats']
        self.expert_metadata = checkpoint.get('expert_metadata', self.expert_metadata)
        self.logger.info(f"RL Router loaded from {path}")


class RLTrainer:
    """
    Trainer for Reinforcement Learning Router
    Implements online learning with simulated environment
    """
    
    def __init__(
        self,
        router: ReinforcementLearningRouter,
        train_data: List[Tuple[str, int, float]],  # (text, optimal_expert, accuracy)
        val_data: Optional[List[Tuple[str, int, float]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RL Trainer
        
        Args:
            router: RL router instance
            train_data: Training data with optimal expert annotations
            val_data: Validation data
            logger: Logger instance
        """
        self.router = router
        self.train_data = train_data
        self.val_data = val_data
        self.logger = logger or logging.getLogger(__name__)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_carbon = []
        self.validation_scores = []
        
    def simulate_environment_step(
        self,
        text: str,
        action: int,
        optimal_expert: int,
        base_accuracy: float
    ) -> Tuple[float, float, float, float, bool]:
        """
        Simulate environment response to action
        
        Returns:
            accuracy, carbon_used, cost, latency, done
        """
        expert_meta = self.router.expert_metadata[action]
        
        # Simulate accuracy based on expert selection
        if action == optimal_expert:
            accuracy = base_accuracy
        else:
            # Degrade accuracy based on model capability difference
            quality_diff = expert_meta["quality_score"] - self.router.expert_metadata[optimal_expert]["quality_score"]
            accuracy = max(0.0, base_accuracy + quality_diff * 0.3)
            
        # Simulate resource usage
        text_length = len(text.split())
        tokens_generated = text_length * 3  # Rough estimate
        
        carbon_used = expert_meta["carbon_per_token"] * tokens_generated
        cost = expert_meta["cost_per_token"] * tokens_generated
        latency = expert_meta["avg_latency_ms"] * (tokens_generated / 100)
        
        done = True  # Single-step episodes for simplicity
        
        return accuracy, carbon_used, cost, latency, done
    
    def train_episode(self, num_episodes: int = 100):
        """Train for multiple episodes"""
        self.logger.info(f"Starting RL training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_carbon = 0
            losses = []
            
            # Shuffle training data
            episode_data = random.sample(self.train_data, min(100, len(self.train_data)))
            
            for text, optimal_expert, base_accuracy in episode_data:
                # Get current state
                state = self.router.encode_state(text)
                
                # Select action
                result = self.router.forward(text, training=True)
                action = result["selected_expert"]
                
                # Simulate environment step
                accuracy, carbon_used, cost, latency, done = self.simulate_environment_step(
                    text, action, optimal_expert, base_accuracy
                )
                
                # Calculate reward
                reward = self.router.compute_reward(
                    action, accuracy, carbon_used, cost, latency,
                    carbon_budget=self.router.config.carbon_budget_per_request
                )
                
                # Get next state (same as current for single-step)
                next_state = state
                
                # Store experience
                info = {
                    "accuracy": accuracy,
                    "carbon_emissions": carbon_used,
                    "cost": cost,
                    "latency": latency
                }
                self.router.add_experience(state, action, reward, next_state, done, info)
                
                # Train
                loss = self.router.train_step()
                if loss is not None:
                    losses.append(loss)
                    
                episode_reward += reward
                episode_carbon += carbon_used
                
            # Log episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_carbon.append(episode_carbon)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_carbon = np.mean(self.episode_carbon[-10:])
                avg_loss = np.mean(losses) if losses else 0
                
                self.logger.info(
                    f"Episode {episode}: Avg Reward: {avg_reward:.4f}, "
                    f"Avg Carbon: {avg_carbon:.6f} kg, Loss: {avg_loss:.4f}, "
                    f"Epsilon: {self.router.epsilon:.4f}"
                )
                
            # Validation
            if episode % 20 == 0 and self.val_data:
                val_score = self.validate()
                self.validation_scores.append(val_score)
                self.logger.info(f"Validation Score: {val_score:.4f}")
                
    def validate(self) -> float:
        """Validate the current policy"""
        if not self.val_data:
            return 0.0
            
        total_reward = 0
        
        for text, optimal_expert, base_accuracy in self.val_data:
            # Get action without exploration
            result = self.router.forward(text, training=False)
            action = result["selected_expert"]
            
            # Simulate step
            accuracy, carbon_used, cost, latency, _ = self.simulate_environment_step(
                text, action, optimal_expert, base_accuracy
            )
            
            # Calculate reward
            reward = self.router.compute_reward(
                action, accuracy, carbon_used, cost, latency
            )
            
            total_reward += reward
            
        return total_reward / len(self.val_data)
