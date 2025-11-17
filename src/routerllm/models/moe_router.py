"""
Dynamic Gating Network Router for Mixture-of-Experts (MoE) Architecture
Implements TODO #1: Dynamic routing with learned gating network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel
from abc import ABC, abstractmethod


class GatingNetwork(nn.Module):
    """
    Gating network for dynamic expert selection in MoE architecture
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # BERT hidden size
        num_experts: int = 4,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
        use_sparse_gating: bool = True,
        top_k: int = 2
    ):
        """
        Initialize Gating Network
        
        Args:
            input_dim: Input dimension (embedding size)
            num_experts: Number of available experts (LLMs)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate
            temperature: Temperature for softmax
            use_sparse_gating: Whether to use sparse gating (select top-k)
            top_k: Number of experts to select
        """
        super(GatingNetwork, self).__init__()
        
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_sparse_gating = use_sparse_gating
        self.top_k = min(top_k, num_experts)
        
        # Gating layers
        self.gate_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Noise layer for exploration during training
        self.noise_layer = nn.Linear(input_dim, num_experts)
        
        # Load balancing auxiliary loss
        self.load_balance_loss = LoadBalancingLoss()
        
    def forward(self, x: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through gating network
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            gates: Gating weights [batch_size, num_experts]
            selected_experts: Indices of selected experts [batch_size, top_k]
        """
        # Compute gating logits
        gate_logits = self.gate_layers(x)
        
        # Add noise during training for exploration
        if training:
            noise = self.noise_layer(x)
            gate_logits = gate_logits + torch.randn_like(noise) * 0.1
            
        # Apply temperature scaling
        gate_logits = gate_logits / self.temperature
        
        if self.use_sparse_gating:
            # Sparse gating: select top-k experts
            gates, selected_experts = self._sparse_gating(gate_logits)
        else:
            # Dense gating: use all experts with softmax weights
            gates = F.softmax(gate_logits, dim=-1)
            selected_experts = torch.arange(self.num_experts).unsqueeze(0).expand(x.size(0), -1)
            
        return gates, selected_experts
    
    def _sparse_gating(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement sparse gating to select top-k experts
        
        Args:
            logits: Gating logits [batch_size, num_experts]
            
        Returns:
            gates: Sparse gating weights [batch_size, num_experts]
            selected_experts: Indices of selected experts [batch_size, top_k]
        """
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Create sparse gates
        gates = torch.zeros_like(logits)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Scatter the top-k weights back to full gate tensor
        gates = gates.scatter(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices
    
    def get_load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage
        
        Args:
            gates: Gating weights [batch_size, num_experts]
            
        Returns:
            Load balancing loss scalar
        """
        return self.load_balance_loss(gates)


class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss to encourage uniform distribution of experts
    """
    
    def __init__(self, eps: float = 1e-10):
        super(LoadBalancingLoss, self).__init__()
        self.eps = eps
        
    def forward(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss
        
        Args:
            gates: Gating weights [batch_size, num_experts]
            
        Returns:
            Loss scalar
        """
        # Compute the mean importance of each expert
        importance = gates.sum(dim=0)  # [num_experts]
        
        # Normalize by batch size
        batch_size = gates.size(0)
        importance = importance / (batch_size + self.eps)
        
        # Compute squared coefficient of variation
        mean_importance = importance.mean()
        variance = ((importance - mean_importance) ** 2).mean()
        cv_squared = variance / (mean_importance ** 2 + self.eps)
        
        return cv_squared


class DynamicMoERouter(nn.Module):
    """
    Dynamic Mixture-of-Experts Router with learned gating
    Addresses TODO #1: Implement Dynamic Router with Gating Network
    """
    
    def __init__(
        self,
        encoder_model: str = "bert-base-uncased",
        num_experts: int = 4,
        hidden_dim: int = 256,
        top_k: int = 2,
        temperature: float = 1.0,
        device: Optional[str] = None,
        carbon_aware: bool = True,
        cost_aware: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Dynamic MoE Router
        
        Args:
            encoder_model: Pre-trained encoder model name
            num_experts: Number of expert LLMs
            hidden_dim: Hidden dimension for gating network
            top_k: Number of experts to select per input
            temperature: Temperature for gating softmax
            device: Device to run on
            carbon_aware: Whether to consider carbon emissions in routing
            cost_aware: Whether to consider cost in routing  
            logger: Logger instance
        """
        super(DynamicMoERouter, self).__init__()
        
        self.encoder_model_name = encoder_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.carbon_aware = carbon_aware
        self.cost_aware = cost_aware
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize encoder
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder.to(self.device)
        
        # Initialize gating network
        encoder_dim = self.encoder.config.hidden_size
        self.gating_network = GatingNetwork(
            input_dim=encoder_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            temperature=temperature,
            use_sparse_gating=True,
            top_k=top_k
        )
        self.gating_network.to(self.device)
        
        # Expert metadata (carbon emissions, cost, performance)
        self.expert_metadata = {
            0: {"name": "llama3_70b", "carbon_per_token": 0.000050, "cost_per_token": 0.0002, "quality_score": 0.95},
            1: {"name": "codellama_34b", "carbon_per_token": 0.000020, "cost_per_token": 0.0001, "quality_score": 0.85},
            2: {"name": "codellama_13b", "carbon_per_token": 0.000008, "cost_per_token": 0.00005, "quality_score": 0.75},
            3: {"name": "deepseek_7b", "carbon_per_token": 0.000003, "cost_per_token": 0.00002, "quality_score": 0.65}
        }
        
        # Multi-objective weights
        self.quality_weight = 0.5
        self.carbon_weight = 0.3 if carbon_aware else 0.0
        self.cost_weight = 0.2 if cost_aware else 0.0
        
        # Normalize weights
        total_weight = self.quality_weight + self.carbon_weight + self.cost_weight
        self.quality_weight /= total_weight
        self.carbon_weight /= total_weight
        self.cost_weight /= total_weight
        
        self.logger.info(f"DynamicMoERouter initialized with {num_experts} experts, top_k={top_k}")
        self.logger.info(f"Weights - Quality: {self.quality_weight:.2f}, Carbon: {self.carbon_weight:.2f}, Cost: {self.cost_weight:.2f}")
        
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to embeddings using pre-trained encoder
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor [1, hidden_dim]
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
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
            
        return embedding
    
    def forward(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Route text to appropriate experts
        
        Args:
            text: Input text
            return_all_scores: Whether to return scores for all experts
            
        Returns:
            Routing decision with selected experts and weights
        """
        # Encode text
        embedding = self.encode_text(text)
        
        # Get gating weights
        gates, selected_experts = self.gating_network(embedding, training=self.training)
        
        # Compute multi-objective scores if needed
        if self.carbon_aware or self.cost_aware:
            adjusted_gates = self._adjust_gates_for_objectives(gates, text)
        else:
            adjusted_gates = gates
            
        # Get top expert
        top_expert_idx = adjusted_gates.argmax(dim=-1).item()
        top_expert_weight = adjusted_gates[0, top_expert_idx].item()
        
        # Prepare response
        result = {
            "selected_expert": top_expert_idx,
            "expert_name": self.expert_metadata[top_expert_idx]["name"],
            "confidence": top_expert_weight,
            "gates": gates.detach().cpu().numpy()[0].tolist(),
            "adjusted_gates": adjusted_gates.detach().cpu().numpy()[0].tolist()
        }
        
        if return_all_scores:
            # Return detailed scores for all experts
            all_scores = {}
            for idx in range(self.num_experts):
                expert_meta = self.expert_metadata[idx]
                all_scores[expert_meta["name"]] = {
                    "gate_weight": gates[0, idx].item(),
                    "adjusted_weight": adjusted_gates[0, idx].item(),
                    "quality_score": expert_meta["quality_score"],
                    "carbon_per_token": expert_meta["carbon_per_token"],
                    "cost_per_token": expert_meta["cost_per_token"]
                }
            result["all_expert_scores"] = all_scores
            
        # Estimate environmental impact
        selected_meta = self.expert_metadata[top_expert_idx]
        estimated_tokens = len(text.split()) * 3  # Rough estimate
        result["estimated_carbon"] = selected_meta["carbon_per_token"] * estimated_tokens
        result["estimated_cost"] = selected_meta["cost_per_token"] * estimated_tokens
        
        return result
    
    def _adjust_gates_for_objectives(self, gates: torch.Tensor, text: str) -> torch.Tensor:
        """
        Adjust gating weights based on multiple objectives (quality, carbon, cost)
        
        Args:
            gates: Original gating weights [batch_size, num_experts]
            text: Input text (for estimating token count)
            
        Returns:
            Adjusted gates considering all objectives
        """
        batch_size = gates.size(0)
        adjusted_gates = gates.clone()
        
        # Estimate token count
        estimated_tokens = len(text.split()) * 3
        
        for i in range(self.num_experts):
            meta = self.expert_metadata[i]
            
            # Compute multi-objective score
            quality_score = meta["quality_score"]
            
            # Normalize carbon and cost (inverse because lower is better)
            carbon_score = 1.0 - (meta["carbon_per_token"] * estimated_tokens / 0.01)  # Normalize by 0.01 kg
            cost_score = 1.0 - (meta["cost_per_token"] * estimated_tokens / 0.1)  # Normalize by $0.1
            
            # Clamp scores to [0, 1]
            carbon_score = max(0.0, min(1.0, carbon_score))
            cost_score = max(0.0, min(1.0, cost_score))
            
            # Compute weighted objective
            objective_multiplier = (
                self.quality_weight * quality_score +
                self.carbon_weight * carbon_score +
                self.cost_weight * cost_score
            )
            
            # Adjust gate weight
            adjusted_gates[:, i] *= objective_multiplier
            
        # Re-normalize gates
        adjusted_gates = adjusted_gates / (adjusted_gates.sum(dim=-1, keepdim=True) + 1e-10)
        
        return adjusted_gates
    
    def train_gating_network(
        self,
        train_data: List[Tuple[str, int, float]],
        val_data: Optional[List[Tuple[str, int, float]]] = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ):
        """
        Train the gating network using supervised learning + load balancing
        
        Args:
            train_data: List of (text, expert_idx, reward) tuples
            val_data: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        self.logger.info(f"Training gating network on {len(train_data)} examples")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.gating_network.train()
            total_loss = 0.0
            
            # Shuffle data
            np.random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Prepare batch
                texts = [item[0] for item in batch]
                expert_labels = torch.tensor([item[1] for item in batch]).to(self.device)
                rewards = torch.tensor([item[2] for item in batch]).to(self.device)
                
                # Encode texts
                embeddings = []
                for text in texts:
                    embeddings.append(self.encode_text(text))
                embeddings = torch.cat(embeddings, dim=0)
                
                # Forward pass
                gates, _ = self.gating_network(embeddings, training=True)
                
                # Compute supervised loss (cross-entropy with soft labels)
                ce_loss = F.cross_entropy(gates, expert_labels)
                
                # Compute load balancing loss
                lb_loss = self.gating_network.get_load_balance_loss(gates)
                
                # Total loss
                loss = ce_loss + 0.01 * lb_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / (len(train_data) // batch_size)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            if val_data:
                val_loss = self._validate(val_data, batch_size)
                self.logger.info(f"Validation Loss: {val_loss:.4f}")
                
    def _validate(self, val_data: List[Tuple[str, int, float]], batch_size: int) -> float:
        """Validate the gating network"""
        self.gating_network.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                
                texts = [item[0] for item in batch]
                expert_labels = torch.tensor([item[1] for item in batch]).to(self.device)
                
                embeddings = []
                for text in texts:
                    embeddings.append(self.encode_text(text))
                embeddings = torch.cat(embeddings, dim=0)
                
                gates, _ = self.gating_network(embeddings, training=False)
                loss = F.cross_entropy(gates, expert_labels)
                
                total_loss += loss.item()
                
        return total_loss / (len(val_data) // batch_size)
    
    def save_model(self, path: str):
        """Save the trained gating network"""
        torch.save({
            'gating_network_state_dict': self.gating_network.state_dict(),
            'encoder_model_name': self.encoder_model_name,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'expert_metadata': self.expert_metadata,
            'weights': {
                'quality': self.quality_weight,
                'carbon': self.carbon_weight,
                'cost': self.cost_weight
            }
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load a trained gating network"""
        checkpoint = torch.load(path, map_location=self.device)
        self.gating_network.load_state_dict(checkpoint['gating_network_state_dict'])
        self.expert_metadata = checkpoint['expert_metadata']
        
        weights = checkpoint['weights']
        self.quality_weight = weights['quality']
        self.carbon_weight = weights['carbon']
        self.cost_weight = weights['cost']
        
        self.logger.info(f"Model loaded from {path}")
        
    def get_model_name_from_class(self, class_id: int) -> str:
        """Get model name from expert class ID"""
        return self.expert_metadata[class_id]["name"]
