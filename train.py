"""
Training Script for Knowledge-Enhanced Transformer

Complete training pipeline with knowledge augmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import os
from tqdm import tqdm
import wandb

from models.knowledge_transformer import KnowledgeEnhancedTransformer
from knowledge.knowledge_base import KnowledgeBase
from training.dataset import KnowledgeAugmentedDataset
from training.loss import KnowledgeAugmentedLoss
from utils.metrics import compute_perplexity, compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train Knowledge-Enhanced Transformer')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--knowledge_path', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Knowledge parameters
    parser.add_argument('--use_knowledge', action='store_true', default=True)
    parser.add_argument('--knowledge_weight', type=float, default=0.5)
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()


class Trainer:
    """Trainer class for knowledge-enhanced transformer."""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 criterion, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_knowledge_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            knowledge_embeddings = batch.get('knowledge_embeddings')
            
            if knowledge_embeddings is not None:
                knowledge_embeddings = knowledge_embeddings.to(self.device)
            
            # Forward pass
            logits = self.model(
                input_ids,
                knowledge_embeddings=knowledge_embeddings,
                use_knowledge=self.args.use_knowledge
            )
            
            # Compute loss
            loss_dict = self.criterion(
                logits,
                target_ids,
                knowledge_embeddings=knowledge_embeddings,
                model=self.model
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.gradient_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            total_lm_loss += loss_dict['lm_loss'].item()
            if 'knowledge_loss' in loss_dict:
                total_knowledge_loss += loss_dict['knowledge_loss'].item()
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lm_loss': loss_dict['lm_loss'].item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Log to wandb
            if self.args.use_wandb and batch_idx % self.args.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lm_loss': loss_dict['lm_loss'].item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_lm_loss = total_lm_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'lm_loss': avg_lm_loss,
            'knowledge_loss': total_knowledge_loss / len(self.train_loader)
        }
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                knowledge_embeddings = batch.get('knowledge_embeddings')
                
                if knowledge_embeddings is not None:
                    knowledge_embeddings = knowledge_embeddings.to(self.device)
                
                # Forward pass
                logits = self.model(
                    input_ids,
                    knowledge_embeddings=knowledge_embeddings,
                    use_knowledge=self.args.use_knowledge
                )
                
                # Compute loss
                loss_dict = self.criterion(
                    logits,
                    target_ids,
                    knowledge_embeddings=knowledge_embeddings,
                    model=self.model
                )
                
                total_loss += loss_dict['total_loss'].item()
                
                # Compute accuracy
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == target_ids).sum().item()
                total_tokens += target_ids.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity.item()
        }
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
            'args': vars(self.args)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Validation Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Perplexity: {val_metrics['perplexity']:.2f}")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/perplexity': val_metrics['perplexity']
                })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics['loss'], is_best)
        
        print("Training completed!")


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project='knowledge-enhanced-transformer', config=vars(args))
    
    # Load knowledge base
    print("Loading knowledge base...")
    knowledge_base = KnowledgeBase.from_file(args.knowledge_path)
    
    # Create model
    print("Creating model...")
    model = KnowledgeEnhancedTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        knowledge_base=knowledge_base
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = KnowledgeAugmentedDataset(
        args.data_path,
        knowledge_base,
        split='train',
        val_split=args.val_split
    )
    val_dataset = KnowledgeAugmentedDataset(
        args.data_path,
        knowledge_base,
        split='val',
        val_split=args.val_split
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Warmup scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return max(0.1, (args.warmup_steps / step) ** 0.5)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create loss function
    criterion = KnowledgeAugmentedLoss(
        knowledge_weight=args.knowledge_weight,
        use_knowledge=args.use_knowledge
    )
    
    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        args.device, args
    )
    
    # Train
    trainer.train()
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
