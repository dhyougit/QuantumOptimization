"""
Basic Usage Example

Demonstrates how to use the Knowledge-Enhanced Transformer.
"""

import torch
from models.knowledge_transformer import KnowledgeEnhancedTransformer
from knowledge.knowledge_base import KnowledgeBase


def create_sample_knowledge_base():
    """Create a sample knowledge base for demonstration."""
    
    # Create knowledge base
    kb = KnowledgeBase(knowledge_dim=512, retrieval_method='dense')
    
    # Add sample knowledge entries
    knowledge_entries = [
        {
            'text': 'Paris is the capital of France.',
            'embedding': torch.randn(512),  # In practice, use proper embeddings
            'metadata': {'category': 'geography', 'confidence': 0.95}
        },
        {
            'text': 'The Eiffel Tower is located in Paris.',
            'embedding': torch.randn(512),
            'metadata': {'category': 'landmarks', 'confidence': 0.98}
        },
        {
            'text': 'France is a country in Western Europe.',
            'embedding': torch.randn(512),
            'metadata': {'category': 'geography', 'confidence': 0.96}
        },
        {
            'text': 'Python is a high-level programming language.',
            'embedding': torch.randn(512),
            'metadata': {'category': 'technology', 'confidence': 0.99}
        },
        {
            'text': 'The speed of light is approximately 299,792,458 meters per second.',
            'embedding': torch.randn(512),
            'metadata': {'category': 'physics', 'confidence': 1.0}
        }
    ]
    
    for entry in knowledge_entries:
        kb.add_entry(
            text=entry['text'],
            embedding=entry['embedding'],
            metadata=entry['metadata']
        )
    
    # Build index
    kb.build_index()
    
    return kb


def example_basic_inference():
    """Example: Basic inference with knowledge enhancement."""
    
    print("=" * 50)
    print("Example 1: Basic Inference")
    print("=" * 50)
    
    # Create knowledge base
    kb = create_sample_knowledge_base()
    
    # Create model
    model = KnowledgeEnhancedTransformer(
        vocab_size=50000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        knowledge_base=kb
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Sample input (batch_size=2, seq_len=10)
    input_ids = torch.randint(0, 50000, (2, 10))
    
    # Inference without knowledge
    print("\n--- Without Knowledge ---")
    model.eval()
    with torch.no_grad():
        output_no_knowledge = model(input_ids, use_knowledge=False)
    print(f"Output shape: {output_no_knowledge.shape}")
    
    # Inference with knowledge
    print("\n--- With Knowledge ---")
    with torch.no_grad():
        output_with_knowledge = model(input_ids, use_knowledge=True)
    print(f"Output shape: {output_with_knowledge.shape}")
    
    # Compare outputs
    difference = torch.abs(output_with_knowledge - output_no_knowledge).mean()
    print(f"\nMean absolute difference: {difference:.4f}")
    print("(Non-zero difference confirms knowledge is being used)")


def example_knowledge_retrieval():
    """Example: Knowledge retrieval demonstration."""
    
    print("\n" + "=" * 50)
    print("Example 2: Knowledge Retrieval")
    print("=" * 50)
    
    # Create knowledge base
    kb = create_sample_knowledge_base()
    
    # Create a query embedding
    query = torch.randn(512)
    
    print("\nRetrieving relevant knowledge...")
    retrieved_embeddings, indices, scores = kb.retrieve(query, top_k=3)
    
    print(f"\nTop-3 Retrieved Knowledge:")
    print("-" * 50)
    
    for i, (idx, score) in enumerate(zip(indices, scores)):
        entry = kb.knowledge_entries[idx.item()]
        print(f"\n{i+1}. Score: {score:.4f}")
        print(f"   Text: {entry['text']}")
        print(f"   Category: {entry['metadata']['category']}")
        print(f"   Confidence: {entry['metadata']['confidence']}")


def example_text_generation():
    """Example: Text generation with knowledge."""
    
    print("\n" + "=" * 50)
    print("Example 3: Text Generation")
    print("=" * 50)
    
    # Create knowledge base
    kb = create_sample_knowledge_base()
    
    # Create model
    model = KnowledgeEnhancedTransformer(
        vocab_size=50000,
        d_model=512,
        n_heads=8,
        n_layers=4,  # Smaller for faster generation
        d_ff=1024,
        knowledge_base=kb
    )
    
    # Sample starting sequence
    input_ids = torch.randint(0, 50000, (1, 5))
    
    print("\nGenerating text...")
    print("Starting sequence:", input_ids[0].tolist())
    
    # Generate without knowledge
    print("\n--- Without Knowledge ---")
    with torch.no_grad():
        generated_no_k = model.generate(
            input_ids,
            max_length=10,
            temperature=0.8,
            use_knowledge=False
        )
    print("Generated:", generated_no_k[0].tolist())
    
    # Generate with knowledge
    print("\n--- With Knowledge ---")
    with torch.no_grad():
        generated_with_k = model.generate(
            input_ids,
            max_length=10,
            temperature=0.8,
            use_knowledge=True
        )
    print("Generated:", generated_with_k[0].tolist())


def example_attention_visualization():
    """Example: Visualizing attention patterns."""
    
    print("\n" + "=" * 50)
    print("Example 4: Attention Visualization")
    print("=" * 50)
    
    # Create knowledge base
    kb = create_sample_knowledge_base()
    
    # Create model
    model = KnowledgeEnhancedTransformer(
        vocab_size=50000,
        d_model=512,
        n_heads=8,
        n_layers=2,  # Fewer layers for clearer visualization
        knowledge_base=kb
    )
    
    # Sample input
    input_ids = torch.randint(0, 50000, (1, 8))
    
    print("\nComputing attention weights...")
    attention_weights = model.get_attention_weights(input_ids, use_knowledge=True)
    
    print(f"\nNumber of layers: {len(attention_weights)}")
    
    for i, layer_attn in enumerate(attention_weights):
        print(f"\nLayer {i+1}:")
        if 'self_attention' in layer_attn:
            print(f"  Self-attention shape: {layer_attn['self_attention'].shape}")
        if 'knowledge_attention' in layer_attn:
            print(f"  Knowledge attention shape: {layer_attn['knowledge_attention'].shape}")
            
            # Show which knowledge items were attended to most
            avg_attn = layer_attn['knowledge_attention'].mean(dim=1)  # Average over sequence
            top_k_scores, top_k_indices = torch.topk(avg_attn, k=2, dim=-1)
            
            print(f"  Top attended knowledge items:")
            for j, (idx, score) in enumerate(zip(top_k_indices[0], top_k_scores[0])):
                if idx < len(kb.knowledge_entries):
                    text = kb.knowledge_entries[idx.item()]['text']
                    print(f"    {j+1}. [{score:.4f}] {text}")


def example_save_and_load():
    """Example: Saving and loading knowledge base."""
    
    print("\n" + "=" * 50)
    print("Example 5: Save and Load Knowledge Base")
    print("=" * 50)
    
    # Create and save knowledge base
    print("\nCreating knowledge base...")
    kb = create_sample_knowledge_base()
    
    save_path = 'sample_knowledge_base.json'
    print(f"Saving to {save_path}...")
    kb.save(save_path)
    print("Saved successfully!")
    
    # Load knowledge base
    print(f"\nLoading from {save_path}...")
    kb_loaded = KnowledgeBase.from_file(save_path)
    print(f"Loaded {len(kb_loaded.knowledge_entries)} knowledge entries")
    
    # Verify
    print("\nVerifying loaded knowledge:")
    for i, entry in enumerate(kb_loaded.knowledge_entries[:3]):
        print(f"{i+1}. {entry['text']}")


def main():
    """Run all examples."""
    
    print("\n" + "=" * 70)
    print(" Knowledge-Enhanced Transformer - Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_basic_inference()
    example_knowledge_retrieval()
    example_text_generation()
    example_attention_visualization()
    example_save_and_load()
    
    print("\n" + "=" * 70)
    print(" All examples completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
