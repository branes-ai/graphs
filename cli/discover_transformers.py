#!/usr/bin/env python
"""
Transformer Model Discovery Tool
=================================

Tests which HuggingFace transformer models are traceable with Dynamo
and can be profiled with the unified profiler.

This tool discovers models from popular HuggingFace model families:
- BERT family (bert, distilbert, roberta, albert)
- GPT family (gpt2, gpt-neo, opt, bloom)
- T5 family (t5, flan-t5)
- Other encoder models (electra, deberta, xlm-roberta)

Usage:
    python cli/discover_transformers.py
    python cli/discover_transformers.py --verbose
    python cli/discover_transformers.py --test-model bert-base-uncased
"""

import torch
import torch._dynamo
import argparse
import warnings
from typing import Tuple, List


# Popular transformer model families to test
# Format: (model_name, needs_attention_mask)
TRANSFORMER_MODELS = [
    # BERT family (encoder-only, needs attention_mask)
    ('bert-base-uncased', True),
    ('bert-base-cased', True),
    ('bert-large-uncased', True),
    ('distilbert-base-uncased', True),
    ('distilbert-base-cased', True),
    ('roberta-base', True),
    ('roberta-large', True),
    ('albert-base-v2', True),
    ('albert-large-v2', True),

    # GPT family (decoder-only, no attention_mask needed for tracing)
    ('gpt2', False),
    ('gpt2-medium', False),
    ('gpt2-large', False),
    ('distilgpt2', False),
    ('EleutherAI/gpt-neo-125m', False),
    ('EleutherAI/gpt-neo-1.3B', False),
    ('facebook/opt-125m', False),
    ('facebook/opt-350m', False),

    # Other encoder models
    ('google/electra-small-discriminator', True),
    ('google/electra-base-discriminator', True),
    ('microsoft/deberta-v3-small', True),
    ('microsoft/deberta-v3-base', True),
    ('xlm-roberta-base', True),

    # Encoder-decoder (T5 family - more complex)
    # ('t5-small', True),  # Commented out - needs encoder_outputs
    # ('google/flan-t5-small', True),
]


def test_transformer_traceability(
    model_name: str,
    needs_attention_mask: bool = True,
    seq_len: int = 128,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Test if a transformer model is traceable with Dynamo

    Returns:
        (success, method) where method is 'symbolic_trace', 'dynamo', or error message
    """
    try:
        from transformers import AutoModel
    except ImportError:
        return False, "transformers not installed"

    try:
        # Load model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModel.from_pretrained(model_name)
            model.eval()

        # Create inputs
        input_ids = torch.randint(0, 30000, (1, seq_len))

        if needs_attention_mask:
            attention_mask = torch.ones(1, seq_len, dtype=torch.long)
            example_inputs = (input_ids, attention_mask)
        else:
            example_inputs = (input_ids,)

        # Warm-up
        with torch.no_grad():
            try:
                _ = model(*example_inputs)
            except:
                pass  # Some models may fail warm-up but still trace

        # Try symbolic_trace first
        try:
            from torch.fx import symbolic_trace
            traced = symbolic_trace(model)
            if verbose:
                print(f"✓ {model_name:<40} - FX symbolic_trace")
            return True, "symbolic_trace"
        except:
            pass

        # Try Dynamo export
        try:
            if len(example_inputs) == 1:
                def forward_fn(x):
                    return model(x)
            elif len(example_inputs) == 2:
                def forward_fn(x1, x2):
                    return model(x1, x2)
            else:
                def forward_fn(*args):
                    return model(*args)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traced, guards = torch._dynamo.export(forward_fn, *example_inputs)

            if verbose:
                print(f"✓ {model_name:<40} - Dynamo export")
            return True, "dynamo"
        except Exception as e:
            error_msg = str(e)[:60]
            if verbose:
                print(f"✗ {model_name:<40} - {error_msg}")
            return False, error_msg

    except Exception as e:
        error_msg = str(e)[:60]
        if verbose:
            print(f"✗ {model_name:<40} - Load failed: {error_msg}")
        return False, f"Load failed: {error_msg}"


def discover_transformers(verbose: bool = False, seq_len: int = 128):
    """Discover all traceable transformer models"""

    print("=" * 100)
    print("HUGGINGFACE TRANSFORMER MODEL DISCOVERY")
    print("=" * 100)
    print(f"\nTotal models to test: {len(TRANSFORMER_MODELS)}")
    print(f"Sequence length: {seq_len}\n")

    if verbose:
        print("Testing models:\n")

    traceable = []
    failed = []
    methods = {'symbolic_trace': [], 'dynamo': []}

    for model_name, needs_attention_mask in TRANSFORMER_MODELS:
        success, method = test_transformer_traceability(
            model_name,
            needs_attention_mask=needs_attention_mask,
            seq_len=seq_len,
            verbose=verbose
        )

        if success:
            traceable.append((model_name, needs_attention_mask, method))
            methods[method].append(model_name)
        else:
            failed.append((model_name, method))  # method contains error message here

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nTraceable:     {len(traceable)} models")
    print(f"  - FX trace:    {len(methods['symbolic_trace'])} models")
    print(f"  - Dynamo:      {len(methods['dynamo'])} models")
    print(f"Failed:        {len(failed)} models")

    # Show traceable models by family
    print("\n" + "=" * 100)
    print("TRACEABLE MODELS BY FAMILY")
    print("=" * 100)

    families = {}
    for name, needs_mask, method in traceable:
        # Extract family (first part before /)
        if '/' in name:
            family = name.split('/')[0]
        else:
            family = name.split('-')[0]
        families.setdefault(family, []).append((name, needs_mask, method))

    for family, models in sorted(families.items()):
        print(f"\n{family.upper()} ({len(models)}):")
        for name, needs_mask, method in models:
            mask_str = "w/ mask" if needs_mask else "no mask"
            print(f"  {name:<45} [{method:15}] ({mask_str})")

    # Show failed models
    if failed:
        print("\n" + "=" * 100)
        print("FAILED MODELS")
        print("=" * 100)
        for name, error in failed:
            print(f"  {name:<45} - {error}")

    return traceable, failed


def generate_examples(traceable_models: List[Tuple[str, bool, str]]):
    """Generate usage examples for traceable models"""
    print("\n" + "=" * 100)
    print("USAGE EXAMPLES")
    print("=" * 100)
    print("\nYou can profile these models with:\n")

    # Group by family
    families = {}
    for name, needs_mask, method in traceable_models:
        if '/' in name:
            family = name.split('/')[0]
        else:
            family = name.split('-')[0]
        families.setdefault(family, []).append(name)

    for family, models in sorted(families.items()):
        print(f"# {family.capitalize()} models")
        for model in models[:2]:  # Show first 2 examples per family
            print(f"python cli/profile_graph.py --model {model}")
        if len(models) > 2:
            print(f"# ... and {len(models) - 2} more {family} models")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Discover traceable HuggingFace transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick summary
  python cli/discover_transformers.py

  # Verbose output showing each model test
  python cli/discover_transformers.py --verbose

  # Test with longer sequence
  python cli/discover_transformers.py --seq-len 256

  # Generate usage examples
  python cli/discover_transformers.py --generate-examples

  # Test a specific model
  python cli/discover_transformers.py --test-model bert-base-uncased

Note:
  This tool requires: pip install transformers
  Models are downloaded on first use (cached locally)
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for each model')

    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length for testing (default: 128)')

    parser.add_argument('--generate-examples', '-g', action='store_true',
                       help='Generate usage examples')

    parser.add_argument('--test-model', type=str,
                       help='Test a specific model (e.g., bert-base-uncased)')

    args = parser.parse_args()

    if args.test_model:
        # Test single model
        print(f"Testing {args.test_model}...")
        # Assume BERT-style (with attention_mask) by default
        needs_mask = 'gpt' not in args.test_model.lower()
        success, method = test_transformer_traceability(
            args.test_model,
            needs_attention_mask=needs_mask,
            seq_len=args.seq_len,
            verbose=True
        )
        if success:
            print(f"\n✓ {args.test_model} is traceable with {method}!")
            print(f"\nProfile it with:")
            print(f"  python cli/profile_graph.py --model {args.test_model} --seq-len {args.seq_len}")
        else:
            print(f"\n✗ {args.test_model} is NOT traceable")
            print(f"  Error: {method}")
    else:
        # Full discovery
        traceable, failed = discover_transformers(
            verbose=args.verbose,
            seq_len=args.seq_len
        )

        if args.generate_examples:
            generate_examples(traceable)


if __name__ == "__main__":
    main()
