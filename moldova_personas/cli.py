"""
Command-line interface for generating Moldova synthetic personas.

Usage:
    python -m moldova_personas generate --count 1000 --output ./output
    python -m moldova_personas validate --input ./output/moldova_personas.parquet
    python -m moldova_personas stats --input ./output/moldova_personas.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from .generator import PersonaGenerator
from .validators import ValidationPipeline
from .exporters import export_all_formats, StatisticsExporter
from .census_data import CENSUS
from .checkpoint import (
    CheckpointManager,
    StreamingGenerator,
    get_checkpoint_summary,
)


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )


logger = logging.getLogger(__name__)


def cmd_generate(args):
    """Generate personas command with checkpointing support."""
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Generating {args.count} synthetic Moldovan personas...")
    logger.info(f"Random seed: {args.seed or 'random'}")
    logger.info(f"Using IPF adjustment: {args.ipf}")
    logger.info(f"LLM provider: {args.llm_provider}")
    
    # Setup checkpointing if enabled
    checkpoint_manager = None
    if args.checkpoint_every > 0:
        checkpoint_dir = args.checkpoint_dir or args.output
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            basename=args.basename
        )
        logger.info(f"Checkpointing enabled: every {args.checkpoint_every} personas")
        if args.resume_from:
            logger.info(f"Resuming from: {args.resume_from}")
    
    # Initialize generator
    generator = PersonaGenerator(seed=args.seed)
    
    # Generate structured data with streaming
    start_time = datetime.now()
    
    if args.ipf:
        # IPF doesn't support streaming yet
        personas = generator.generate_with_ipf_adjustment(args.count)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(personas)} structured personas in {duration:.2f} seconds")
    else:
        # Use streaming generation with checkpointing
        streaming_gen = StreamingGenerator(
            checkpoint_manager=checkpoint_manager,
            checkpoint_every=args.checkpoint_every if args.checkpoint_every > 0 else args.count + 1,
        )
        
        personas = []
        for persona in streaming_gen.generate_streaming(
            generator=generator,
            n=args.count,
            seed=args.seed,
            resume_from=args.resume_from,
        ):
            personas.append(persona)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(personas)} structured personas in {duration:.2f} seconds")
    
    # Generate narratives if requested
    if args.llm_provider != "mock" or args.narratives:
        logger.info("Generating narrative content with LLM...")
        
        llm_kwargs = {}
        
        # OpenAI specific
        if args.llm_provider == 'openai':
            if args.openai_api_key:
                llm_kwargs['api_key'] = args.openai_api_key
            if args.openai_model:
                llm_kwargs['model'] = args.openai_model
        
        # Qwen/DashScope specific
        elif args.llm_provider in ('qwen', 'dashscope'):
            if args.dashscope_api_key:
                llm_kwargs['api_key'] = args.dashscope_api_key
            if args.qwen_model:
                llm_kwargs['model'] = args.qwen_model
        
        # Qwen local specific
        elif args.llm_provider == 'qwen-local':
            llm_kwargs['model_name'] = args.qwen_local_model
        
        # Use parallel generation if workers > 1
        if args.llm_workers > 1:
            from .async_narrative_generator import AsyncNarrativeGenerator
            logger.info(f"Using parallel generation with {args.llm_workers} workers")
            
            nar_gen = AsyncNarrativeGenerator(
                provider=args.llm_provider,
                max_workers=args.llm_workers,
                rate_limit_per_second=args.llm_rate_limit,
                **llm_kwargs
            )
            
            nar_start = datetime.now()
            personas = nar_gen.generate_batch(personas, show_progress=True)
            nar_duration = (datetime.now() - nar_start).total_seconds()
            
            # Log stats
            stats = nar_gen.get_stats()
            logger.info(f"Generated narratives in {nar_duration:.2f} seconds")
            logger.info(f"Success rate: {stats['success_rate']:.1%}")
            logger.info(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
        else:
            from .narrative_generator import NarrativeGenerator
            
            nar_gen = NarrativeGenerator(
                provider=args.llm_provider,
                **llm_kwargs
            )
            
            nar_start = datetime.now()
            personas = nar_gen.generate_batch(
                personas, 
                show_progress=True,
                delay=args.llm_delay
            )
            nar_duration = (datetime.now() - nar_start).total_seconds()
            logger.info(f"Generated narratives in {nar_duration:.2f} seconds")
    
    # Validate
    logger.info("Validating generated personas...")
    validator = ValidationPipeline()
    report = validator.validate(personas)
    print(report.summary())  # Keep this print for user-facing output
    
    # Export
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting to {output_dir}...")
    results = export_all_formats(personas, str(output_dir), basename=args.basename)
    
    logger.info("Generation complete!")
    return 0


def cmd_validate(args):
    """Validate existing personas command."""
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Validating personas from {args.input}...")
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {args.input}")
        return 1
    
    try:
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix in ['.json', '.jsonl']:
            df = pd.read_json(input_path, lines=(input_path.suffix == '.jsonl'))
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return 1
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return 1
    
    logger.info(f"Loaded {len(df)} personas")
    
    # Convert to Persona objects
    from .models import Persona
    try:
        personas = [Persona(**row) for row in df.to_dict('records')]
    except Exception as e:
        logger.error(f"Failed to convert data to Persona objects: {e}")
        return 1
    
    # Validate
    validator = ValidationPipeline()
    report = validator.validate(personas, tolerance=args.tolerance)
    
    print("\n" + report.summary())  # Keep for user-facing output
    
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(report.summary())
            logger.info(f"Report saved to {args.output}")
        except IOError as e:
            logger.error(f"Failed to write report: {e}")
    
    return 0 if report.is_valid else 1


def cmd_stats(args):
    """Generate statistics command."""
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Generating statistics for {args.input}...")
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {args.input}")
        return 1
    
    try:
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        else:
            df = pd.read_json(input_path, lines=(input_path.suffix == '.jsonl'))
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return 1
    
    from .models import Persona
    try:
        personas = [Persona(**row) for row in df.to_dict('records')]
    except Exception as e:
        logger.error(f"Failed to convert data: {e}")
        return 1
    
    # Generate stats
    exporter = StatisticsExporter()
    stats = exporter.generate(personas)
    
    # Export
    output_path = Path(args.output) if args.output else input_path.parent / "stats.md"
    exporter.export_markdown(stats, str(output_path), census_data=CENSUS)
    
    logger.info(f"Statistics saved to {output_path}")
    return 0


def cmd_example(args):
    """Generate example personas for inspection."""
    generator = PersonaGenerator(seed=42)
    personas = generator.generate(5, show_progress=False)
    
    for i, p in enumerate(personas, 1):
        print(f"{'='*60}")
        print(f"Persona #{i}")
        print(f"{'='*60}")
        print(f"UUID:        {p.uuid}")
        print(f"Name:        {p.name}")
        print(f"Sex:         {p.sex}")
        print(f"Age:         {p.age} ({p.age_group})")
        print(f"Ethnicity:   {p.ethnicity}")
        print(f"Language:    {p.mother_tongue}")
        print(f"Religion:    {p.religion}")
        print(f"Education:   {p.education_level}")
        if p.field_of_study:
            print(f"Field:       {p.field_of_study}")
        print(f"Occupation:  {p.occupation}")
        print(f"Location:    {p.city}, {p.district} ({p.region}, {p.residence_type})")
        print(f"Marital:     {p.marital_status}")
        print()
    
    return 0


def cmd_checkpoint(args):
    """Checkpoint management command."""
    if not args.ckpt_command:
        print("Usage: checkpoint {list|info|export}")
        return 1
    
    if args.ckpt_command == 'list':
        return cmd_checkpoint_list(args)
    elif args.ckpt_command == 'info':
        return cmd_checkpoint_info(args)
    elif args.ckpt_command == 'export':
        return cmd_checkpoint_export(args)
    
    return 0


def cmd_checkpoint_list(args):
    """List available checkpoints."""
    from .checkpoint import CheckpointManager
    
    manager = CheckpointManager(checkpoint_dir=args.dir)
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print(f"No checkpoints found in {args.dir}")
        return 0
    
    print(f"Checkpoints in {args.dir}:")
    print("-" * 80)
    
    for ckpt_path in checkpoints:
        summary = get_checkpoint_summary(str(ckpt_path))
        if "error" in summary:
            print(f"  {ckpt_path.name} - ERROR: {summary['error']}")
        else:
            print(f"  {ckpt_path.name}")
            print(f"    Progress: {summary['completed_count']:,} / {summary['total_target']:,} ({summary['progress_pct']:.1f}%)")
            print(f"    Remaining: {summary['remaining']:,}")
            print(f"    Timestamp: {summary['timestamp']}")
            if summary['seed']:
                print(f"    Seed: {summary['seed']}")
        print()
    
    return 0


def cmd_checkpoint_info(args):
    """Show detailed checkpoint information."""
    summary = get_checkpoint_summary(args.path)
    
    if "error" in summary:
        print(f"Error reading checkpoint: {summary['error']}")
        return 1
    
    print("Checkpoint Information")
    print("=" * 60)
    print(f"Path: {summary['path']}")
    print(f"Version: {summary['version']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Seed: {summary['seed'] or 'Not set'}")
    print()
    print("Progress")
    print("-" * 60)
    print(f"Target: {summary['total_target']:,} personas")
    print(f"Completed: {summary['completed_count']:,} personas")
    print(f"Remaining: {summary['remaining']:,} personas")
    print(f"Progress: {summary['progress_pct']:.1f}%")
    
    return 0


def cmd_checkpoint_export(args):
    """Export checkpoint to Parquet."""
    from .checkpoint import export_checkpoint_to_parquet
    
    try:
        export_checkpoint_to_parquet(args.path, args.output)
        print(f"Exported checkpoint to {args.output}")
        return 0
    except Exception as e:
        print(f"Export failed: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Moldovan personas based on 2024 Census data"
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate personas')
    gen_parser.add_argument('-n', '--count', type=int, default=1000,
                          help='Number of personas to generate (default: 1000)')
    gen_parser.add_argument('-o', '--output', type=str, default='./output',
                          help='Output directory (default: ./output)')
    gen_parser.add_argument('--basename', type=str, default='moldova_personas',
                          help='Base filename (default: moldova_personas)')
    gen_parser.add_argument('--seed', type=int, default=None,
                          help='Random seed for reproducibility')
    gen_parser.add_argument('--ipf', action='store_true',
                          help='Use IPF adjustment for better distribution matching')
    gen_parser.add_argument('--narratives', action='store_true',
                          help='Generate narrative content (even with mock provider)')
    gen_parser.add_argument('--llm-provider', type=str, default='mock',
                          choices=['mock', 'openai', 'qwen', 'qwen-local', 'dashscope', 'local'],
                          help='LLM provider for narratives (default: mock)')
    gen_parser.add_argument('--openai-api-key', type=str, default=None,
                          help='OpenAI API key (or set OPENAI_API_KEY env var)')
    gen_parser.add_argument('--openai-model', type=str, default='gpt-3.5-turbo',
                          help='OpenAI model to use (default: gpt-3.5-turbo)')
    gen_parser.add_argument('--dashscope-api-key', type=str, default=None,
                          help='DashScope API key for Qwen (or set DASHSCOPE_API_KEY env var)')
    gen_parser.add_argument('--qwen-model', type=str, default='qwen-turbo',
                          help='Qwen model: qwen-turbo, qwen-plus, qwen-max, qwen2.5-7b, etc.')
    gen_parser.add_argument('--qwen-local-model', type=str, default='qwen2.5-7b',
                          help='Local Qwen model: qwen2.5-7b, qwen2.5-14b, etc.')
    gen_parser.add_argument('--llm-delay', type=float, default=0.1,
                          help='Delay between LLM API calls in seconds (default: 0.1)')
    gen_parser.add_argument('--llm-workers', type=int, default=1,
                          help='Number of parallel LLM workers (default: 1, use 5-20 for speedup)')
    gen_parser.add_argument('--llm-rate-limit', type=float, default=None,
                          help='Rate limit in requests per second (optional)')
    gen_parser.add_argument('--checkpoint-every', type=int, default=1000,
                          help='Save checkpoint every N personas (default: 1000, set 0 to disable)')
    gen_parser.add_argument('--checkpoint-dir', type=str, default=None,
                          help='Directory for checkpoint files (default: same as output)')
    gen_parser.add_argument('--resume-from', type=str, default=None,
                          help='Resume from checkpoint file')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate existing personas')
    val_parser.add_argument('-i', '--input', type=str, required=True,
                          help='Input file (parquet, csv, json, jsonl)')
    val_parser.add_argument('-o', '--output', type=str, default=None,
                          help='Output file for validation report')
    val_parser.add_argument('-t', '--tolerance', type=float, default=0.02,
                          help='Statistical tolerance (default: 0.02 = 2%%)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate statistics report')
    stats_parser.add_argument('-i', '--input', type=str, required=True,
                            help='Input file')
    stats_parser.add_argument('-o', '--output', type=str, default=None,
                            help='Output file for statistics report')
    
    # Example command
    example_parser = subparsers.add_parser('example', help='Show example personas')
    
    # Checkpoint command
    ckpt_parser = subparsers.add_parser('checkpoint', help='Manage checkpoints')
    ckpt_subparsers = ckpt_parser.add_subparsers(dest='ckpt_command', help='Checkpoint commands')
    
    # List checkpoints
    ckpt_list = ckpt_subparsers.add_parser('list', help='List available checkpoints')
    ckpt_list.add_argument('-d', '--dir', type=str, default='./output',
                          help='Checkpoint directory (default: ./output)')
    
    # Show checkpoint info
    ckpt_info = ckpt_subparsers.add_parser('info', help='Show checkpoint details')
    ckpt_info.add_argument('path', type=str, help='Path to checkpoint file')
    
    # Export checkpoint
    ckpt_export = ckpt_subparsers.add_parser('export', help='Export checkpoint to Parquet')
    ckpt_export.add_argument('path', type=str, help='Path to checkpoint file')
    ckpt_export.add_argument('-o', '--output', type=str, required=True,
                            help='Output Parquet file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        'generate': cmd_generate,
        'validate': cmd_validate,
        'stats': cmd_stats,
        'example': cmd_example,
        'checkpoint': cmd_checkpoint,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
