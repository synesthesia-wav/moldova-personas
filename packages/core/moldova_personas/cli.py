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
from collections import Counter
from typing import List, Optional, Dict

import pandas as pd

from .generator import PersonaGenerator
from .validators import ValidationPipeline
from .exporters import export_formats, StatisticsExporter, build_statistics_from_counts
from .census_data import CENSUS, CensusDistributions
from .pxweb_fetcher import NBSDataManager
from .geo_tables import strict_geo_enabled
from .run_manifest import RunManifest
from . import __version__ as generator_version
from .checkpoint import (
    CheckpointManager,
    CheckpointStreamingGenerator,
    get_checkpoint_summary,
)
from .streaming import (
    StreamingGenerator as BatchStreamingGenerator,
    StreamingParquetExporter,
    StreamingJSONLExporter,
    StreamingCSVExporter,
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

SUPPORTED_FORMATS = {"parquet", "json", "jsonl", "csv", "all"}
STREAMING_FORMATS = {"parquet", "jsonl", "csv"}


def _parse_comma_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_formats(value: str) -> List[str]:
    raw = (value or "all").strip().lower()
    if raw == "all":
        return ["parquet", "json", "jsonl", "csv"]
    formats = _parse_comma_list(raw)
    if "all" in formats:
        return ["parquet", "json", "jsonl", "csv"]
    unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
    if unknown:
        raise ValueError(f"Unsupported format(s): {', '.join(unknown)}")
    return formats


def _get_census_data(cache_dir: Optional[str]) -> CensusDistributions:
    if cache_dir:
        manager = NBSDataManager(cache_dir=cache_dir)
        return CensusDistributions(_manager=manager)
    return CENSUS


def _write_manifest(
    output_dir: Path,
    output_files: Dict[str, str],
    args: argparse.Namespace,
    census_data: CensusDistributions,
    start_time: datetime,
    end_time: datetime,
) -> Optional[str]:
    manifest = RunManifest()
    manifest.generator_version = generator_version
    manifest.generation_start_time = start_time.isoformat()
    manifest.generation_end_time = end_time.isoformat()
    manifest.duration_seconds = (end_time - start_time).total_seconds()
    manifest.random_seed = args.seed
    manifest.np_random_seed = args.seed
    manifest.target_persona_count = args.count
    manifest.output_files = output_files

    config = {
        "command": args.command,
        "count": args.count,
        "seed": args.seed,
        "ipf": args.ipf,
        "narratives": args.narratives,
        "llm_provider": args.llm_provider,
        "llm_workers": args.llm_workers,
        "llm_rate_limit": args.llm_rate_limit,
        "llm_delay": args.llm_delay,
        "strict_geo": strict_geo_enabled(),
        "format": args.format,
        "partition_by": args.partition_by,
        "drop_fields": args.drop_fields,
        "stream": args.stream,
        "batch_size": args.batch_size,
        "cache_dir": args.cache_dir,
        "checkpoint_every": args.checkpoint_every,
        "resume_from": args.resume_from,
    }
    manifest.config_summary = config
    manifest.compute_config_hash(config)

    manifest.pxweb_snapshot_timestamp = census_data.get_pxweb_snapshot_timestamp()
    for field, info in census_data.get_all_provenance().items():
        manifest.add_field_provenance(
            field_name=field,
            source_type=info.get("provenance", "unknown"),
            source_table=info.get("source_table"),
            source_timestamp=info.get("last_fetched"),
            confidence=float(info.get("confidence", 1.0) or 1.0),
        )

    manifest_path = output_dir / "metadata.json"
    manifest.save(str(manifest_path))
    return str(manifest_path)


def cmd_generate(args):
    """Generate personas command with checkpointing support."""
    setup_logging(verbose=args.verbose)

    try:
        formats = _resolve_formats(args.format)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    drop_fields = _parse_comma_list(args.drop_fields)
    partition_cols = _parse_comma_list(args.partition_by)
    include_stats = args.stats or (args.format or "").strip().lower() == "all"

    if partition_cols and "parquet" not in formats:
        logger.warning("Ignoring --partition-by because parquet format is not selected.")

    census_data = _get_census_data(args.cache_dir)

    logger.info(f"Generating {args.count} synthetic Moldovan personas...")
    logger.info(f"Random seed: {args.seed or 'random'}")
    logger.info(f"Using IPF adjustment: {args.ipf}")
    logger.info(f"LLM provider: {args.llm_provider}")

    start_time = datetime.now()

    # Streaming export mode (no in-memory accumulation)
    if args.stream:
        if args.ipf:
            logger.error("Streaming mode does not support IPF adjustment.")
            return 1
        if args.llm_provider != "mock" or args.narratives:
            logger.error("Streaming mode does not support narrative generation.")
            return 1
        if args.resume_from:
            logger.error("Streaming mode does not support checkpoint resume.")
            return 1
        if args.checkpoint_every > 0:
            logger.info("Checkpointing is ignored in streaming mode.")
        if partition_cols:
            logger.error("Streaming mode does not support Parquet partitioning.")
            return 1
        if len(formats) != 1 or formats[0] not in STREAMING_FORMATS:
            logger.error("Streaming mode supports a single format: parquet, jsonl, or csv.")
            return 1
        if not args.skip_validation:
            logger.warning("Skipping validation in streaming mode. Use non-streaming for validation.")

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        format_name = formats[0]
        if format_name == "parquet":
            output_path = output_dir / f"{args.basename}.parquet"
            exporter = StreamingParquetExporter(
                str(output_path),
                compression=args.parquet_compression,
                row_group_size=args.row_group_size,
                drop_fields=drop_fields,
            )
        elif format_name == "jsonl":
            output_path = output_dir / f"{args.basename}.jsonl"
            exporter = StreamingJSONLExporter(str(output_path), drop_fields=drop_fields)
        else:
            output_path = output_dir / f"{args.basename}.csv"
            exporter = StreamingCSVExporter(
                str(output_path),
                separator=args.csv_separator,
                drop_fields=drop_fields,
            )

        generator = PersonaGenerator(census_data=census_data, seed=args.seed)
        streamer = BatchStreamingGenerator(generator=generator)

        sex_counts: Counter = Counter()
        age_counts: Counter = Counter()
        region_counts: Counter = Counter()
        ethnicity_counts: Counter = Counter()
        education_counts: Counter = Counter()
        marital_counts: Counter = Counter()
        residence_counts: Counter = Counter()
        employment_counts: Counter = Counter()
        total = 0

        with exporter as writer:
            for batch in streamer.generate_batches(n=args.count, batch_size=args.batch_size):
                writer.write_batch(batch)
                for p in batch:
                    total += 1
                    sex_counts[p.sex] += 1
                    age_counts[p.age_group] += 1
                    region_counts[p.region] += 1
                    ethnicity_counts[p.ethnicity] += 1
                    education_counts[p.education_level] += 1
                    marital_counts[p.marital_status] += 1
                    residence_counts[p.residence_type] += 1
                    employment_counts[p.employment_status] += 1

        results = {format_name: str(output_path)}

        if include_stats:
            stats = build_statistics_from_counts(
                total=total,
                sex_counts=dict(sex_counts),
                age_counts=dict(age_counts),
                region_counts=dict(region_counts),
                ethnicity_counts=dict(ethnicity_counts),
                education_counts=dict(education_counts),
                marital_counts=dict(marital_counts),
                residence_counts=dict(residence_counts),
                employment_counts=dict(employment_counts),
            )
            stats_json_path = output_dir / f"{args.basename}_stats.json"
            StatisticsExporter().export_json(stats, str(stats_json_path))
            results["stats_json"] = str(stats_json_path)
            stats_md_path = output_dir / f"{args.basename}_stats.md"
            StatisticsExporter().export_markdown(stats, str(stats_md_path), census_data=census_data)
            results["stats_markdown"] = str(stats_md_path)

        end_time = datetime.now()
        if not args.no_manifest:
            manifest_path = _write_manifest(output_dir, results, args, census_data, start_time, end_time)
            if manifest_path:
                results["manifest"] = manifest_path

        logger.info("Generation complete!")
        return 0

    # Non-streaming mode (supports checkpointing and narratives)
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
    generator = PersonaGenerator(
        census_data=census_data,
        seed=args.seed,
        include_ocean=args.ocean,
    )

    # Generate structured data with streaming
    start_time = datetime.now()
    if args.ipf:
        personas = generator.generate_with_ipf_adjustment(args.count)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(personas)} structured personas in {duration:.2f} seconds")
    else:
        streaming_gen = CheckpointStreamingGenerator(
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
        elif args.llm_provider == 'gemini':
            if args.gemini_api_key:
                llm_kwargs['api_key'] = args.gemini_api_key
            if args.gemini_model:
                llm_kwargs['model'] = args.gemini_model
        elif args.llm_provider == 'kimi':
            if args.kimi_api_key:
                llm_kwargs['api_key'] = args.kimi_api_key
            if args.kimi_model:
                llm_kwargs['model'] = args.kimi_model
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
            delay = args.llm_delay
            if args.llm_rate_limit and args.llm_rate_limit > 0:
                delay = max(delay, 1.0 / args.llm_rate_limit)
            personas = nar_gen.generate_batch(
                personas,
                show_progress=True,
                delay=delay
            )
            nar_duration = (datetime.now() - nar_start).total_seconds()
            logger.info(f"Generated narratives in {nar_duration:.2f} seconds")

    report = None
    if not args.skip_validation:
        logger.info("Validating generated personas...")
        validator = ValidationPipeline(census_data=census_data)
        report = validator.validate(personas)
        print(report.summary())  # Keep this print for user-facing output
    else:
        logger.info("Skipping validation per --skip-validation")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to {output_dir}...")
    results = export_formats(
        personas,
        str(output_dir),
        basename=args.basename,
        formats=formats,
        drop_fields=drop_fields,
        partition_cols=partition_cols or None,
        parquet_compression=args.parquet_compression,
        include_stats=include_stats,
        census_data=census_data,
    )

    end_time = datetime.now()
    if not args.no_manifest:
        manifest_path = _write_manifest(output_dir, results, args, census_data, start_time, end_time)
        if manifest_path:
            results["manifest"] = manifest_path

    logger.info("Generation complete!")
    return 0


def cmd_validate(args):
    """Validate existing personas command."""
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Validating personas from {args.input}...")
    census_data = _get_census_data(args.cache_dir)
    
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
    validator = ValidationPipeline(census_data=census_data)
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
    census_data = _get_census_data(args.cache_dir)
    
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
    exporter.export_markdown(stats, str(output_path), census_data=census_data)
    
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


def cmd_cache(args):
    """Cache management command."""
    setup_logging(verbose=args.verbose)
    if not args.cache_command:
        print("Usage: cache {refresh}")
        return 1
    if args.cache_command == "refresh":
        return cmd_cache_refresh(args)
    return 0


def cmd_cache_refresh(args):
    """Refresh PxWeb cache data."""
    census_data = _get_census_data(args.cache_dir)
    logger.info("Refreshing PxWeb cache...")
    results = census_data.refresh_from_pxweb()
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"{name}: {status}")
    return 0 if all(results.values()) else 1


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
    gen_parser.add_argument('--ocean', dest='ocean', action='store_true', default=True,
                          help='Include OCEAN personality fields (default: enabled)')
    gen_parser.add_argument('--no-ocean', dest='ocean', action='store_false',
                          help='Disable OCEAN personality fields')
    gen_parser.add_argument('--llm-provider', type=str, default='mock',
                          choices=['mock', 'openai', 'gemini', 'kimi', 'qwen', 'qwen-local', 'dashscope', 'local'],
                          help='LLM provider for narratives (default: mock)')
    gen_parser.add_argument('--openai-api-key', type=str, default=None,
                          help='OpenAI API key (or set OPENAI_API_KEY env var)')
    gen_parser.add_argument('--openai-model', type=str, default='gpt-3.5-turbo',
                          help='OpenAI model to use (default: gpt-3.5-turbo)')
    gen_parser.add_argument('--dashscope-api-key', type=str, default=None,
                          help='DashScope API key for Qwen (or set DASHSCOPE_API_KEY env var)')
    gen_parser.add_argument('--qwen-model', type=str, default='qwen-turbo',
                          help='Qwen model: qwen-turbo, qwen-plus, qwen-max, qwen2.5-7b, etc.')
    gen_parser.add_argument('--gemini-api-key', type=str, default=None,
                          help='Gemini API key (or set GEMINI_API_KEY env var)')
    gen_parser.add_argument('--gemini-model', type=str, default='gemini-2.5-flash',
                          help='Gemini model (default: gemini-2.5-flash)')
    gen_parser.add_argument('--kimi-api-key', type=str, default=None,
                          help='Kimi API key (or set KIMI_API_KEY env var)')
    gen_parser.add_argument('--kimi-model', type=str, default='moonshot-v1-8k',
                          help='Kimi model (default: moonshot-v1-8k)')
    gen_parser.add_argument('--qwen-local-model', type=str, default='qwen2.5-7b',
                          help='Local Qwen model: qwen2.5-7b, qwen2.5-14b, etc.')
    gen_parser.add_argument('--llm-delay', type=float, default=0.1,
                          help='Delay between LLM API calls in seconds (default: 0.1)')
    gen_parser.add_argument('--llm-workers', type=int, default=1,
                          help='Number of parallel LLM workers (default: 1, use 5-20 for speedup)')
    gen_parser.add_argument('--llm-rate-limit', type=float, default=None,
                          help='Rate limit in requests per second (optional, serial/parallel)')
    gen_parser.add_argument('--checkpoint-every', type=int, default=1000,
                          help='Save checkpoint every N personas (default: 1000, set 0 to disable)')
    gen_parser.add_argument('--checkpoint-dir', type=str, default=None,
                          help='Directory for checkpoint files (default: same as output)')
    gen_parser.add_argument('--resume-from', type=str, default=None,
                          help='Resume from checkpoint file')
    gen_parser.add_argument('--format', type=str, default='all',
                          help='Output format: parquet, json, jsonl, csv, all (comma-separated allowed)')
    gen_parser.add_argument('--partition-by', type=str, default=None,
                          help='Comma-separated columns to partition Parquet output by')
    gen_parser.add_argument('--drop-fields', type=str, default=None,
                          help='Comma-separated fields to exclude from output')
    gen_parser.add_argument('--parquet-compression', type=str, default='snappy',
                          help='Parquet compression codec (default: snappy)')
    gen_parser.add_argument('--row-group-size', type=int, default=10000,
                          help='Parquet row group size for streaming export (default: 10000)')
    gen_parser.add_argument('--csv-separator', type=str, default=',',
                          help='CSV separator for streaming export (default: ,)')
    gen_parser.add_argument('--stats', action='store_true',
                          help='Export stats files alongside selected formats')
    gen_parser.add_argument('--stream', action='store_true',
                          help='Use streaming generation/export (no in-memory accumulation)')
    gen_parser.add_argument('--batch-size', type=int, default=10000,
                          help='Streaming batch size (default: 10000)')
    gen_parser.add_argument('--skip-validation', action='store_true',
                          help='Skip validation step')
    gen_parser.add_argument('--no-manifest', action='store_true',
                          help='Disable metadata manifest output')
    gen_parser.add_argument('--cache-dir', type=str, default=None,
                          help='Override PxWeb cache directory')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate existing personas')
    val_parser.add_argument('-i', '--input', type=str, required=True,
                          help='Input file (parquet, csv, json, jsonl)')
    val_parser.add_argument('-o', '--output', type=str, default=None,
                          help='Output file for validation report')
    val_parser.add_argument('-t', '--tolerance', type=float, default=0.02,
                          help='Statistical tolerance (default: 0.02 = 2%%)')
    val_parser.add_argument('--cache-dir', type=str, default=None,
                          help='Override PxWeb cache directory')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate statistics report')
    stats_parser.add_argument('-i', '--input', type=str, required=True,
                            help='Input file')
    stats_parser.add_argument('-o', '--output', type=str, default=None,
                            help='Output file for statistics report')
    stats_parser.add_argument('--cache-dir', type=str, default=None,
                            help='Override PxWeb cache directory')
    
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

    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage PxWeb cache')
    cache_subparsers = cache_parser.add_subparsers(dest='cache_command', help='Cache commands')
    cache_refresh = cache_subparsers.add_parser('refresh', help='Refresh PxWeb cache')
    cache_refresh.add_argument('--cache-dir', type=str, default=None,
                               help='Override PxWeb cache directory')
    
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
        'cache': cmd_cache,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
