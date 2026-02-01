"""
Streaming Generation and Export for Large Datasets

Supports generating and exporting 1M+ personas without memory issues.
"""

import json
import logging
from typing import Iterator, Optional, Callable, Dict, Any
from pathlib import Path
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from .models import Persona
from .generator import PersonaGenerator


logger = logging.getLogger(__name__)


class StreamingGenerator:
    """
    Generate personas in streaming fashion for memory efficiency.
    
    Usage:
        streamer = StreamingGenerator(seed=42)
        for batch in streamer.generate_batches(n=1_000_000, batch_size=10_000):
            # Process batch (write to file, etc.)
            exporter.write_batch(batch)
    """
    
    def __init__(self, generator: Optional[PersonaGenerator] = None, seed: Optional[int] = None):
        """
        Initialize streaming generator.
        
        Args:
            generator: Existing generator (creates new if None)
            seed: Random seed for reproducibility
        """
        self.generator = generator or PersonaGenerator(seed=seed)
        self.generated_count = 0
    
    def generate_batches(
        self,
        n: int,
        batch_size: int = 10_000,
        use_ethnicity_correction: bool = False,
    ) -> Iterator[list]:
        """
        Generate personas in batches.
        
        Args:
            n: Total number of personas to generate
            batch_size: Number of personas per batch
            use_ethnicity_correction: Whether to apply IPF (not supported in streaming)
            
        Yields:
            Batches of Persona objects
        """
        if use_ethnicity_correction:
            logger.warning("IPF correction not supported in streaming mode. "
                          "Use standard generate_with_ethnicity_correction for smaller datasets.")
        
        remaining = n
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            batch = []
            
            for _ in range(current_batch_size):
                persona = self.generator.generate_single()
                batch.append(persona)
            
            self.generated_count += len(batch)
            remaining -= len(batch)
            
            logger.debug(f"Generated batch: {len(batch)} personas "
                        f"(total: {self.generated_count}/{n})")
            
            yield batch
    
    def generate_single_stream(self, n: int) -> Iterator[Persona]:
        """
        Generate personas one at a time (lowest memory footprint).
        
        Args:
            n: Number of personas to generate
            
        Yields:
            Individual Persona objects
        """
        for i in range(n):
            yield self.generator.generate_single()
            self.generated_count += 1
            
            if (i + 1) % 10000 == 0:
                logger.debug(f"Generated {i + 1}/{n} personas")


class StreamingParquetExporter:
    """
    Stream personas to Parquet file without loading all into memory.
    
    Usage:
        with StreamingParquetExporter("output.parquet") as exporter:
            for batch in streamer.generate_batches(1_000_000):
                exporter.write_batch(batch)
    """
    
    def __init__(
        self,
        filepath: str,
        schema: Optional[pa.Schema] = None,
        compression: str = "snappy",
        row_group_size: int = 10000,
    ):
        """
        Initialize streaming Parquet exporter.
        
        Args:
            filepath: Output file path
            schema: PyArrow schema (auto-detected if None)
            compression: Compression codec
            row_group_size: Rows per row group
        """
        self.filepath = Path(filepath)
        self.compression = compression
        self.row_group_size = row_group_size
        self.schema = schema
        self.writer = None
        self.total_rows = 0
    
    def _persona_to_dict(self, persona: Persona) -> Dict[str, Any]:
        """Convert persona to dictionary suitable for PyArrow."""
        data = persona.model_dump()
        
        # Convert list columns to JSON strings
        for col in ['skills_and_expertise_list', 'hobbies_and_interests_list']:
            if col in data and data[col] is not None:
                data[col] = json.dumps(data[col], ensure_ascii=False)
        
        return data
    
    def _create_table(self, personas: list) -> pa.Table:
        """Create PyArrow table from personas."""
        data = [self._persona_to_dict(p) for p in personas]
        
        # Convert to columnar format
        columns = {k: [d[k] for d in data] for k in data[0].keys()}
        
        return pa.table(columns)
    
    def __enter__(self):
        """Context manager entry."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def write_batch(self, personas: list):
        """Write a batch of personas."""
        if not personas:
            return
        
        table = self._create_table(personas)
        
        if self.writer is None:
            # First batch - create writer
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.filepath,
                self.schema,
                compression=self.compression,
                use_dictionary=True,
                write_statistics=True,
            )
        
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.total_rows += len(personas)
        
        logger.debug(f"Wrote batch: {len(personas)} rows (total: {self.total_rows})")
    
    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()
            self.writer = None
            logger.info(f"Streaming export complete: {self.total_rows} rows to {self.filepath}")


class StreamingJSONLExporter:
    """Stream personas to JSONL file."""
    
    def __init__(self, filepath: str):
        """
        Initialize streaming JSONL exporter.
        
        Args:
            filepath: Output file path
        """
        self.filepath = Path(filepath)
        self.file_handle = None
        self.total_rows = 0
    
    def __enter__(self):
        """Context manager entry."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def write_batch(self, personas: list):
        """Write a batch of personas."""
        if not self.file_handle:
            raise RuntimeError("Exporter not opened. Use as context manager.")
        
        for persona in personas:
            line = json.dumps(persona.model_dump(), ensure_ascii=False)
            self.file_handle.write(line + '\n')
        
        self.total_rows += len(personas)
    
    def close(self):
        """Close the file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            logger.info(f"JSONL export complete: {self.total_rows} rows to {self.filepath}")


def generate_and_export_streaming(
    n: int,
    output_path: str,
    batch_size: int = 10_000,
    seed: Optional[int] = None,
    format: str = "parquet",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Generate and export personas in streaming fashion.
    
    Args:
        n: Number of personas to generate
        output_path: Output file path
        batch_size: Number of personas per batch
        seed: Random seed
        format: Output format ("parquet" or "jsonl")
        progress_callback: Optional callback(complete, total) for progress updates
        
    Returns:
        Summary dict with generation stats
    """
    start_time = datetime.now()
    
    streamer = StreamingGenerator(seed=seed)
    
    if format == "parquet":
        exporter_class = StreamingParquetExporter
    elif format == "jsonl":
        exporter_class = StreamingJSONLExporter
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    with exporter_class(output_path) as exporter:
        for batch in streamer.generate_batches(n, batch_size):
            exporter.write_batch(batch)
            
            if progress_callback:
                progress_callback(streamer.generated_count, n)
    
    duration = (datetime.now() - start_time).total_seconds()
    throughput = n / duration if duration > 0 else 0
    
    return {
        "total_generated": n,
        "duration_seconds": duration,
        "throughput_per_second": throughput,
        "output_path": output_path,
        "format": format,
    }


# Convenience function for 1M+ generation
def generate_million_personas(
    output_dir: str = "./output_1m",
    seed: int = 42,
    batch_size: int = 50_000,
) -> Dict[str, Any]:
    """
    Generate 1 million personas with streaming.
    
    Args:
        output_dir: Output directory
        seed: Random seed
        batch_size: Batch size for generation
        
    Returns:
        Summary dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting 1M persona generation with seed={seed}")
    
    # Generate in streaming mode
    parquet_path = output_path / "personas_1m.parquet"
    
    result = generate_and_export_streaming(
        n=1_000_000,
        output_path=str(parquet_path),
        batch_size=batch_size,
        seed=seed,
        format="parquet",
        progress_callback=lambda complete, total: logger.info(
            f"Progress: {complete:,}/{total:,} ({complete/total*100:.1f}%)"
        ) if complete % 100_000 == 0 else None,
    )
    
    # Also export statistics
    logger.info("Generating statistics...")
    # Note: For 1M personas, we'd need to stream through the file
    # This is left as an exercise or done with a separate stats pass
    
    logger.info(f"Complete: {result}")
    return result
