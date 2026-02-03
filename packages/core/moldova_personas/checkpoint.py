"""Checkpoint system for resumable persona generation.

Provides save/resume functionality for long-running generation tasks,
enabling crash recovery and progress tracking.
"""

import json
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator, Callable
from dataclasses import dataclass, asdict

import numpy as np

from .models import Persona


logger = logging.getLogger(__name__)


def _serialize_random_state(state: object) -> object:
    """Convert Python random state into JSON-serializable structure."""
    if isinstance(state, tuple):
        return [_serialize_random_state(item) for item in state]
    if isinstance(state, list):
        return [_serialize_random_state(item) for item in state]
    return state


def _deserialize_random_state(state: object) -> object:
    """Convert JSON-serialized random state back to tuple structure."""
    if isinstance(state, list):
        return tuple(_deserialize_random_state(item) for item in state)
    return state


def _serialize_numpy_state(state: Optional[tuple]) -> Optional[Dict[str, Any]]:
    """Convert NumPy RNG state into JSON-friendly dict."""
    if state is None:
        return None
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist() if hasattr(keys, "tolist") else list(keys),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _deserialize_numpy_state(state: Optional[Dict[str, Any]]) -> Optional[tuple]:
    """Reconstruct NumPy RNG state tuple from JSON-friendly dict."""
    if not state:
        return None
    keys = np.array(state["keys"], dtype="uint32")
    return (
        state["name"],
        keys,
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


@dataclass
class Checkpoint:
    """Represents a generation checkpoint."""
    
    # Generation state
    total_target: int
    completed_count: int
    seed: Optional[int]
    
    # Progress tracking
    completed_personas: List[Dict[str, Any]]

    # Metadata
    timestamp: str

    # Optional JSONL file path for persisted personas (streaming)
    data_file: Optional[str] = None
    version: str = "1.0"

    # RNG state for deterministic resume
    random_state: Optional[object] = None
    numpy_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "total_target": self.total_target,
            "completed_count": self.completed_count,
            "seed": self.seed,
            "completed_personas": self.completed_personas,
            "data_file": self.data_file,
            "random_state": self.random_state,
            "numpy_state": self.numpy_state,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            total_target=data["total_target"],
            completed_count=data["completed_count"],
            seed=data.get("seed"),
            completed_personas=data["completed_personas"],
            data_file=data.get("data_file"),
            random_state=data.get("random_state"),
            numpy_state=data.get("numpy_state"),
        )
    
    @property
    def remaining_count(self) -> int:
        """Number of personas remaining to generate."""
        return self.total_target - self.completed_count
    
    @property
    def progress_pct(self) -> float:
        """Progress percentage."""
        return (self.completed_count / self.total_target) * 100 if self.total_target > 0 else 0


class CheckpointManager:
    """Manages checkpoint save/load operations."""
    
    def __init__(self, checkpoint_dir: str, basename: str = "checkpoint"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            basename: Base name for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.basename = basename

    def get_data_path(self) -> Path:
        """Path for JSONL data file storing persisted personas."""
        return self.checkpoint_dir / f"{self.basename}_data.jsonl"

    def reset_data_file(self) -> Path:
        """Create or truncate the JSONL data file."""
        path = self.get_data_path()
        with open(path, "w", encoding="utf-8"):
            pass
        return path

    def append_persona(self, persona_dict: Dict[str, Any]) -> None:
        """Append a persona record to the JSONL data file."""
        path = self.get_data_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(persona_dict, ensure_ascii=False) + "\n")

    def iter_personas(self, data_file: Optional[str] = None):
        """Yield personas from a JSONL data file."""
        path = Path(data_file) if data_file else self.get_data_path()
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Persona(**json.loads(line))
                except Exception as e:
                    logger.warning(f"Skipping invalid persona line in {path}: {e}")
    
    def save(self, checkpoint: Checkpoint, intermediate: bool = False) -> str:
        """
        Save checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint to save
            intermediate: If True, save as intermediate checkpoint (not final)
            
        Returns:
            Path to saved checkpoint file
        """
        if intermediate:
            filename = f"{self.basename}_{checkpoint.completed_count:06d}.json"
        else:
            filename = f"{self.basename}_latest.json"
        
        filepath = self.checkpoint_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Checkpoint saved: {filepath} ({checkpoint.progress_pct:.1f}% complete)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load(self, filepath: Optional[str] = None) -> Optional[Checkpoint]:
        """
        Load checkpoint from disk.
        
        Args:
            filepath: Specific checkpoint file to load. If None, loads latest.
            
        Returns:
            Loaded checkpoint or None if not found
        """
        if filepath:
            path = Path(filepath)
        else:
            # Find latest checkpoint
            path = self._find_latest_checkpoint()
        
        if not path or not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = Checkpoint.from_dict(data)
            logger.info(f"Checkpoint loaded: {path} ({checkpoint.progress_pct:.1f}% complete)")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file by modification time."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.basename}_*.json"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoint files."""
        return sorted(self.checkpoint_dir.glob(f"{self.basename}_*.json"))
    
    def cleanup_old_checkpoints(self, keep_last: int = 3) -> int:
        """
        Remove old intermediate checkpoints, keeping only the most recent.
        
        Args:
            keep_last: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints removed
        """
        all_checkpoints = self.list_checkpoints()
        
        # Keep latest and remove others
        to_remove = all_checkpoints[:-keep_last] if len(all_checkpoints) > keep_last else []
        
        removed = 0
        for path in to_remove:
            try:
                path.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {path}: {e}")
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old checkpoint(s)")
        
        return removed


class CheckpointStreamingGenerator:
    """
    Generator that yields personas one at a time with checkpoint support.

    This enables:
    - Constant memory usage regardless of batch size
    - Crash recovery via checkpoints
    - Real-time progress tracking
    """
    
    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        checkpoint_every: int = 1000,
        on_progress: Optional[Callable[[int, int], None]] = None,
        persist_personas: bool = True,
    ):
        """
        Initialize streaming generator.
        
        Args:
            checkpoint_manager: Optional checkpoint manager for persistence
            checkpoint_every: Save checkpoint every N personas
            on_progress: Callback(current, total) for progress updates
        """
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_every = checkpoint_every
        self.on_progress = on_progress
        self.persist_personas = persist_personas
    
    def generate_streaming(
        self,
        generator,
        n: int,
        seed: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> Iterator[Persona]:
        """
        Generate personas with streaming and checkpoint support.
        
        Args:
            generator: PersonaGenerator instance
            n: Total number of personas to generate
            seed: Random seed for reproducibility
            resume_from: Path to checkpoint file to resume from
            
        Yields:
            Persona objects one at a time
        """
        # Try to load checkpoint
        checkpoint = None
        if resume_from and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load(resume_from)
        
        # Determine starting point
        if checkpoint:
            start_count = checkpoint.completed_count
            personas_to_generate = n - start_count
            
            # Yield already completed personas first
            logger.info(f"Resuming from checkpoint: {start_count} personas already generated")
            if checkpoint.data_file and self.checkpoint_manager:
                for persona in self.checkpoint_manager.iter_personas(checkpoint.data_file):
                    yield persona
            else:
                for persona_data in checkpoint.completed_personas:
                    yield Persona(**persona_data)
                if self.checkpoint_manager and self.persist_personas:
                    data_path = self.checkpoint_manager.reset_data_file()
                    for persona_data in checkpoint.completed_personas:
                        self.checkpoint_manager.append_persona(persona_data)
                    checkpoint.data_file = str(data_path)
            # Restore RNG state for deterministic resume if available
            if checkpoint.random_state is not None:
                try:
                    random.setstate(_deserialize_random_state(checkpoint.random_state))
                except Exception as exc:
                    logger.warning(f"Failed to restore Python RNG state: {exc}")
            if checkpoint.numpy_state is not None:
                try:
                    np_state = _deserialize_numpy_state(checkpoint.numpy_state)
                    if np_state:
                        np.random.set_state(np_state)
                except Exception as exc:
                    logger.warning(f"Failed to restore NumPy RNG state: {exc}")
        else:
            start_count = 0
            personas_to_generate = n
            checkpoint = Checkpoint(
                total_target=n,
                completed_count=0,
                seed=seed,
                completed_personas=[],
                timestamp=datetime.now().isoformat(),
            )
            if self.checkpoint_manager and self.persist_personas:
                data_path = self.checkpoint_manager.reset_data_file()
                checkpoint.data_file = str(data_path)
        
        # Generate remaining personas
        buffer = checkpoint.completed_personas.copy() if (checkpoint and not self.persist_personas) else []
        
        for i in range(personas_to_generate):
            # Generate next persona
            persona = generator.generate_single()
            
            # Convert to dict for checkpoint
            persona_dict = persona.model_dump()
            if self.checkpoint_manager and self.persist_personas:
                self.checkpoint_manager.append_persona(persona_dict)
            else:
                buffer.append(persona_dict)
            
            current_count = start_count + i + 1
            
            # Save checkpoint if needed
            if self.checkpoint_manager and current_count % self.checkpoint_every == 0:
                checkpoint.completed_count = current_count
                checkpoint.completed_personas = buffer
                checkpoint.timestamp = datetime.now().isoformat()
                checkpoint.random_state = _serialize_random_state(random.getstate())
                checkpoint.numpy_state = _serialize_numpy_state(np.random.get_state())
                
                self.checkpoint_manager.save(checkpoint, intermediate=True)
                
                # Cleanup old checkpoints
                self.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)
            
            # Progress callback
            if self.on_progress:
                self.on_progress(current_count, n)
            
            yield persona
        
        # Save final checkpoint
        if self.checkpoint_manager:
            checkpoint.completed_count = n
            checkpoint.completed_personas = buffer
            checkpoint.timestamp = datetime.now().isoformat()
            checkpoint.random_state = _serialize_random_state(random.getstate())
            checkpoint.numpy_state = _serialize_numpy_state(np.random.get_state())
            self.checkpoint_manager.save(checkpoint, intermediate=False)
            logger.info(f"Generation complete: {n} personas")


# Backward-compatible alias (deprecated): prefer CheckpointStreamingGenerator.
StreamingGenerator = CheckpointStreamingGenerator


def export_checkpoint_to_parquet(checkpoint_path: str, output_path: str) -> None:
    """
    Export a checkpoint file to Parquet format.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        output_path: Output Parquet file path
    """
    import pandas as pd
    
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    personas = data.get("completed_personas", [])
    data_file = data.get("data_file")
    if not personas and data_file:
        # Load from JSONL if checkpoint used streaming persistence
        try:
            path = Path(data_file)
            with open(path, "r", encoding="utf-8") as f:
                personas = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.warning(f"Failed to load JSONL data file {data_file}: {e}")

    if not personas:
        logger.warning("No personas found in checkpoint")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(personas)
    
    # Convert list columns to strings
    for col in ['skills_and_expertise_list', 'hobbies_and_interests_list']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else "[]")
    
    # Write to Parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    logger.info(f"Exported {len(personas)} personas to {output_path}")


def get_checkpoint_summary(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get summary information about a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Summary dictionary
    """
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = data.get("total_target", 0)
        completed = data.get("completed_count", 0)
        
        return {
            "path": checkpoint_path,
            "version": data.get("version", "unknown"),
            "timestamp": data.get("timestamp", "unknown"),
            "total_target": total,
            "completed_count": completed,
            "remaining": total - completed,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
            "seed": data.get("seed"),
        }
    except Exception as e:
        return {
            "path": checkpoint_path,
            "error": str(e),
        }
