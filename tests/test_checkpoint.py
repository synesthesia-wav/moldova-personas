"""Tests for checkpoint system."""

import pytest
import json
import tempfile
from pathlib import Path

from moldova_personas.checkpoint import (
    Checkpoint,
    CheckpointManager,
    CheckpointStreamingGenerator,
    get_checkpoint_summary,
)
from moldova_personas import PersonaGenerator


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""
    
    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        ckpt = Checkpoint(
            total_target=1000,
            completed_count=500,
            seed=42,
            completed_personas=[],
            timestamp="2026-01-29T12:00:00",
        )
        
        assert ckpt.total_target == 1000
        assert ckpt.completed_count == 500
        assert ckpt.remaining_count == 500
        assert ckpt.progress_pct == 50.0
    
    def test_checkpoint_to_dict(self):
        """Test checkpoint serialization."""
        ckpt = Checkpoint(
            total_target=100,
            completed_count=50,
            seed=42,
            completed_personas=[{"name": "Test"}],
            timestamp="2026-01-29T12:00:00",
        )
        
        data = ckpt.to_dict()
        assert data["total_target"] == 100
        assert data["completed_count"] == 50
        assert data["seed"] == 42
    
    def test_checkpoint_from_dict(self):
        """Test checkpoint deserialization."""
        data = {
            "version": "1.0",
            "timestamp": "2026-01-29T12:00:00",
            "total_target": 100,
            "completed_count": 50,
            "seed": 42,
            "completed_personas": [],
        }
        
        ckpt = Checkpoint.from_dict(data)
        assert ckpt.total_target == 100
        assert ckpt.completed_count == 50
        assert ckpt.seed == 42


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    def test_save_and_load(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            ckpt = Checkpoint(
                total_target=1000,
                completed_count=500,
                seed=42,
                completed_personas=[{"name": "Test Persona"}],
                timestamp="2026-01-29T12:00:00",
            )
            
            # Save
            path = manager.save(ckpt)
            assert Path(path).exists()
            
            # Load
            loaded = manager.load(path)
            assert loaded.total_target == 1000
            assert loaded.completed_count == 500
    
    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            # Create two checkpoints
            ckpt1 = Checkpoint(
                total_target=1000,
                completed_count=100,
                seed=42,
                completed_personas=[],
                timestamp="2026-01-29T12:00:00",
            )
            ckpt2 = Checkpoint(
                total_target=1000,
                completed_count=200,
                seed=42,
                completed_personas=[],
                timestamp="2026-01-29T12:01:00",
            )
            
            manager.save(ckpt1, intermediate=True)
            import time
            time.sleep(0.1)
            manager.save(ckpt2, intermediate=True)
            
            # Find latest
            latest = manager._find_latest_checkpoint()
            assert latest is not None
            
            # Verify it's the second one
            loaded = manager.load(str(latest))
            assert loaded.completed_count == 200
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            # Create checkpoints
            ckpt = Checkpoint(
                total_target=1000,
                completed_count=100,
                seed=42,
                completed_personas=[],
                timestamp="2026-01-29T12:00:00",
            )
            
            manager.save(ckpt, intermediate=True)
            
            # List
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1
    
    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            # Create multiple checkpoints
            for i in range(5):
                ckpt = Checkpoint(
                    total_target=1000,
                    completed_count=(i + 1) * 100,
                    seed=42,
                    completed_personas=[],
                    timestamp=f"2026-01-29T12:0{i}:00",
                )
                manager.save(ckpt, intermediate=True)
                import time
                time.sleep(0.05)
            
            # Should have 5 checkpoints
            assert len(manager.list_checkpoints()) == 5
            
            # Cleanup, keeping last 2
            removed = manager.cleanup_old_checkpoints(keep_last=2)
            assert removed == 3
            assert len(manager.list_checkpoints()) == 2


class TestStreamingGenerator:
    """Tests for CheckpointStreamingGenerator."""
    
    def test_streaming_generation(self):
        """Test basic streaming generation."""
        generator = PersonaGenerator(seed=42)
        streaming = CheckpointStreamingGenerator()
        
        personas = []
        for persona in streaming.generate_streaming(generator, n=10):
            personas.append(persona)
        
        assert len(personas) == 10
        assert all(p.name for p in personas)
    
    def test_streaming_with_checkpointing(self):
        """Test streaming with checkpoint save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            generator = PersonaGenerator(seed=42)
            
            streaming = CheckpointStreamingGenerator(
                checkpoint_manager=manager,
                checkpoint_every=5,
            )
            
            personas = []
            for persona in streaming.generate_streaming(generator, n=10):
                personas.append(persona)
            
            # Should have saved intermediate checkpoint
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 1
    
    def test_resume_from_checkpoint(self):
        """Test resuming generation from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            generator = PersonaGenerator(seed=42)
            
            # First, generate some with checkpointing
            streaming = CheckpointStreamingGenerator(
                checkpoint_manager=manager,
                checkpoint_every=5,
            )
            
            personas_first = []
            for i, persona in enumerate(streaming.generate_streaming(generator, n=10)):
                personas_first.append(persona)
                if i == 4:  # Stop after 5
                    break
            
            # Create manual checkpoint for 5 personas
            ckpt = Checkpoint(
                total_target=10,
                completed_count=5,
                seed=42,
                completed_personas=[p.model_dump() for p in personas_first],
                timestamp="2026-01-29T12:00:00",
            )
            ckpt_path = manager.save(ckpt, intermediate=True)
            
            # Resume
            personas_second = []
            for persona in streaming.generate_streaming(
                generator, 
                n=10,
                resume_from=ckpt_path
            ):
                personas_second.append(persona)
            
            # Should have total of 10 (5 from checkpoint + 5 new)
            assert len(personas_second) == 10

    def test_resume_is_deterministic(self):
        """Test that resume produces the same sequence as a full run."""
        def signature(persona):
            return (
                persona.name,
                persona.sex,
                persona.age,
                persona.ethnicity,
                persona.region,
                persona.residence_type,
                persona.education_level,
                persona.occupation,
                persona.marital_status,
            )

        # Full run
        generator_full = PersonaGenerator(seed=123)
        streaming_full = CheckpointStreamingGenerator()
        expected = [
            signature(p) for p in streaming_full.generate_streaming(generator_full, n=20)
        ]

        # Partial run with checkpointing
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            generator_partial = PersonaGenerator(seed=123)
            streaming_partial = CheckpointStreamingGenerator(
                checkpoint_manager=manager,
                checkpoint_every=5,
            )

            for i, _ in enumerate(
                streaming_partial.generate_streaming(generator_partial, n=20)
            ):
                if i == 9:
                    break

            latest = manager._find_latest_checkpoint()
            assert latest is not None

            generator_resume = PersonaGenerator(seed=123)
            streaming_resume = CheckpointStreamingGenerator(
                checkpoint_manager=manager,
                checkpoint_every=5,
            )
            resumed = [
                signature(p)
                for p in streaming_resume.generate_streaming(
                    generator_resume, n=20, resume_from=str(latest)
                )
            ]

        assert resumed == expected

    def test_resume_smaller_target_raises(self):
        """Test resume fails when target n is smaller than completed_count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            ckpt = Checkpoint(
                total_target=10,
                completed_count=5,
                seed=42,
                completed_personas=[],
                timestamp="2026-01-29T12:00:00",
            )
            ckpt_path = manager.save(ckpt, intermediate=True)

            streaming = CheckpointStreamingGenerator(checkpoint_manager=manager)
            generator = PersonaGenerator(seed=42)
            with pytest.raises(ValueError):
                list(
                    streaming.generate_streaming(
                        generator, n=4, resume_from=ckpt_path
                    )
                )


class TestCheckpointSummary:
    """Tests for checkpoint summary."""
    
    def test_get_checkpoint_summary(self):
        """Test getting checkpoint summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            ckpt = Checkpoint(
                total_target=1000,
                completed_count=750,
                seed=42,
                completed_personas=[],
                timestamp="2026-01-29T12:00:00",
            )
            
            path = manager.save(ckpt)
            summary = get_checkpoint_summary(path)
            
            assert summary["total_target"] == 1000
            assert summary["completed_count"] == 750
            assert summary["remaining"] == 250
            assert summary["progress_pct"] == 75.0
            assert summary["seed"] == 42
    
    def test_get_checkpoint_summary_error(self):
        """Test summary with invalid file."""
        summary = get_checkpoint_summary("/nonexistent/path.json")
        assert "error" in summary
