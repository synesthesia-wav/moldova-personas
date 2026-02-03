# TODO: 100K Persona Generation

**Project:** Moldova Personas v0.3.0  
**Target:** 100,000 synthetic personas  
**Reference Standard:** external reference personas dataset (Brazil)

---

## Executive Summary

Based on comparison with reference Brazil dataset, we need to:
1. **Fix ethnicity sampling** (6% underrepresentation)
2. **Add missing narrative fields** (career goals, persona summary)
3. **Run LLM narrative generation** (currently empty!)
4. **Match Reference's narrative quality** (~1,800 char cultural backgrounds)

---

## Phase 1: Critical Fixes (BLOCKERS for 100K)

### Task 1: Fix Ethnicity Underrepresentation 🔴 CRITICAL
**Status:** ⏳ Pending  
**Estimated Time:** 2-4 hours  
**Assignee:** TBD

**Problem:** Moldovans underrepresented by 6.4% (70.3% vs 76.7% NBS target)

**Solution Options:**
- [ ] **Option A:** Adjust ETHNICITY_BY_REGION weights
  - Increase Moldovan % in Chisinau (70% → 78%)
  - Increase Moldovan % in Nord (65% → 75%)
  - Validate national aggregate matches NBS 2024
  
- [ ] **Option B:** Implement IPF (Iterative Proportional Fitting) ← RECOMMENDED
  - Generate initial oversample
  - Resample to match national ethnicity targets
  - Preserve regional correlations
  
- [ ] **Option C:** Two-stage sampling
  - Sample ethnicity from national distribution first
  - Then sample region conditional on ethnicity
  - More complex but exact match

**Validation:** Run 1K test batch, verify chi-square p > 0.05 for ethnicity

---

### Task 2: Delete Old Generated Data ✅ COMPLETE
**Status:** ✅ Done  
**Completed:** 2026-01-29

**Actions Taken:**
- Deleted `output_500_personas/` (outdated 500-sample)
- Deleted `output_test_10/` (outdated test sample)
- Deleted `demo_output/` (old demo data)
- Deleted `demo_output_narrative/` (old demo data)
- Created clean `output_500/` directory
- Created clean `output_100k/` directory

---

### Task 3: Run LLM Narrative Generation 🔴 CRITICAL
**Status:** ⏳ Pending  
**Estimated Time:** 3-4 hours  
**Blocker:** Requires API key and credits

**Problem:** Current generated personas have **EMPTY narrative fields**!

```python
# Current state (generated without LLM):
persona.cultural_background = ""  # Empty!
persona.descriere_generala = ""   # Empty!
persona.profil_profesional = ""   # Empty!
persona.hobby_sport = ""          # Empty!
```

**Required Action:**
- [ ] Get Qwen API key (DashScope)
- [ ] Run `apps/demos/demo_narrative.py` or `async_narrative_generator.py`
- [ ] Generate for 500 sample first
- [ ] Validate quality against Reference benchmark
- [ ] Target: ~1,800 character cultural backgrounds

**Cost Estimate:** 
- 100K personas × ~5 narrative sections = ~500K API calls
- Qwen-turbo: ~$0.001/1K tokens
- **Total: ~$50-100**

---

## Phase 2: Benchmark-Inspired Improvements

### Task 4: Add Career Goals Field 🔴 HIGH PRIORITY
**Status:** ⏳ Pending  
**Estimated Time:** 1 hour  
**Motivation:** Reference has this; adds professional depth

**Implementation:**
1. Add field to `models.py`:
   ```python
   career_goals_and_ambitions: str = Field(default="", description="Career aspirations and future plans")
   ```
2. Add to `prompts.py` narrative generation
3. Include in LLM generation pipeline

**Example (Reference):**
> "Marcos pretende concluir o ensino médio e ingressar em um curso técnico em 
> mecatrônica... A médio prazo, deseja especializar-se em manutenção preditiva..."

---

### Task 5: Add Persona Summary (One-Liner) 🟡 MEDIUM PRIORITY
**Status:** ⏳ Pending  
**Estimated Time:** 30 minutes

**Field:** `persona_summary` - 100-200 character one-liner

**Implementation:**
- Generate via LLM or template
- Template: "{name} is a {age}-year-old {occupation} from {region} who..."

**Example (Reference):**
> "Marcos Antunes, operador de máquinas CNC e montador experiente em uma fábrica 
> de peças, combina sua organização metódica e espírito colaborativo..."

---

### Task 6: Add Narrative Skills/Hobbies 🟡 MEDIUM PRIORITY
**Status:** ⏳ Pending  
**Estimated Time:** 1 hour

**Current:** Only have `skills_and_expertise_list` (array)  
**Add:** `skills_and_expertise` (text narrative)  
**Same for:** hobbies_and_interests

**Example (Reference):**
> "Marcos possui sólida experiência como operador de instalações e máquinas...
> Destaca-se pela organização, planejamento de tarefas..."

---

## Phase 3: Validation & Testing

### Task 7: Regenerate 500-Sample Dataset
**Status:** ⏳ Pending  
**Estimated Time:** 2 hours  
**Depends On:** Tasks 1, 3, 4

**Steps:**
1. [ ] Generate 500 personas with fixed ethnicity
2. [ ] Run LLM narrative generation
3. [ ] Export to all formats (Parquet, JSON, CSV)
4. [ ] Generate statistics report
5. [ ] Validate distributions match NBS 2024

**Success Criteria:**
- Moldovan ethnicity: 76.7% ± 2%
- All age groups: NBS target ± 2%
- Narrative fields non-empty and high quality
- Cultural backgrounds: 1,500+ characters

---

### Task 8: Quality Benchmarking
**Status:** ⏳ Pending  
**Estimated Time:** 1 hour  
**Depends On:** Task 7

**Compare 500-sample against Reference:**
- [ ] Narrative length (target: ~1,800 chars for cultural_background)
- [ ] Professional detail (career goals quality)
- [ ] Skills/hobbies richness
- [ ] Overall persona coherence

**Document:** Create quality comparison report

---

### Task 9: Statistical Validation
**Status:** ⏳ Pending  
**Estimated Time:** 30 minutes  
**Depends On:** Task 7

**Steps:**
1. [ ] Run chi-square tests on 500-sample
2. [ ] Verify all p-values > 0.05
3. [ ] Check joint distributions:
   - Region × Ethnicity
   - Age × Education
   - Employment status × Age
4. [ ] Generate validation report

---

## Phase 4: 100K Generation

### Task 10: Generate 100K Dataset
**Status:** ⏳ Pending  
**Estimated Time:** 17-24 hours (with 10 LLM workers)  
**Depends On:** Tasks 1-9

**Configuration:**
- Target: 100,000 personas
- Checkpoint frequency: Every 1,000
- LLM workers: 10 (adjust based on API rate limits)
- Output directory: `output_100k/`

**Two-Stage Process:**

**Stage 1: Structured Generation (Fast)**
```python
# ~2 minutes for 100K
from moldova_personas.checkpoint import CheckpointStreamingGenerator
gen = CheckpointStreamingGenerator(checkpoint_manager=checkpoint_mgr)
personas = gen.generate_stream(100000, show_progress=True)
```

**Stage 2: Narrative Enrichment (Slow, Parallel)**
```python
# ~17 hours for 100K with 10 workers
from moldova_personas.async_narrative_generator import generate_narratives_parallel

results = generate_narratives_parallel(
    personas=personas,
    max_workers=10,
    provider='dashscope',
    model='qwen-turbo'
)
```

**Monitoring:**
- Watch checkpoint files created
- Monitor ethnicity/age distributions at each checkpoint
- Abort if statistical tests fail

---

### Task 11: Export 100K to All Formats
**Status:** ⏳ Pending  
**Estimated Time:** 30 minutes  
**Depends On:** Task 10

**Steps:**
1. [ ] Export to Parquet (primary format)
2. [ ] Export to JSONL (for streaming)
3. [ ] Export to CSV (for Excel users)
4. [ ] Generate statistics JSON
5. [ ] Generate statistics Markdown report

**Expected Output Files:**
- `moldova_personas_100k.parquet` (~100-200 MB)
- `moldova_personas_100k.jsonl` (~500 MB)
- `moldova_personas_100k.csv` (~300 MB)
- `moldova_personas_100k_stats.json`
- `moldova_personas_100k_stats.md`

---

## Phase 5: Finalization

### Task 12: Final Validation
**Status:** ⏳ Pending  
**Estimated Time:** 1 hour  
**Depends On:** Task 11

**Steps:**
1. [ ] Load 100K dataset and verify row count
2. [ ] Run full statistical test suite
3. [ ] Verify all distributions match NBS 2024
4. [ ] Check for missing/null fields
5. [ ] Validate narrative field completeness

---

### Task 13: Documentation Update
**Status:** ⏳ Pending  
**Estimated Time:** 2 hours

**Steps:**
1. [ ] Update README.md with final statistics
2. [ ] Document known limitations (18+ only, etc.)
3. [ ] Update API examples if needed
4. [ ] Create dataset card for 100K release
5. [ ] Update CHANGELOG.md
6. [ ] Update comparison with Reference

---

## Cost Estimation

### Compute Costs
| Item | Cost |
|------|------|
| Structured generation | Free (local CPU) |
| LLM narratives (100K × ~5 sections) | **$50-100** |
| Storage (1GB) | Negligible |
| **Total** | **~$50-100** |

### Time Estimation
| Phase | Tasks | Time |
|-------|-------|------|
| Phase 1 | Fixes | 6-10 hours |
| Phase 2 | Improvements | 2-3 hours |
| Phase 3 | Validation | 4-5 hours |
| Phase 4 | 100K Generation | 17-24 hours (automated) |
| Phase 5 | Finalization | 3 hours |
| **Total** | | **~32-45 hours** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM API rate limiting | Use checkpointing, retry logic, multiple workers |
| LLM API cost overrun | Test with 500 first, set budget limits |
| Ethnicity still off after fix | Validate with 1K before 100K |
| Power/internet failure | Checkpoint every 1000 personas |
| Disk space | Ensure 2GB free for 100K + checkpoints |
| Narrative quality below Reference | Iterate on prompts, use stronger model if needed |

---

## Current Status

```
Phase 1: Fixes
  [✅] Delete old generated data
  [⏳] Fix ethnicity underrepresentation  
  [⏳] Run LLM narrative generation
  
Phase 2: Improvements
  [⏳] Add career_goals_and_ambitions
  [⏳] Add persona_summary
  [⏳] Add narrative skills/hobbies

Phase 3: Validation
  [⏳] Regenerate 500-sample
  [⏳] Quality benchmarking vs Reference
  [⏳] Statistical validation

Phase 4: Generation
  [⏳] Generate 100K
  [⏳] Export formats

Phase 5: Finalization
  [⏳] Final validation
  [⏳] Documentation

Progress: 1/13 complete (8%)
```

---

## Next Actions (Priority Order)

### Immediate (This Session)
1. **Fix ethnicity sampling** - Implement IPF or adjust weights

### This Week
2. **Get LLM API access** - Qwen/DashScope credits
3. **Run narrative generation** - Populate empty fields
4. **Add career_goals field** - Model + prompt updates

### Next Week
5. **Regenerate 500-sample** - Full pipeline test
6. **Quality benchmarking** - Compare to Reference
7. **100K generation** - If all validations pass

---

## Reference Documents

- `NBS_2024_VERIFICATION_REPORT.md` - NBS data verification
- `COMPREHENSIVE_PROJECT_REVIEW.md` - Full project assessment
- `NBS_2024_IMPLEMENTATION_SUMMARY.md` - Recent changes summary
