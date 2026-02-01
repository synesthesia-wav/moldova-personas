# Production Readiness TODO

## Critical Issues (Fix Before Production)

### 1. Fix PxWeb 2.0 Query Format
- [ ] Investigate correct PxWeb 2.0 API query format
- [ ] Update `_make_request` to use proper POST body structure
- [ ] Test with actual API calls to verify data retrieval
- [ ] Update all 10 distribution fetchers to use correct queries

### 2. Add Unit Tests for pxweb_fetcher Module
- [ ] Test `DataProvenance` enum values
- [ ] Test `Distribution` class (validation, normalization, serialization)
- [ ] Test `JSONStatParser` with mock JSON-stat data
- [ ] Test `IPFEngine` with known marginal distributions
- [ ] Test `NBSDataManager` cache operations
- [ ] Test fallback mechanism when API fails

### 3. Verify Dataset Paths
- [ ] Confirm correct API paths for all 10 distributions
- [ ] Update `DATASETS` dictionary with verified paths
- [ ] Test each dataset fetch individually

## High Priority (Strongly Recommended)

### 4. Complete IPF Integration
- [ ] Implement `get_cross_tabulation_ipf()` method
- [ ] Replace hardcoded `REGION_URBAN_CROSS` with IPF-derived
- [ ] Replace hardcoded `ETHNICITY_BY_REGION` with IPF-derived
- [ ] Add confidence intervals for IPF-derived cross-tabs

### 5. Add Retry Logic
- [ ] Implement exponential backoff for API failures
- [ ] Add circuit breaker pattern for repeated failures
- [ ] Log retry attempts for debugging

### 6. Error Handling Improvements
- [ ] Distinguish between transient and permanent errors
- [ ] Better error messages for users
- [ ] Health check endpoint for data freshness

## Medium Priority (Nice to Have)

### 7. Performance Optimizations
- [ ] Implement async fetching for multiple distributions
- [ ] Add cache compression for large datasets
- [ ] Batch PxWeb requests where possible

### 8. Monitoring & Observability
- [ ] Add metrics for API call success/failure rates
- [ ] Track cache hit/miss ratios
- [ ] Log data provenance for audit trails

### 9. Documentation
- [ ] Add architecture diagram to docs
- [ ] Document data provenance schema
- [ ] Write "Adding New Distributions" guide

## Completed ✅

- [x] Remove fabricated village names
- [x] Remove template-based cultural backgrounds
- [x] Fix hardcoded 70% probabilities with census sources
- [x] Mark estimated employment data clearly
- [x] Document religious affiliations as estimated
- [x] Make mock mode safe (empty strings)
- [x] Implement PxWeb fetching infrastructure
- [x] Add Distribution dataclass with provenance
- [x] Create IPF engine skeleton
- [x] Update census_data.py with lazy loading
- [x] All 101 tests passing

---

## Current Status

**Tests**: 101/101 passing ✅  
**PxWeb Integration**: Architecture complete, query format needs fixing ⚠️  
**Data Quality**: All fabricated data removed ✅  
**Scientific Rigor**: Provenance tracking implemented ✅  

**Estimated Time to Production**: 2-3 days (fixing query format + adding tests)
