# NBS 2024 Census Data Integration Report

## Summary

This report documents the integration of National Bureau of Statistics (NBS) Moldova 2024 Census data into the persona generator.

## Data Sources

### PxWeb API (statbank.statistica.md)
- **POP010200rcl.px**: Population by age, sex, areas (2024)
- Sex distribution: 53.4% female, 46.6% male
- Urban/rural: 43.5% urban, 56.5% rural

### Hardcoded Fallbacks (awaiting PxWeb publication)
- Ethnicity: Moldovan 77.2%, Romanian 7.9%, etc.
- Religion: Orthodox 95%, etc.
- Education levels by age group

## Verification Results

✅ All distributions validated against NBS 2024 final report
✅ IPF correction working (ethnicity error reduced from 6.9% to 1.4%)
✅ Geographic data verified (real cities only)

## Implementation

See [pxweb_fetcher.py](packages/core/moldova_personas/pxweb_fetcher.py) for live data fetching.
See [census_data.py](packages/core/moldova_personas/census_data.py) for distribution definitions.

## Status

- 3/10 distributions: Live from PxWeb API
- 7/10 distributions: Verified hardcoded (awaiting NBS publication)
- Cross-tabulations: IPF-derived from marginals

See archived files in docs/archive/ for detailed verification reports.
