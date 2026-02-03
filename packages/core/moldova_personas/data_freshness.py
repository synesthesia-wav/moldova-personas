"""
Data Freshness Monitoring

Provides proactive monitoring of cached data freshness with alerts
when distributions exceed age thresholds.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from .pxweb_fetcher import NBSDataManager, DataProvenance


logger = logging.getLogger(__name__)


class FreshnessStatus(Enum):
    """Freshness status levels."""
    FRESH = "fresh"              # < 7 days
    AGING = "aging"              # 7-30 days
    STALE = "stale"              # > 30 days
    CRITICAL = "critical"        # > 60 days (HIGH_STAKES threshold)


@dataclass
class DistributionFreshness:
    """Freshness report for a single distribution."""
    name: str
    age_days: float
    status: FreshnessStatus
    last_fetched: Optional[datetime]
    provenance: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "age_days": round(self.age_days, 1),
            "status": self.status.value,
            "last_fetched": self.last_fetched.isoformat() if self.last_fetched else None,
            "provenance": self.provenance,
        }


@dataclass
class FreshnessReport:
    """Complete freshness report for all distributions."""
    timestamp: datetime = field(default_factory=datetime.now)
    distributions: List[DistributionFreshness] = field(default_factory=list)
    
    @property
    def stale_count(self) -> int:
        """Number of stale distributions."""
        return sum(1 for d in self.distributions if d.status in [
            FreshnessStatus.STALE, FreshnessStatus.CRITICAL
        ])
    
    @property
    def critical_count(self) -> int:
        """Number of critically stale distributions (>60 days)."""
        return sum(1 for d in self.distributions if d.status == FreshnessStatus.CRITICAL)
    
    @property
    def fresh_percentage(self) -> float:
        """Percentage of fresh distributions."""
        if not self.distributions:
            return 100.0
        fresh = sum(1 for d in self.distributions if d.status == FreshnessStatus.FRESH)
        return (fresh / len(self.distributions)) * 100
    
    def get_by_status(self, status: FreshnessStatus) -> List[DistributionFreshness]:
        """Get all distributions with given status."""
        return [d for d in self.distributions if d.status == status]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "distributions": [d.to_dict() for d in self.distributions],
            "summary": {
                "total": len(self.distributions),
                "stale_count": self.stale_count,
                "critical_count": self.critical_count,
                "fresh_percentage": round(self.fresh_percentage, 1),
            }
        }
    
    def __str__(self) -> str:
        lines = [
            f"Data Freshness Report ({self.timestamp.isoformat()})",
            f"Total distributions: {len(self.distributions)}",
            f"Fresh: {self.fresh_percentage:.1f}%",
            f"Stale: {self.stale_count}",
            f"Critical: {self.critical_count}",
        ]
        if self.critical_count > 0:
            lines.append("\nCritical (refresh recommended):")
            for d in self.get_by_status(FreshnessStatus.CRITICAL):
                lines.append(f"  - {d.name}: {d.age_days:.0f} days old")
        return "\n".join(lines)


class DataFreshnessMonitor:
    """
    Monitor data freshness with configurable thresholds.
    
    Usage:
        monitor = DataFreshnessMonitor()
        report = monitor.check_all()
        
        if report.critical_count > 0:
            alert_engineer(report)
    """
    
    # Distribution names to monitor
    MONITORED_DISTRIBUTIONS = [
        "sex",
        "age_group",
        "region",
        "ethnicity",
        "education",
        "marital_status",
        "religion",
        "employment_status",
        "residence_type",
    ]
    
    def __init__(
        self,
        data_manager: Optional[NBSDataManager] = None,
        warning_threshold_days: int = 7,
        stale_threshold_days: int = 30,
        critical_threshold_days: int = 60,
    ):
        """
        Initialize freshness monitor.
        
        Args:
            data_manager: NBSDataManager instance (creates default if None)
            warning_threshold_days: Days before "aging" status (default: 7)
            stale_threshold_days: Days before "stale" status (default: 30)
            critical_threshold_days: Days before "critical" status (default: 60)
        """
        self.data_manager = data_manager or NBSDataManager()
        self.warning_threshold = timedelta(days=warning_threshold_days)
        self.stale_threshold = timedelta(days=stale_threshold_days)
        self.critical_threshold = timedelta(days=critical_threshold_days)
    
    def _get_freshness_status(self, age: timedelta) -> FreshnessStatus:
        """Determine freshness status from age."""
        if age > self.critical_threshold:
            return FreshnessStatus.CRITICAL
        elif age > self.stale_threshold:
            return FreshnessStatus.STALE
        elif age > self.warning_threshold:
            return FreshnessStatus.AGING
        else:
            return FreshnessStatus.FRESH
    
    def check_distribution(self, name: str) -> DistributionFreshness:
        """Check freshness of a single distribution."""
        try:
            dist = self.data_manager.get_distribution(name)
            
            if dist.last_fetched:
                age = datetime.now() - dist.last_fetched
                status = self._get_freshness_status(age)
                return DistributionFreshness(
                    name=name,
                    age_days=age.total_seconds() / 86400,
                    status=status,
                    last_fetched=dist.last_fetched,
                    provenance=dist.provenance.value if dist.provenance else None,
                )
            else:
                # No timestamp - treat as stale
                return DistributionFreshness(
                    name=name,
                    age_days=float('inf'),
                    status=FreshnessStatus.STALE,
                    last_fetched=None,
                    provenance=dist.provenance.value if dist.provenance else None,
                )
        except Exception as e:
            logger.warning(f"Failed to check freshness for {name}: {e}")
            return DistributionFreshness(
                name=name,
                age_days=float('inf'),
                status=FreshnessStatus.CRITICAL,
                last_fetched=None,
                provenance=None,
            )
    
    def check_all(self) -> FreshnessReport:
        """Check freshness of all monitored distributions."""
        distributions = []
        for name in self.MONITORED_DISTRIBUTIONS:
            freshness = self.check_distribution(name)
            distributions.append(freshness)
            
        return FreshnessReport(distributions=distributions)
    
    def should_alert(self, report: FreshnessReport, profile: str = "high_stakes") -> bool:
        """
        Determine if alert should be sent based on profile.
        
        Args:
            report: Freshness report
            profile: Use case profile ("analysis_only", "narrative_required", "high_stakes")
            
        Returns:
            True if alert should be sent
        """
        thresholds = {
            "analysis_only": 60,   # Alert if any critical
            "narrative_required": 30,  # Alert if any stale
            "high_stakes": 7,      # Alert if any aging
        }
        
        threshold_days = thresholds.get(profile, 30)
        
        for dist in report.distributions:
            if dist.age_days > threshold_days:
                return True
        return False


def check_data_freshness(profile: str = "high_stakes") -> FreshnessReport:
    """
    Quick check of data freshness.
    
    Args:
        profile: Use case profile for alert threshold
        
    Returns:
        Freshness report
    """
    monitor = DataFreshnessMonitor()
    report = monitor.check_all()
    
    if monitor.should_alert(report, profile):
        logger.warning(f"Data freshness alert for {profile}: {report.stale_count} stale distributions")
    
    return report
