"""Tests for data freshness monitoring."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from moldova_personas.data_freshness import (
    DataFreshnessMonitor,
    FreshnessReport,
    FreshnessStatus,
    DistributionFreshness,
    check_data_freshness,
)


class TestFreshnessStatus:
    """Tests for freshness status logic."""
    
    def test_fresh_status(self):
        """Test fresh status (< 7 days)."""
        monitor = DataFreshnessMonitor()
        status = monitor._get_freshness_status(timedelta(days=3))
        assert status == FreshnessStatus.FRESH
    
    def test_aging_status(self):
        """Test aging status (7-30 days)."""
        monitor = DataFreshnessMonitor()
        status = monitor._get_freshness_status(timedelta(days=15))
        assert status == FreshnessStatus.AGING
    
    def test_stale_status(self):
        """Test stale status (30-60 days)."""
        monitor = DataFreshnessMonitor()
        status = monitor._get_freshness_status(timedelta(days=45))
        assert status == FreshnessStatus.STALE
    
    def test_critical_status(self):
        """Test critical status (> 60 days)."""
        monitor = DataFreshnessMonitor()
        status = monitor._get_freshness_status(timedelta(days=90))
        assert status == FreshnessStatus.CRITICAL


class TestFreshnessReport:
    """Tests for freshness report."""
    
    def test_report_summary(self):
        """Test report summary calculations."""
        distributions = [
            DistributionFreshness("dist1", 5, FreshnessStatus.FRESH, datetime.now()),
            DistributionFreshness("dist2", 15, FreshnessStatus.AGING, datetime.now()),
            DistributionFreshness("dist3", 45, FreshnessStatus.STALE, datetime.now()),
            DistributionFreshness("dist4", 90, FreshnessStatus.CRITICAL, datetime.now()),
        ]
        
        report = FreshnessReport(distributions=distributions)
        
        assert report.stale_count == 2  # STALE + CRITICAL
        assert report.critical_count == 1
        assert report.fresh_percentage == 25.0
    
    def test_get_by_status(self):
        """Test filtering by status."""
        distributions = [
            DistributionFreshness("fresh", 3, FreshnessStatus.FRESH, datetime.now()),
            DistributionFreshness("stale", 45, FreshnessStatus.STALE, datetime.now()),
        ]
        
        report = FreshnessReport(distributions=distributions)
        
        fresh_list = report.get_by_status(FreshnessStatus.FRESH)
        assert len(fresh_list) == 1
        assert fresh_list[0].name == "fresh"


class TestDataFreshnessMonitor:
    """Tests for DataFreshnessMonitor."""
    
    def test_should_alert_high_stakes(self):
        """Test alert logic for high stakes profile."""
        monitor = DataFreshnessMonitor()
        
        distributions = [
            DistributionFreshness("dist", 10, FreshnessStatus.AGING, datetime.now()),
        ]
        report = FreshnessReport(distributions=distributions)
        
        # High stakes: alert if > 7 days
        assert monitor.should_alert(report, "high_stakes") == True
        
        # Analysis only: alert only if > 60 days
        assert monitor.should_alert(report, "analysis_only") == False
    
    def test_should_alert_analysis_only(self):
        """Test alert logic for analysis only profile."""
        monitor = DataFreshnessMonitor()
        
        distributions = [
            DistributionFreshness("dist", 70, FreshnessStatus.CRITICAL, datetime.now()),
        ]
        report = FreshnessReport(distributions=distributions)
        
        # Analysis only: alert if > 60 days
        assert monitor.should_alert(report, "analysis_only") == True


class TestCheckDataFreshness:
    """Tests for check_data_freshness function."""
    
    @patch('moldova_personas.data_freshness.DataFreshnessMonitor')
    def test_check_data_freshness(self, mock_monitor_class):
        """Test the convenience function."""
        mock_monitor = Mock()
        mock_report = Mock()
        mock_report.stale_count = 0
        mock_report.distributions = []
        mock_monitor.check_all.return_value = mock_report
        mock_monitor.should_alert.return_value = False
        mock_monitor_class.return_value = mock_monitor
        
        report = check_data_freshness("high_stakes")
        
        assert report == mock_report
        mock_monitor.check_all.assert_called_once()
