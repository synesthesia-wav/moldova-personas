#!/bin/bash
# Monitor the generation process and report when complete

LOG_FILE="/Users/victorvanica/Coding Projects/moldova-personas/generation_500.log"
PID=14637

echo "🔍 Monitoring generation process (PID: $PID)..."
echo "Started at: $(date)"
echo ""

while true; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo ""
        echo "✅ GENERATION COMPLETE!"
        echo "Finished at: $(date)"
        echo ""
        echo "=== Final Results ==="
        tail -30 "$LOG_FILE"
        echo ""
        echo "=== Output Files ==="
        ls -la /Users/victorvanica/Coding\ Projects/moldova-personas/output_500_personas/ 2>/dev/null || \
        ls -la /Users/victorvanica/Coding\ Projects/moldova-personas/*.parquet 2>/dev/null || \
        echo "Checking for output files..."
        break
    fi
    
    # Show current progress every 2 minutes
    CURRENT=$(tail -1 "$LOG_FILE" | grep -o '[0-9]*/500' | tail -1 | cut -d'/' -f1)
    if [ -n "$CURRENT" ]; then
        PERCENT=$((CURRENT * 100 / 500))
        echo "$(date): Progress $CURRENT/500 ($PERCENT%)"
    fi
    
    sleep 60
done
