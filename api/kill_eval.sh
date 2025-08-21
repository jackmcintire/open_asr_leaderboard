#!/bin/bash

# Kill all run_eval.py processes
echo "Killing all run_eval.py processes..."
pkill -f "run_eval.py"

# Wait a moment
sleep 2

# Check if any are still running
remaining=$(pgrep -f "run_eval.py" | wc -l)
if [ $remaining -gt 0 ]; then
    echo "Some processes still running, using SIGKILL..."
    pkill -9 -f "run_eval.py"
else
    echo "All run_eval.py processes terminated."
fi

# Show any remaining python processes
echo ""
echo "Remaining Python processes:"
ps aux | grep python | grep -v grep
