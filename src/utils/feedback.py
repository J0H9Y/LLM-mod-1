import json
from pathlib import Path
import datetime
from typing import Dict, Any, Optional

def log_feedback(
    log_file: Path,
    query: str,
    context: str,
    response: str,
    feedback: str, # e.g., 'thumbs_up', 'thumbs_down'
    comment: Optional[str] = None
):
    """Logs user feedback to a JSONL file."""
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "context": context,
        "response": response,
        "feedback": feedback,
        "comment": comment
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
