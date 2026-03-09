from apscheduler.schedulers.background import BackgroundScheduler
from datetime import timezone

# Global singleton to be shared across all modules
scheduler = BackgroundScheduler(timezone=timezone.utc)
