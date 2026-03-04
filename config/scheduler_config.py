from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from config.settings import TradingMode
from loguru import logger

# Global singleton to be shared across all modules
scheduler = BackgroundScheduler()

def _build_analysis_trigger(mode: TradingMode):
    """Build APScheduler trigger based on trading mode.
    - DAY_TRADING: every 1 hour
    - SWING: every 8 hours (00:00, 08:00, 16:00 UTC)
    - POSITION: once daily at 00:00 UTC (= 09:00 KST)
    """
    if mode == TradingMode.POSITION:
        return CronTrigger(hour=0, minute=0)
    else:
        return CronTrigger(hour='0,8,16', minute=0)

def reschedule_analysis_job(new_mode: str):
    """Dynamically update the analysis job frequency when mode changes."""
    try:
        mode_enum = TradingMode(new_mode.lower())
        new_trigger = _build_analysis_trigger(mode_enum)
        
        # Ensure job exists before rescheduling
        job = scheduler.get_job('job_analysis')
        if job:
            scheduler.reschedule_job('job_analysis', trigger=new_trigger)
            logger.success(f"Analysis job rescheduled for {new_mode.upper()} mode triggers.")
        else:
            logger.warning(f"Could not reschedule: job_analysis not found in this scheduler instance.")
    except Exception as e:
        logger.error(f"Failed to reschedule analysis job: {e}")
