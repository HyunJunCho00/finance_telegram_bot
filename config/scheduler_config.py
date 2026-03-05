from apscheduler.schedulers.background import BackgroundScheduler

# Global singleton to be shared across all modules
scheduler = BackgroundScheduler()
