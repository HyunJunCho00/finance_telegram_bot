from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from datetime import timezone

# Global singleton to be shared across all modules
# max_workers=20: Daily Precision (chart/context 6~8 threads) + 1min jobs (3) + hourly jobs (4) + 여유
# misfire_grace_time=30: mpf.plot 같은 CPU-bound 작업이 1분 job을 잠깐 밀어도 재실행 허용
_executors = {
    "default": ThreadPoolExecutor(max_workers=20),
}
_job_defaults = {
    "misfire_grace_time": 30,  # seconds (default=1 → missed 남발 방지)
    "coalesce": True,           # 밀린 동일 job 중복 실행 방지
}
scheduler = BackgroundScheduler(
    timezone=timezone.utc,
    executors=_executors,
    job_defaults=_job_defaults,
)
