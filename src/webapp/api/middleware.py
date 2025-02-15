import logging
from collections import defaultdict
from datetime import datetime
from fastapi import Request
from ..properties import Properties

logger = logging.getLogger("uvicorn")


daily_call_counts = defaultdict(int)


class DailyLimitExceededException(Exception):
    def __init__(self, detail: str):
        self.detail = detail


async def track_daily_calls(request: Request):
    today_date = datetime.today().strftime('%Y-%m-%d')

    if today_date not in daily_call_counts:
        daily_call_counts[today_date] = 0

    daily_call_counts[today_date] += 1
    logger.info("Date: %s, Calls: %s", today_date, daily_call_counts[today_date])

    if daily_call_counts[today_date] > Properties.call_limit:
        raise DailyLimitExceededException("Poll-E received too many requests today. Time to rest! Please try again "
                                          "tomorrow.")


