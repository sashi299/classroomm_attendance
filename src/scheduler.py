import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BackgroundScheduler:
    """Simple thread-based scheduler for checking completed periods and triggering notifications."""

    def __init__(self, notification_manager, interval_seconds=60):
        self.notification_manager = notification_manager
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="NotificationScheduler", daemon=True)
        self._thread.start()
        logger.info("Notification Background Scheduler started (interval: %ds).", self.interval)

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        logger.info("Notification Background Scheduler stopped.")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                # Process pending notifications
                self.notification_manager.process_pending_notifications()
            except Exception as e:
                logger.error("Error in notification scheduler loop: %s", e)

            # Wait for next interval or stop event
            self._stop_event.wait(self.interval)
