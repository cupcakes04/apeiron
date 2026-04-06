import redis
import json
import time
import threading

class ProgressManager:
    """Manages task progress tracking using Redis for real-time monitoring.
    
    Provides a centralized system for tracking long-running tasks and subtasks
    with progress updates, status changes, and logging capabilities. Useful for
    web interfaces and multi-user environments.
    
    Args:
        user (str, optional): Default user identifier for tasks
        host (str): Redis server hostname. Default 'localhost'
        port (int): Redis server port. Default 6379
        expire_seconds (int): Task data expiration time. Default 86400 (24 hours)
    
    Task Structure:
        Each task stores:
        - user: User identifier
        - status: 'starting', 'running', or 'finished'
        - progress: Current progress value
        - total: Total steps (optional)
        - info: Status message (optional)
        - results: Task results (optional, can be dict)
    
    Example:
        >>> pm = ProgressManager(user='alice')
        >>> pm.start_task('Extract Features', total=100)
        >>> pm.update('Extract Features', progress=50, status='running')
        >>> pm.get('Extract Features')
        {'user': 'alice', 'status': 'running', 'progress': '50', 'total': '100', ...}
    """
    def __init__(self, user=None, host="localhost", port=6379, expire_seconds=86400):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.user = user
        self.expire = expire_seconds

    def _task_key(self, user, task_name, sub_task=None):
        if sub_task:
            return f"task:{user}:{task_name}:{sub_task}"
        return f"task:{user}:{task_name}"

    def _log_key(self, user, task_name, sub_task=None):
        if sub_task:
            return f"task:{user}:{task_name}:{sub_task}:logs"
        return f"task:{user}:{task_name}:logs"

    def start_task(self, task_name, user=None, sub_task=None, total="", info=""):
        """Initialize a new task or sub-task. If it exists, replace it."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")

        key = self._task_key(user, task_name, sub_task)

        # overwrite existing
        self.r.delete(key)
        self.r.delete(self._log_key(user, task_name, sub_task))

        self.r.hset(key, mapping={
            "user": user,
            "status": "starting",
            "progress": 0,
            "total": total,
            "info": info,
            "results": ""
        })
        self.r.expire(key, self.expire)
        self.r.expire(self._log_key(user, task_name, sub_task), self.expire)
        return sub_task or task_name
        
    def stop(self, user=None):
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")

        # Mark "stop" signal
        self.start_task("stop", user=user)

        # Find all tasks for this user
        keys = self.r.keys(f"task:{user}:*")
        for key in keys:
            if key.endswith(":logs") or key.endswith(":stop"):
                continue
            status = self.r.hget(key, "status")
            if status in ("starting", "running"):
                self.r.hset(key, "status", "finished")
                self.r.expire(key, self.expire)

    def update(self, task_name, progress=None, status=None, total=None,
               info=None, results=None, error=None, user=None, sub_task=None):
        """Update fields for a task or sub-task."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")
        
        if self.get('stop', user=user) != {}:
            self.reset('stop', user=user, remove=True)
            raise "Processing Stopped"

        key = self._task_key(user, task_name, sub_task)
        updates = {}
        if progress is not None: updates["progress"] = progress
        if status is not None: updates["status"] = status
        if total is not None: updates["total"] = total
        if info is not None: updates["info"] = info
        if error is not None: updates["error"] = error
        if results is not None: 
            updates["results"] = json.dumps(results) if isinstance(results, dict) else results
        if updates:
            self.r.hset(key, mapping=updates)
            self.r.expire(key, self.expire)
            
    def auto_progress(self, task_name, duration=10, steps=100, user=None, sub_task=None, info=""):
        """
        Auto-update progress from 0 to 100 for given duration.
        - duration: total time in seconds
        - steps: number of increments (default = 100, so 1% each step)
        """
        def run():
            for i in range(steps + 1):
                pct = int((i / steps) * 100)
                self.update(
                    task_name,
                    progress=pct,
                    status="running" if pct < 100 else "finished",
                    total=steps,
                    info=info,
                    user=user,
                    sub_task=sub_task
                )
                time.sleep(duration / steps)

        # run in background so it doesn't block caller
        threading.Thread(target=run, daemon=True).start()
    
    def reset(self, task_name, user=None, sub_task=None, remove=False):
        """Reset a task or sub-task to initial empty/default state."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")

        key = self._task_key(user, task_name, sub_task)
        if remove:
            self.r.delete(key)
        else:
            updates = {
                "user": user,
                "status": "starting",
                "progress": 0,
                "total": "",
                "info": "",
                "results": ""
            }
            self.r.hset(key, mapping=updates)
            self.r.expire(key, self.expire)
            

    def get(self, task_name, user=None, sub_task=None):
        """Retrieve all info for a task or sub-task."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")
        key = self._task_key(user, task_name, sub_task)
        return self.r.hgetall(key)

    def add_log(self, task_name, message, user=None, sub_task=None):
        """Append a log line for a task or sub-task."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")

        log_key = self._log_key(user, task_name, sub_task)
        self.r.rpush(log_key, message)
        self.r.expire(log_key, self.expire)

    def get_logs(self, task_name, start=0, end=-1, user=None, sub_task=None):
        """Retrieve logs for a task or sub-task."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")
        log_key = self._log_key(user, task_name, sub_task)
        return self.r.lrange(log_key, start, end)

    def list_tasks(self, user=None):
        """Return all task names (including sub-tasks) for this user."""
        user = user or self.user
        if not user:
            raise ValueError("User must be specified either at init or here")
        keys = self.r.keys(f"task:{user}:*")
        # returns things like task:alice:upload:part1 → ["upload:part1"]
        return [":".join(k.split(":")[2:]) for k in keys if not k.endswith(":logs")]