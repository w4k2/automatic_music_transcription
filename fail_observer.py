from sacred.observers.base import RunObserver
from snapshot import Snapshot

class FailObserver(RunObserver):
    def __init__(self, logdir, snapshot_capacity, TOTAL_DEBUG=False):
        self.snapshot = Snapshot(logdir, snapshot_capacity, TOTAL_DEBUG)
        print("Fail or interruption observer is initialized!")

    def interrupted_event(self, interrupt_time, status):
        print(f"Status: {status}")
        print(f"Training interrupted at {interrupt_time} - gathering snapshot!")
        self.snapshot.trigger_snapshot()

    def failed_event(self, fail_time, fail_trace):
        print("Training failed - gathering snapshot!")
        self.snapshot.trigger_snapshot()
