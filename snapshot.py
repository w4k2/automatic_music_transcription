import os
import torch


class Snapshot:
    def __init__(self, logdir, snapshot_capacity, TOTAL_DEBUG=False):
        self.TOTAL_DEBUG = TOTAL_DEBUG
        self.logdir = logdir
        self.snapshot_capacity = snapshot_capacity
        self.basic_info_queue = []
        self.model_queue = []
        self.optimizer_queue = []
        self.scheduler_queue = []
        self.loss_monitor = []
        self.is_in_warning=False
        self.snapshot_triggered=False
        self.number_of_looses_taken_into_consideration_in_warning_state_detection = 10
        self.warning_snapshot_capa = 20
        self.internal_counter = self.warning_snapshot_capa - snapshot_capacity
        self.snapshot_dir = os.path.join(self.logdir, "SNAPSHOT")
        os.makedirs(self.snapshot_dir)
        assert self.snapshot_capacity < self.warning_snapshot_capa

    def is_in_warning_based_on_loss(self, current_loss):
        if (len(self.loss_monitor)>30):
            sum_of_last_losses = 0
            for previous_loss in self.loss_monitor[-self.number_of_looses_taken_into_consideration_in_warning_state_detection:]:
                sum_of_last_losses += previous_loss
            average_losses = sum_of_last_losses/self.number_of_looses_taken_into_consideration_in_warning_state_detection
            if current_loss > (average_losses*25):
                self.snapshot_capacity = self.warning_snapshot_capa
                print(f"Anomaly detected - loss increased 25 times - activating warning mode and increasing snapshot capacity to {self.snapshot_capacity}")
                return True
        return False

    def add_to_snapshot(self,
                        current_epoch,
                        current_loss,
                        basic_info,
                        model_state_dict,
                        optimizer_state_dict,
                        scheduler_state_dict):
        self.loss_monitor.append(current_loss)
        if not self.is_in_warning:
            self.is_in_warning = self.is_in_warning_based_on_loss(current_loss)
        else:
            self.internal_counter -= 1
            if(self.internal_counter <= 0):
                self.trigger_snapshot()
        # don't remove models in TOTAL_DEBUG mode
        if not self.TOTAL_DEBUG:
            if(len(self.basic_info_queue) >= self.snapshot_capacity):
                del self.basic_info_queue[0]
            if(len(self.model_queue) >= self.snapshot_capacity):
                os.remove(self.model_queue[0])
                del self.model_queue[0]
            if(len(self.optimizer_queue) >= self.snapshot_capacity):
                os.remove(self.optimizer_queue[0])
                del self.optimizer_queue[0]
            if(len(self.scheduler_queue) >= self.snapshot_capacity):
                os.remove(self.scheduler_queue[0])
                del self.scheduler_queue[0]

        self.basic_info_queue.append(basic_info)
        model_path = os.path.join(self.snapshot_dir, f"{current_epoch}_model.pt")
        torch.save(model_state_dict, model_path)
        self.model_queue.append(model_path)

        optimizer_path = os.path.join(self.snapshot_dir, f"{current_epoch}_optimizer.pt")
        torch.save(optimizer_state_dict, optimizer_path)
        self.optimizer_queue.append(optimizer_path)

        scheduler_path = os.path.join(self.snapshot_dir, f"{current_epoch}_scheduler.pt")
        torch.save(scheduler_state_dict, scheduler_path)
        self.scheduler_queue.append(scheduler_path)

    def change_capacity(self, snapshot_capacity):
        self.snapshot_capacity = snapshot_capacity

    def trigger_snapshot(self):
        print("Triggering snapshot based on observations of loss!")
        self.snapshot_triggered=True
        assert len(self.basic_info_queue) == len(self.model_queue)
        assert len(self.model_queue) == len(self.optimizer_queue)

        with open(os.path.join(self.snapshot_dir, "log.txt"), "w") as logfile:
            for i, basic_info in enumerate(self.basic_info_queue):
                logfile.write(basic_info+"\n")
        with open(os.path.join(self.snapshot_dir, "loss_history.txt"), "w") as lossfile:
            for i, loss in enumerate(self.loss_monitor):
                lossfile.write(f"{i}: {loss:.6f} \n")

