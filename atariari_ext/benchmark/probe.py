from atariari.benchmark.probe import ProbeTrainer


class WandbLoggingProbeTrainer(ProbeTrainer):

    def log_train_epoch(self, epoch_idx, dictionaries):
        # Only log validation accuracy metrics for linear probes
        for dictionary in dictionaries:
            filtered = {k: v for k, v in dictionary.items() if 'val' in k and 'acc' in k}

            if filtered:
                self.wandb.log(filtered, step=epoch_idx)
    
    def log_test_epoch(self, *dictionaries):
        for dictionary in dictionaries:
            self.wandb.run.summary.update(dictionary)

    def log_results(self, epoch_idx, *dictionaries):
        super().log_results(epoch_idx, *dictionaries)

        if self.wandb is None:
            return

        if epoch_idx == "Test":
            self.log_test_epoch(*dictionaries)
        elif epoch_idx is not None:
            self.log_train_epoch(epoch_idx, dictionaries)
