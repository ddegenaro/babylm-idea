import csv

from transformers import TrainerCallback

class TsvLossCallback(TrainerCallback):
    def __init__(self, output_path: str):
        self.output_path = output_path
        self._writer = None
        self._file = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.writer(self._file, delimiter="\t")
        self._writer.writerow(["step", "epoch", "loss", "eval_loss"])
        self._file.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = [
            state.global_step,
            round(state.epoch, 4) if state.epoch is not None else "",
            logs.get("loss", ""),
            logs.get("eval_loss", ""),
        ]
        self._writer.writerow(row)
        self._file.flush()  # write immediately so the file is readable during training

    def on_train_end(self, args, state, control, **kwargs):
        if self._file:
            self._file.close()