import threading
import queue
from time import sleep

from ml.train import train_conv_step_batch

class TrainingExecutor:
    def __init__(self, model, max_workers=4):
        self.model = model
        self.task_queue = queue.Queue()
        self.max_workers = max_workers
        self.threads = []
        self.lock = threading.Lock()
        self._start_workers()

    def _start_workers(self):
        for _ in range(self.max_workers):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def _worker(self):
        while True:
            batch, replay_batch = self.task_queue.get()
            if self.model is None:
                break
            self._run_training(batch, replay_batch)
            self.task_queue.task_done()

    def _run_training(self, batch, replay_batch=True):
        if batch == None:
            return
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = batch
        if len(batch_states) < 1:
            return
        
        with self.lock:
            train_conv_step_batch(self.model, batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones)
            if not replay_batch:
                batch_states.clear()
                batch_next_states.clear()
                batch_actions.clear()
                batch_rewards.clear()
                batch_dones.clear()
            else:
                print('finish replay batch training')
        print ('done training')

    def submit(self, batch, replay_batch=True):
        
        self.task_queue.put((batch, replay_batch))
        print(f"queue size: {self.task_queue.qsize()}")

    def shutdown(self):
        sleep(20)
        self.model.save(self.model.name, overwrite=True)