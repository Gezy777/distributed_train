from pynvml import *
import swanlab
import time

class GlobalGPUMonitor(threading.Thread):
    def __init__(self, gpu_indices=[0], interval=0.1):
        super().__init__()
        self.gpu_indices = gpu_indices  # 需要监控的 GPU 列表，例如 [0, 1]
        self.interval = interval
        self.stop_event = threading.Event()
        self.start_time = time.time()

        nvmlInit()

        # 为每块显卡获取句柄
        self.handles = [nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]

    def run(self):
        while not self.stop_event.is_set():
            for i, handle in enumerate(self.handles):
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                elapsed_time = time.time() - self.start_time

                swanlab.log({
                    f"GPU{i}_util": util.gpu,
                    f"GPU{i}_mem_used_mb": mem.used / 1024 / 1024,
                }, step=elapsed_time)

            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        nvmlShutdown()