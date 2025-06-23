import threading
import psutil
import GPUtil
import time

class CpuGpuMonitor:
    def __init__(self):
        self.cpu_data = []
        self.gpu_data = []
        self.flag = threading.Event()
        self.monitor_thread = None

    def start(self):
        self.flag.set()
        self.monitor_thread = threading.Thread(target=self.update_usage)
        self.monitor_thread.start()

    def end(self):
        # Set the flag to stop the monitoring thread
        self.flag.clear()
        # Wait for the monitoring thread to finish
        self.monitor_thread.join()
        # Print the average CPU and GPU usage
        cpu = self.get_cpu_average_usage()
        gpu = self.get_gpu_average_usage()
        print(f"Average CPU usage: {cpu}")
        print(f"Average GPU usage: {gpu}%")
        return cpu,gpu

    def update_usage(self):
        while self.flag.is_set():
            time.sleep(1)
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_data.append(cpu_percent)
            gpu_percent = GPUtil.getGPUs()[0].load * 100
            self.gpu_data.append(gpu_percent)

    def get_cpu_average_usage(self):
        return sum(self.cpu_data) / len(self.cpu_data) if len(self.cpu_data) > 0 else 0

    def get_gpu_average_usage(self):
        return sum(self.gpu_data) / len(self.gpu_data) if len(self.gpu_data) > 0 else 0