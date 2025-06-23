import logging
import os
from datetime import datetime

class ModelLogger:
    def __init__(self, log_path='logs/', log_filename='model_performance.log'):
        """
        初始化日志类，并设置日志文件路径与格式。
        """
        # 创建日志目录（如果不存在）
        os.makedirs(log_path, exist_ok=True)

        # 配置日志记录器
        self.logger = logging.getLogger('ModelLogger')
        self.logger.setLevel(logging.INFO)

        # 设置日志文件的完整路径
        log_file = os.path.join(log_path, log_filename)

        # 创建文件处理器并设置格式
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 防止重复添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log_accuracy(self, model_name: str, accuracy: float):
        """
        记录模型名称及其准确率到日志文件中。
        """
        self.logger.info(f"model: {model_name} | ACC: {accuracy:.2f}%")

    def log_custom_message(self, message: str):
        """
        记录自定义信息到日志文件中。
        """
        self.logger.info(message)

# 示例用法
if __name__ == "__main__":
    # 创建日志类实例
    logger = ModelLogger(log_path='model_logs/', log_filename='performance.log')

    # 记录模型准确率
    logger.log_accuracy('ResNet-50', 92.57)
    logger.log_accuracy('VGG-16', 88.32)

    # 记录自定义信息
    logger.log_custom_message("开始训练下一个模型...")

