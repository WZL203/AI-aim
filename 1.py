import sys
import ctypes
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import time
import threading
import cv2
import numpy as np
import pygetwindow as gw
import traceback

# 尝试导入ONNX Runtime GPU版本
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
    
    # 检测GPU可用性
    available_providers = ort.get_available_providers()
    print(f"✅ 可用执行提供者: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        print("🎮 GPU加速可用 - 使用CUDA")
        PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("⚠️ GPU不可用 - 使用CPU")
        PROVIDERS = ['CPUExecutionProvider']
        
except ImportError as e:
    ONNXRUNTIME_AVAILABLE = False
    print(f"❌ ONNX Runtime 导入失败: {e}")

# 提升到管理员权限
def run_as_admin():
    if ctypes.windll.shell32.IsUserAnAdmin():
        return True
    params = ' '.join(sys.argv[1:])
    cmd = f'"{sys.executable}" "{os.path.realpath(__file__)}" {params}'
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, cmd, None, 1)
    sys.exit(0)

# 导入MSS库
try:
    import mss
    MSS_AVAILABLE = True
    print("✓ MSS 可用")
except ImportError:
    MSS_AVAILABLE = False
    print("❌ MSS 不可用")

if __name__ == "__main__":
    if not run_as_admin():
        print("请求管理员权限...")
    else:
        print("已以管理员权限运行")

        class HighPerformanceAI:
            """高性能AI - GPU加速版本"""
            def __init__(self, config):
                self.config = config
                self.is_running = False
                self.thread = None
                self.game_window = None
                self.exact_window_title = config.get("exact_window_title", "")
                self.emergency_stop = False
                self.last_right_click_state = False
                
                # 模型尺寸参数
                self.model_input_size = 416
                self.capture_size = 416
                self.center_x = 208
                self.center_y = 208
                
                # 性能优化参数
                self.smoothing_factor = 0.3
                self.last_dx = 0
                self.last_dy = 0
                
                # ONNX模型加载 - GPU版本
                self.session = None
                if not ONNXRUNTIME_AVAILABLE:
                    print("❌ ONNX Runtime 不可用，无法加载模型")
                    return
                
                try:
                    # 使用你提供的准确路径
                    onnx_model_path = "C:/Users/Administrator/Downloads/Thefinals.onnx"
                    
                    print(f"📁 尝试加载模型: {onnx_model_path}")
                    
                    # 检查文件是否存在
                    if not os.path.exists(onnx_model_path):
                        print(f"❌ 模型文件不存在: {onnx_model_path}")
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        current_dir_model = os.path.join(current_dir, "Thefinals.onnx")
                        if os.path.exists(current_dir_model):
                            onnx_model_path = current_dir_model
                            print(f"✅ 在当前目录找到模型: {current_dir_model}")
                        else:
                            raise FileNotFoundError("未找到模型文件")
                    
                    print(f"✅ 模型文件存在: {onnx_model_path}")
                    print(f"📏 文件大小: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")
                    
                    # GPU优化配置
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    print("🔄 正在加载ONNX模型...")
                    self.session = ort.InferenceSession(onnx_model_path, sess_options, providers=PROVIDERS)
                    
                    # 获取输入输出信息
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name
                    print(f"📊 输入名称: {self.input_name}")
                    print(f"📊 输出名称: {self.output_name}")
                    
                    input_shape = self.session.get_inputs()[0].shape
                    output_shape = self.session.get_outputs()[0].shape
                    print(f"📊 输入形状: {input_shape}")
                    print(f"📊 输出形状: {output_shape}")
                    
                    # 根据模型输入形状调整参数
                    if len(input_shape) == 4:
                        self.model_input_size = input_shape[2]
                        self.capture_size = self.model_input_size
                        self.center_x = self.capture_size // 2
                        self.center_y = self.capture_size // 2
                        print(f"🔄 调整模型尺寸为: {self.model_input_size}x{self.model_input_size}")
                    
                    # 预热模型
                    print("🔥 预热模型...")
                    dummy_input = np.random.randn(1, 3, self.model_input_size, self.model_input_size).astype(np.float32)
                    
                    for i in range(3):
                        start_time = time.perf_counter()
                        _ = self.session.run([self.output_name], {self.input_name: dummy_input})
                        warmup_time = (time.perf_counter() - start_time) * 1000
                        print(f"  预热 {i+1}/3: {warmup_time:.1f}ms")
                    
                    print("✅ ONNX模型加载和预热完成")
                    
                    # 检查实际使用的提供者
                    actual_provider = self.session.get_providers()[0]
                    if 'CUDA' in actual_provider:
                        print("🎮 GPU加速已启用!")
                    else:
                        print("⚡ CPU模式运行")
                    
                except Exception as e:
                    print(f"❌ ONNX模型加载失败: {e}")
                    print("💡 详细错误信息:")
                    traceback.print_exc()
                    self.session = None
                
                # 性能参数 - GPU优化
                self.target_fps = 120  # GPU可以支持更高FPS
                self.detection_interval = 1 / 60  # 60FPS检测
                self.move_interval = 1 / 120      # 120FPS移动
                self.last_detection_time = 0
                self.last_move_time = 0
                self.frame_count = 0
                
                # 双缓冲机制
                self.current_frame = None
                self.frame_ready = False
                self.frame_lock = threading.Lock()
                
                # 瞄准参数
                self.aim_radius = int(config.get("aim_radius", "200"))
                self.current_target = None
                self.is_aiming = False
                self.last_target_time = 0
                self.target_timeout = 0.15

            def find_game_window(self):
                """查找游戏窗口"""
                if not self.exact_window_title:
                    print("❌ 窗口标题为空")
                    return False
                
                try:
                    print(f"🔍 正在查找窗口: '{self.exact_window_title}'")
                    
                    all_windows = gw.getAllWindows()
                    
                    # 精确匹配（排除自身）
                    windows = gw.getWindowsWithTitle(self.exact_window_title)
                    valid_windows = []
                    for window in windows:
                        if ("AI助手" not in window.title and 
                            "ONNX高性能版" not in window.title and
                            "ConfigEditor" not in window.title):
                            valid_windows.append(window)
                    
                    if valid_windows:
                        self.game_window = valid_windows[0]
                        print(f"✅ 精确找到游戏窗口: '{self.game_window.title}'")
                        return True
                    
                    # 模糊匹配
                    game_windows = []
                    for window in all_windows:
                        if (window.title and 
                            "FINALS" in window.title.upper() and 
                            "AI助手" not in window.title and 
                            "ONNX高性能版" not in window.title and
                            "ConfigEditor" not in window.title):
                            game_windows.append(window)
                    
                    if game_windows:
                        self.game_window = game_windows[0]
                        self.exact_window_title = self.game_window.title
                        print(f"✅ 模糊找到游戏窗口: '{self.game_window.title}'")
                        return True
                    
                    print("❌ 自动检测失败，请使用手动选择功能")
                    return False
                    
                except Exception as e:
                    print(f"❌ 查找窗口时出错: {e}")
                    return False

            def get_available_windows(self):
                """获取所有可用窗口（排除自身）"""
                try:
                    all_windows = gw.getAllWindows()
                    available_windows = []
                    
                    for window in all_windows:
                        if (window.title and 
                            len(window.title.strip()) > 0 and
                            "AI助手" not in window.title and 
                            "ONNX高性能版" not in window.title and
                            "ConfigEditor" not in window.title):
                            available_windows.append(window)
                    
                    return available_windows
                except Exception as e:
                    print(f"❌ 获取窗口列表时出错: {e}")
                    return []

            def manual_select_window_gui(self, parent_window):
                """GUI手动选择窗口"""
                try:
                    available_windows = self.get_available_windows()
                    if not available_windows:
                        messagebox.showerror("错误", "没有找到可用窗口")
                        return False
                    
                    select_window = tk.Toplevel(parent_window)
                    select_window.title("手动选择窗口")
                    select_window.geometry("600x400")
                    select_window.configure(bg='#1a1a1a')
                    select_window.transient(parent_window)
                    select_window.grab_set()
                    
                    title_label = tk.Label(select_window, text="请选择游戏窗口:", 
                                         font=("Arial", 14, "bold"), 
                                         bg='#1a1a1a', fg='white')
                    title_label.pack(pady=10)
                    
                    frame = tk.Frame(select_window, bg='#1a1a1a')
                    frame.pack(fill='both', expand=True, padx=20, pady=10)
                    
                    listbox = tk.Listbox(frame, bg='#2d2d2d', fg='white', 
                                       selectbackground='#3498db', font=("Arial", 10),
                                       height=15)
                    listbox.pack(fill='both', expand=True)
                    
                    scrollbar = tk.Scrollbar(listbox)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    listbox.config(yscrollcommand=scrollbar.set)
                    scrollbar.config(command=listbox.yview)
                    
                    for i, window in enumerate(available_windows):
                        listbox.insert(tk.END, f"{i+1}. {window.title}")
                    
                    selected_index = [None]
                    
                    def on_select():
                        selection = listbox.curselection()
                        if selection:
                            selected_index[0] = selection[0]
                    
                    def on_confirm():
                        if selected_index[0] is not None:
                            idx = selected_index[0]
                            if 0 <= idx < len(available_windows):
                                self.game_window = available_windows[idx]
                                self.exact_window_title = self.game_window.title
                                select_window.destroy()
                                messagebox.showinfo("成功", f"已选择窗口: {self.game_window.title}")
                                return True
                        messagebox.showerror("错误", "请先选择一个窗口")
                    
                    def on_cancel():
                        select_window.destroy()
                    
                    listbox.bind('<<ListboxSelect>>', lambda e: on_select())
                    
                    button_frame = tk.Frame(select_window, bg='#1a1a1a')
                    button_frame.pack(pady=10)
                    
                    tk.Button(button_frame, text="确认选择", command=on_confirm,
                             bg="#27ae60", fg="white", width=12).pack(side='left', padx=5)
                    tk.Button(button_frame, text="取消", command=on_cancel,
                             bg="#e74c3c", fg="white", width=12).pack(side='left', padx=5)
                    
                    select_window.wait_window()
                    
                    return selected_index[0] is not None
                    
                except Exception as e:
                    print(f"❌ 手动选择窗口时出错: {e}")
                    messagebox.showerror("错误", f"选择窗口时出错: {e}")
                    return False

            def capture_frame_optimized(self):
                """优化版截图"""
                if not self.game_window or not MSS_AVAILABLE:
                    return None
                
                try:
                    with mss.mss() as sct:
                        left = self.game_window.left
                        top = self.game_window.top
                        width = self.game_window.width
                        height = self.game_window.height
                        
                        center_x = width // 2
                        center_y = height // 2
                        
                        half_size = self.capture_size // 2
                        crop_left = left + max(0, center_x - half_size)
                        crop_top = top + max(0, center_y - half_size)
                        
                        monitor = {
                            "left": crop_left,
                            "top": crop_top, 
                            "width": self.capture_size,
                            "height": self.capture_size
                        }
                        
                        screenshot = sct.grab(monitor)
                        img = np.array(screenshot, dtype=np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        return img
                        
                except Exception as e:
                    return None

            def preprocess_image_fast(self, image):
                """快速预处理图像"""
                img = cv2.resize(image, (self.model_input_size, self.model_input_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)
                return img

            def detect_targets_fast(self, screenshot):
                """使用ONNX进行目标检测 - GPU优化"""
                if self.session is None:
                    return []
                
                try:
                    input_tensor = self.preprocess_image_fast(screenshot)
                    
                    start_time = time.perf_counter()
                    outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
                    inference_time = (time.perf_counter() - start_time) * 1000
                    
                    output = outputs[0]
                    targets = self.parse_yolov5_output(output)
                    
                    filtered_targets = []
                    for target in targets:
                        x_center, y_center = target['x'], target['y']
                        if self.is_target_in_aiming_zone(x_center, y_center):
                            distance = np.sqrt((x_center - self.center_x)**2 + (y_center - self.center_y)**2)
                            target['distance'] = distance
                            filtered_targets.append(target)
                    
                    return filtered_targets
                    
                except Exception as e:
                    return []

            def parse_yolov5_output(self, output):
                """解析YOLOv5输出"""
                targets = []
                conf_threshold = 0.35
                
                try:
                    if len(output.shape) == 3 and output.shape[0] == 1:
                        predictions = output[0]
                        
                        for pred in predictions:
                            if len(pred) >= 7:
                                x1, y1, x2, y2, conf, class_conf, class_id = pred[0:7]
                                
                                if conf > conf_threshold:
                                    x_center = int((x1 + x2) / 2)
                                    y_center = int((y1 + y2) / 2)
                                    targets.append({
                                        'x': x_center, 
                                        'y': y_center,
                                        'conf': conf
                                    })
                    
                    return targets
                    
                except Exception as e:
                    return []

            def is_target_in_aiming_zone(self, target_x, target_y):
                """检查目标是否在瞄准区域内"""
                distance = np.sqrt((target_x - self.center_x)**2 + (target_y - self.center_y)**2)
                return distance <= self.aim_radius

            def smooth_movement(self, new_dx, new_dy):
                """平滑移动"""
                smoothed_dx = self.last_dx * (1 - self.smoothing_factor) + new_dx * self.smoothing_factor
                smoothed_dy = self.last_dy * (1 - self.smoothing_factor) + new_dy * self.smoothing_factor
                
                self.last_dx = smoothed_dx
                self.last_dy = smoothed_dy
                
                return int(smoothed_dx), int(smoothed_dy)

            def move_mouse_optimized(self, target):
                """优化版鼠标移动"""
                try:
                    current_time = time.perf_counter()
                    if current_time - self.last_move_time < self.move_interval:
                        return True
                    
                    self.last_move_time = current_time
                    
                    screen_center_x = 960
                    screen_center_y = 540
                    
                    scale_x = 1920 / self.capture_size
                    scale_y = 1080 / self.capture_size
                    
                    target_x_absolute = target['x'] * scale_x
                    target_y_absolute = target['y'] * scale_y
                    
                    y_offset = int(self.config.get("y_offset", "0"))
                    target_y_absolute += y_offset
                    
                    dx = target_x_absolute - screen_center_x
                    dy = target_y_absolute - screen_center_y
                    
                    x_speed = int(self.config.get("x_speed", "20"))
                    y_speed = int(self.config.get("y_speed", "20"))
                    
                    dx = dx * (x_speed / 50.0)
                    dy = dy * (y_speed / 50.0)
                    
                    dx, dy = self.smooth_movement(dx, dy)
                    
                    dx = max(-20, min(20, dx))
                    dy = max(-20, min(20, dy))
                    
                    deadzone = int(self.config.get("deadzone", "2"))
                    if abs(dx) < deadzone: dx = 0
                    if abs(dy) < deadzone: dy = 0
                    
                    if dx != 0 or dy != 0:
                        ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)
                        return True
                    
                    return False
                    
                except Exception as e:
                    return False

            def start_ai(self):
                """启动AI"""
                if not self.is_running:
                    self.exact_window_title = self.config.get("exact_window_title", "")
                    print(f"🔍 窗口标题: '{self.exact_window_title}'")
                    
                    if not self.exact_window_title:
                        print("❌ 错误: 未设置游戏窗口标题")
                        messagebox.showerror("错误", "请先设置游戏窗口标题")
                        return False
                        
                    if not self.find_game_window():
                        print("❌ 错误: 找不到游戏窗口")
                        messagebox.showerror("错误", "找不到游戏窗口，请尝试手动选择")
                        return False
                    
                    if self.session is None:
                        print("❌ 错误: ONNX模型未加载成功")
                        messagebox.showerror("错误", "模型加载失败，请检查控制台输出")
                        return False
                    
                    self.emergency_stop = False
                    self.is_running = True
                    
                    self.last_dx = 0
                    self.last_dy = 0
                    self.current_target = None
                    self.is_aiming = False
                    
                    self.capture_thread = threading.Thread(target=self._capture_loop)
                    self.ai_thread = threading.Thread(target=self._ai_loop)
                    
                    self.capture_thread.daemon = True
                    self.ai_thread.daemon = True
                    
                    self.capture_thread.start()
                    self.ai_thread.start()
                    
                    print(f"🚀 高性能AI已启动 - GPU加速模式")
                    return True
                return False

            def _capture_loop(self):
                """截图循环"""
                print("📸 截图线程启动")
                
                while self.is_running and not self.emergency_stop:
                    frame_start = time.perf_counter()
                    
                    frame = self.capture_frame_optimized()
                    
                    if frame is not None:
                        with self.frame_lock:
                            self.current_frame = frame
                            self.frame_ready = True
                    
                    frame_time = (time.perf_counter() - frame_start) * 1000
                    target_frame_time = 1000.0 / 60
                    if frame_time < target_frame_time:
                        time.sleep((target_frame_time - frame_time) / 1000)
                    
                print("📸 截图线程停止")

            def _ai_loop(self):
                """AI循环 - GPU优化版本"""
                print("🤖 AI线程启动")
                last_fps_time = time.time()
                fps_counter = 0
                total_inference_time = 0
                detection_count = 0
                
                while self.is_running and not self.emergency_stop:
                    loop_start = time.perf_counter()
                    self.frame_count += 1
                    fps_counter += 1
                    
                    right_click_pressed = self.is_right_mouse_pressed()
                    if not right_click_pressed:
                        self.is_aiming = False
                        self.current_target = None
                        time.sleep(0.001)
                        continue
                    
                    current_frame = None
                    with self.frame_lock:
                        if self.frame_ready and self.current_frame is not None:
                            current_frame = self.current_frame.copy()
                            self.frame_ready = False
                    
                    if current_frame is None:
                        time.sleep(0.001)
                        continue
                    
                    current_time = time.perf_counter()
                    
                    if current_time - self.last_detection_time >= self.detection_interval:
                        self.last_detection_time = current_time
                        detection_count += 1
                        
                        inference_start = time.perf_counter()
                        targets = self.detect_targets_fast(current_frame)
                        inference_time = (time.perf_counter() - inference_start) * 1000
                        total_inference_time += inference_time
                        
                        if targets:
                            best_target = min(targets, key=lambda x: x['distance'])
                            self.current_target = best_target
                            self.is_aiming = True
                            self.last_target_time = current_time
                        else:
                            if current_time - self.last_target_time > self.target_timeout:
                                self.is_aiming = False
                                self.current_target = None
                    
                    if self.is_aiming and self.current_target:
                        self.move_mouse_optimized(self.current_target)
                    
                    current_time_display = time.time()
                    if current_time_display - last_fps_time >= 2.0:
                        fps = fps_counter / 2
                        avg_inference_time = total_inference_time / detection_count if detection_count > 0 else 0
                        fps_counter = 0
                        detection_count = 0
                        total_inference_time = 0
                        last_fps_time = current_time_display
                        
                        provider = "GPU" if 'CUDA' in str(self.session.get_providers()[0]) else "CPU"
                        if fps >= 80:
                            color = "✅"
                        elif fps >= 50:
                            color = "⚠️"
                        else:
                            color = "❌"
                            
                        print(f"{color} {provider}性能: {fps:.1f}FPS | 平均推理: {avg_inference_time:.1f}ms")
                    
                    loop_time = (time.perf_counter() - loop_start) * 1000
                    target_loop_time = 1000.0 / self.target_fps
                    if loop_time < target_loop_time:
                        time.sleep((target_loop_time - loop_time) / 1000)
                    
                print("🤖 AI线程停止")

            def is_right_mouse_pressed(self):
                """检查右键是否被按下"""
                try:
                    return ctypes.windll.user32.GetAsyncKeyState(0x02) != 0
                except Exception as e:
                    return False

            def stop_ai(self):
                """停止AI"""
                self.is_running = False
                self.emergency_stop = True
                if hasattr(self, 'capture_thread'):
                    self.capture_thread.join(timeout=1.0)
                if hasattr(self, 'ai_thread'):
                    self.ai_thread.join(timeout=1.0)

        # ConfigEditor类保持不变...

        class ConfigEditor:
            def __init__(self):
                self.window = tk.Tk()
                self.window.title("THE FINALS AI助手 - GPU加速版")
                self.window.geometry("700x800")
                self.window.configure(bg='#1a1a1a')
                self.config = self.load_config()
                self.game_ai = HighPerformanceAI(self.config)
                self.ai_running = False
                self.create_widgets()
                print("🎮 THE FINALS AI助手已启动! (GPU加速版)")

            def load_config(self):
                try:
                    with open("finals_config.json", "r", encoding="utf-8") as f:
                        return json.load(f)
                except FileNotFoundError:
                    return {
                        "x_speed": "20",
                        "y_speed": "20", 
                        "aim_radius": "200",
                        "y_offset": "0",
                        "deadzone": "2",
                        "exact_window_title": "THE FINALS"
                    }

            def create_widgets(self):
                main_frame = tk.Frame(self.window, bg='#1a1a1a')
                main_frame.pack(fill='both', expand=True, padx=20, pady=20)
                
                title_label = tk.Label(main_frame, text="🎯 THE FINALS AI助手 - GPU加速版",
                                      font=("Arial", 16, "bold"), bg='#1a1a1a', fg='white')
                title_label.pack(pady=(0, 15))
                
                # 模型状态
                model_status = "✅ 模型已加载" if self.game_ai.session is not None else "❌ 模型未加载"
                model_size = f"{self.game_ai.model_input_size}x{self.game_ai.model_input_size}" if self.game_ai.session else "未知"
                provider = "GPU" if self.game_ai.session and 'CUDA' in str(self.game_ai.session.get_providers()[0]) else "CPU"
                
                model_frame = tk.Frame(main_frame, bg='#1a1a1a')
                model_frame.pack(fill='x', pady=(0, 10))
                model_label = tk.Label(model_frame, text=f"模型状态: {model_status} | 尺寸: {model_size} | 加速: {provider}",
                                     font=("Arial", 10, "bold"), bg='#1a1a1a', 
                                     fg='#00ff00' if self.game_ai.session else '#ff0000')
                model_label.pack()
                
                # 创建选项卡
                notebook = ttk.Notebook(main_frame)
                notebook.pack(fill='both', expand=True)
                
                # 基本设置
                basic_frame = tk.Frame(notebook, bg='#1a1a1a')
                notebook.add(basic_frame, text="🎯 基本设置")
                
                sens_frame = tk.LabelFrame(basic_frame, text="🎮 瞄准灵敏度",
                                         font=("Arial", 12, "bold"),
                                         bg='#2d2d2d', fg='white', padx=15, pady=15)
                sens_frame.pack(fill='x', pady=10, padx=10)
                
                # X轴速度
                x_frame = tk.Frame(sens_frame, bg='#2d2d2d')
                x_frame.pack(fill='x', pady=8)
                tk.Label(x_frame, text="X轴速度:", bg='#2d2d2d', fg='white', width=10).pack(side='left')
                self.x_speed_var = tk.IntVar(value=int(self.config.get("x_speed", "20")))
                x_scale = tk.Scale(x_frame, from_=1, to=100, orient='horizontal', 
                                 variable=self.x_speed_var, bg='#2d2d2d', fg='white',
                                 highlightbackground='#2d2d2d', length=300)
                x_scale.pack(side='left', fill='x', expand=True)
                self.x_value_label = tk.Label(x_frame, text="20", bg='#2d2d2d', fg='#3498db', width=4)
                self.x_value_label.pack(side='right')
                x_scale.configure(command=lambda v: self.x_value_label.config(text=v))
                
                # Y轴速度
                y_frame = tk.Frame(sens_frame, bg='#2d2d2d')
                y_frame.pack(fill='x', pady=8)
                tk.Label(y_frame, text="Y轴速度:", bg='#2d2d2d', fg='white', width=10).pack(side='left')
                self.y_speed_var = tk.IntVar(value=int(self.config.get("y_speed", "20")))
                y_scale = tk.Scale(y_frame, from_=1, to=100, orient='horizontal',
                                 variable=self.y_speed_var, bg='#2d2d2d', fg='white',
                                 highlightbackground='#2d2d2d', length=300)
                y_scale.pack(side='left', fill='x', expand=True)
                self.y_value_label = tk.Label(y_frame, text="20", bg='#2d2d2d', fg='#3498db', width=4)
                self.y_value_label.pack(side='right')
                y_scale.configure(command=lambda v: self.y_value_label.config(text=v))
                
                # 高级设置
                advanced_frame = tk.Frame(notebook, bg='#1a1a1a')
                notebook.add(advanced_frame, text="⚙️ 高级设置")
                
                aim_frame = tk.LabelFrame(advanced_frame, text="🎯 瞄准设置",
                                        font=("Arial", 12, "bold"),
                                        bg='#2d2d2d', fg='white', padx=15, pady=15)
                aim_frame.pack(fill='x', pady=10, padx=10)
                
                # 瞄准半径
                radius_frame = tk.Frame(aim_frame, bg='#2d2d2d')
                radius_frame.pack(fill='x', pady=8)
                tk.Label(radius_frame, text="瞄准半径:", bg='#2d2d2d', fg='white', width=12).pack(side='left')
                self.radius_var = tk.IntVar(value=int(self.config.get("aim_radius", "200")))
                radius_scale = tk.Scale(radius_frame, from_=50, to=300, orient='horizontal',
                                      variable=self.radius_var, bg='#2d2d2d', fg='white',
                                      highlightbackground='#2d2d2d', length=300)
                radius_scale.pack(side='left', fill='x', expand=True)
                self.radius_value_label = tk.Label(radius_frame, text="200", bg='#2d2d2d', fg='#e67e22', width=4)
                self.radius_value_label.pack(side='right')
                radius_scale.configure(command=lambda v: self.radius_value_label.config(text=v))
                
                # Y轴偏移
                offset_frame = tk.Frame(aim_frame, bg='#2d2d2d')
                offset_frame.pack(fill='x', pady=8)
                tk.Label(offset_frame, text="Y轴偏移:", bg='#2d2d2d', fg='white', width=12).pack(side='left')
                self.offset_var = tk.IntVar(value=int(self.config.get("y_offset", "0")))
                offset_scale = tk.Scale(offset_frame, from_=-100, to=100, orient='horizontal',
                                      variable=self.offset_var, bg='#2d2d2d', fg='white',
                                      highlightbackground='#2d2d2d', length=300)
                offset_scale.pack(side='left', fill='x', expand=True)
                self.offset_value_label = tk.Label(offset_frame, text="0", bg='#2d2d2d', fg='#e67e22', width=4)
                self.offset_value_label.pack(side='right')
                offset_scale.configure(command=lambda v: self.offset_value_label.config(text=v))
                
                # 死区设置
                deadzone_frame = tk.Frame(aim_frame, bg='#2d2d2d')
                deadzone_frame.pack(fill='x', pady=8)
                tk.Label(deadzone_frame, text="瞄准死区:", bg='#2d2d2d', fg='white', width=12).pack(side='left')
                self.deadzone_var = tk.IntVar(value=int(self.config.get("deadzone", "2")))
                deadzone_scale = tk.Scale(deadzone_frame, from_=0, to=20, orient='horizontal',
                                        variable=self.deadzone_var, bg='#2d2d2d', fg='white',
                                        highlightbackground='#2d2d2d', length=300)
                deadzone_scale.pack(side='left', fill='x', expand=True)
                self.deadzone_value_label = tk.Label(deadzone_frame, text="2", bg='#2d2d2d', fg='#e67e22', width=4)
                self.deadzone_value_label.pack(side='right')
                deadzone_scale.configure(command=lambda v: self.deadzone_value_label.config(text=v))
                
                # 窗口设置
                window_frame = tk.LabelFrame(basic_frame, text="🖥️ 窗口设置",
                                           font=("Arial", 12, "bold"),
                                           bg='#2d2d2d', fg='white', padx=15, pady=15)
                window_frame.pack(fill='x', pady=10, padx=10)
                
                tk.Label(window_frame, text="游戏窗口标题:", bg='#2d2d2d', fg='white').pack(anchor='w')
                window_input_frame = tk.Frame(window_frame, bg='#2d2d2d')
                window_input_frame.pack(fill='x', pady=5)
                self.window_entry = tk.Entry(window_input_frame, bg='#404040', fg='white',
                                           insertbackground='white', width=40)
                self.window_entry.insert(0, self.config.get("exact_window_title", "THE FINALS"))
                self.window_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
                
                # 窗口操作按钮框架
                window_buttons_frame = tk.Frame(window_frame, bg='#2d2d2d')
                window_buttons_frame.pack(fill='x', pady=5)
                
                tk.Button(window_buttons_frame, text="自动检测", command=self.auto_detect_window,
                         bg='#e67e22', fg='white', width=12).pack(side='left', padx=2)
                tk.Button(window_buttons_frame, text="手动选择", command=self.manual_select_window,
                         bg='#3498db', fg='white', width=12).pack(side='left', padx=2)
                tk.Button(window_buttons_frame, text="测试截图", command=self.test_capture,
                         bg='#2ecc71', fg='white', width=12).pack(side='left', padx=2)
                
                # 状态和按钮
                self.status_label = tk.Label(main_frame, text="🔴 就绪 - 请先设置参数",
                                           font=("Arial", 11), bg='#1a1a1a', fg="#00ff00")
                self.status_label.pack(pady=10)
                
                button_frame = tk.Frame(main_frame, bg='#1a1a1a')
                button_frame.pack(pady=20)
                
                self.ai_btn = tk.Button(button_frame, text="🚀 启动AI", command=self.toggle_ai,
                                      bg="#27ae60", fg="white", width=12, height=2,
                                      font=("Arial", 11, "bold"))
                self.ai_btn.pack(side='left', padx=8)
                tk.Button(button_frame, text="💾 保存配置", command=self.save_config,
                         bg="#2980b9", fg="white", width=12, height=2,
                         font=("Arial", 11)).pack(side='left', padx=8)
                tk.Button(button_frame, text="🔄 重置默认", command=self.reset_defaults,
                         bg="#f39c12", fg="white", width=12, height=2,
                         font=("Arial", 11)).pack(side='left', padx=8)
                
                # 说明文本
                info_text = f"""🚀 GPU加速方案:

⚡ 硬件信息:
• GPU: NVIDIA GeForce RTX 4060 Ti
• 显存: 8GB
• CUDA: 13.0
• 推理速度: <5ms

🎯 性能目标:
• 检测频率: 60FPS
• 移动频率: 120FPS  
• 平滑瞄准: 启用

💡 使用说明:
1. 点击"自动检测"或"手动选择"设置游戏窗口
2. 点击"测试截图"验证窗口捕获
3. 调整灵敏度参数
4. 右键瞄准触发AI"""
                
                info_label = tk.Label(main_frame, text=info_text,
                                     justify="left", fg="#cccccc", font=("Arial", 9),
                                     bg='#1a1a1a')
                info_label.pack(pady=10)

            def auto_detect_window(self):
                """自动检测游戏窗口"""
                all_windows = gw.getAllWindows()
                for window in all_windows:
                    if (window.title and 
                        "FINALS" in window.title.upper() and 
                        "AI助手" not in window.title and 
                        "ONNX高性能版" not in window.title and
                        "ConfigEditor" not in window.title):
                        self.window_entry.delete(0, tk.END)
                        self.window_entry.insert(0, window.title)
                        self.status_label.config(text=f"✅ 已检测到游戏窗口: {window.title}", fg="#00ff00")
                        return
                self.status_label.config(text="❌ 未找到THE FINALS游戏窗口", fg="red")

            def manual_select_window(self):
                """手动选择窗口"""
                if self.game_ai.manual_select_window_gui(self.window):
                    self.window_entry.delete(0, tk.END)
                    self.window_entry.insert(0, self.game_ai.exact_window_title)
                    self.status_label.config(text=f"✅ 手动选择窗口: {self.game_ai.exact_window_title}", fg="#00ff00")
                else:
                    self.status_label.config(text="❌ 手动选择窗口失败或已取消", fg="red")

            def test_capture(self):
                """测试截图功能"""
                if not self.game_ai.exact_window_title:
                    self.status_label.config(text="❌ 请先设置窗口标题", fg="red")
                    return
                
                if not self.game_ai.find_game_window():
                    self.status_label.config(text="❌ 找不到游戏窗口", fg="red")
                    return
                
                frame = self.game_ai.capture_frame_optimized()
                if frame is not None:
                    # 保存测试截图
                    test_dir = "test_captures"
                    if not os.path.exists(test_dir):
                        os.makedirs(test_dir)
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(test_dir, f"test_capture_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    
                    self.status_label.config(text=f"✅ 截图测试成功: {filename}", fg="#00ff00")
                    print(f"📸 测试截图已保存: {filename}")
                    
                    # 显示截图信息
                    h, w = frame.shape[:2]
                    print(f"📏 截图尺寸: {w}x{h}")
                else:
                    self.status_label.config(text="❌ 截图测试失败", fg="red")

            def save_config(self):
                try:
                    config = {
                        "x_speed": str(self.x_speed_var.get()),
                        "y_speed": str(self.y_speed_var.get()),
                        "aim_radius": str(self.radius_var.get()),
                        "y_offset": str(self.offset_var.get()),
                        "deadzone": str(self.deadzone_var.get()),
                        "exact_window_title": self.window_entry.get()
                    }
                    with open("finals_config.json", "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                    self.config = config
                    self.game_ai.config = config
                    self.game_ai.aim_radius = int(config.get("aim_radius", "200"))
                    current_time = time.strftime("%H:%M:%S")
                    self.status_label.config(text=f"✅ 保存成功：{current_time}", fg="#00ff00")
                except Exception as e:
                    self.status_label.config(text=f"❌ 保存失败：{str(e)}", fg="red")

            def reset_defaults(self):
                self.x_speed_var.set(20)
                self.y_speed_var.set(20)
                self.radius_var.set(200)
                self.offset_var.set(0)
                self.deadzone_var.set(2)
                self.x_value_label.config(text="20")
                self.y_value_label.config(text="20")
                self.radius_value_label.config(text="200")
                self.offset_value_label.config(text="0")
                self.deadzone_value_label.config(text="2")
                self.status_label.config(text="✅ 已重置为默认设置", fg="#00ff00")

            def toggle_ai(self):
                if not self.ai_running:
                    self.save_config()
                    if self.game_ai.start_ai():
                        self.ai_running = True
                        self.ai_btn.config(text="🛑 停止AI", bg="#c0392b")
                        self.status_label.config(text="🎯 AI运行中 - 监控控制台性能", fg="#3498db")
                    else:
                        self.status_label.config(text="❌ AI启动失败", fg="red")
                else:
                    self.game_ai.stop_ai()
                    self.ai_running = False
                    self.ai_btn.config(text="🚀 启动AI", bg="#27ae60")
                    self.status_label.config(text="🔴 AI已停止", fg="#00ff00")

            def run(self):
                self.window.mainloop()

        try:
            app = ConfigEditor()
            app.run()
            print("程序正常退出")
        except Exception as e:
            print(f"程序启动失败: {e}")
            traceback.print_exc()
            input("按回车键退出...")