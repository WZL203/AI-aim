import sys
import ctypes
import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
import threading
import cv2
import numpy as np
import pygetwindow as gw
import traceback
from ultralytics import YOLO
import torch

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

        class GameInputController:
            """游戏输入控制器 - 可调节版本"""
            def __init__(self):
                try:
                    self.last_move_time = 0
                    self.move_interval = 0.01  # 100FPS移动间隔
                    print("✓ 游戏输入控制器初始化成功")
                except Exception as e:
                    print(f"❌ 输入控制器初始化失败: {e}")

            def move_mouse_relative(self, dx, dy):
                """相对移动鼠标"""
                try:
                    current_time = time.perf_counter()
                    if current_time - self.last_move_time < self.move_interval:
                        return True
                    
                    self.last_move_time = current_time
                    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
                    return True
                except Exception as e:
                    return False

            def send_mouse_input(self, x, y, game_window, config):
                """发送鼠标输入到游戏 - 可调节版本"""
                try:
                    # 1920x1080分辨率下的中心点
                    screen_center_x = 960
                    screen_center_y = 540
                    
                    # 计算在1920x1080屏幕上的绝对坐标
                    scale_x = 1920 / 640
                    scale_y = 1080 / 640
                    
                    target_x_absolute = x * scale_x
                    target_y_absolute = y * scale_y
                    
                    # 应用Y轴偏移
                    y_offset = int(config.get("y_offset", "0"))
                    target_y_absolute += y_offset
                    
                    # 计算偏移量
                    dx = target_x_absolute - screen_center_x
                    dy = target_y_absolute - screen_center_y
                    
                    # 获取灵敏度设置
                    x_speed = int(config.get("x_speed", "50"))
                    y_speed = int(config.get("y_speed", "50"))
                    
                    # 应用独立的X/Y灵敏度
                    dx = int(dx * (x_speed / 50.0))
                    dy = int(dy * (y_speed / 50.0))
                    
                    # 移动限制
                    dx = max(-100, min(100, dx))
                    dy = max(-100, min(100, dy))
                    
                    # 死区控制
                    deadzone = int(config.get("deadzone", "5"))
                    if abs(dx) < deadzone: dx = 0
                    if abs(dy) < deadzone: dy = 0
                    
                    if dx != 0 or dy != 0:
                        success = self.move_mouse_relative(dx, dy)
                        return success
                    return True
                except Exception as e:
                    return False

        class GameAI:
            def __init__(self, config):
                self.config = config
                self.is_running = False
                self.thread = None
                self.game_window = None
                self.exact_window_title = config.get("exact_window_title", "")
                self.emergency_stop = False
                self.last_right_click_state = False
                self.model = YOLO("C:/Users/Administrator/yolov12/runs/detect/train5/weights/best.pt")
                
                # 可调节参数
                self.aim_radius = int(config.get("aim_radius", "150"))
                self.center_x = 320
                self.center_y = 320
                self.frame_count = 0
                self.current_target = None
                self.is_aiming = False
                
                # 性能参数
                self.last_detection_time = 0
                self.detection_interval = 1/60  # 60FPS
                self.frame_times = []

                # 初始化输入控制器
                self.input_controller = GameInputController()

            def find_game_window(self):
                """查找游戏窗口"""
                if not self.exact_window_title:
                    return False
                try:
                    windows = gw.getWindowsWithTitle(self.exact_window_title)
                    if windows:
                        self.game_window = windows[0]
                        return True
                    return False
                except Exception as e:
                    return False

            def capture_game_window_mss(self):
                """使用MSS截取游戏窗口"""
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
                        
                        crop_left = left + max(0, center_x - 320)
                        crop_top = top + max(0, center_y - 320)
                        
                        monitor = {
                            "left": crop_left,
                            "top": crop_top, 
                            "width": 640,
                            "height": 640
                        }
                        
                        screenshot = sct.grab(monitor)
                        img = np.array(screenshot)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        return img
                        
                except Exception as e:
                    return None

            def capture_game_window(self):
                """主截图方法"""
                if not self.game_window:
                    if not self.find_game_window():
                        return None
                return self.capture_game_window_mss()

            def start_ai(self):
                """启动AI"""
                if not self.is_running:
                    self.exact_window_title = self.config.get("exact_window_title", "")
                    if not self.exact_window_title:
                        return False
                    if not self.find_game_window():
                        return False
                    self.emergency_stop = False
                    self.is_running = True
                    self.thread = threading.Thread(target=self._ai_loop)
                    self.thread.daemon = True
                    self.thread.start()
                    print("AI线程已启动 - 可调节版本")
                    return True
                return False

            def _ai_loop(self):
                """AI主循环"""
                try:
                    print("AI开始运行...")
                    
                    while self.is_running and not self.emergency_stop:
                        frame_start = time.perf_counter()
                        self.frame_count += 1
                        
                        # 检查右键状态
                        right_click_pressed = self.is_right_mouse_pressed()
                        if not right_click_pressed:
                            self.is_aiming = False
                            self.current_target = None
                            time.sleep(0.01)
                            continue
                        
                        # 控制检测频率
                        current_time = time.perf_counter()
                        if current_time - self.last_detection_time < self.detection_interval:
                            time.sleep(0.001)
                            continue
                        
                        self.last_detection_time = current_time
                        
                        # 截图和检测
                        game_screenshot = self.capture_game_window()
                        if game_screenshot is not None:
                            targets_in_zone = self.detect_targets_in_aiming_zone(game_screenshot)
                            
                            if targets_in_zone:
                                best_target = self.select_best_target_in_zone(targets_in_zone)
                                
                                if not self.is_aiming:
                                    print(f"🎯 发现目标! 距离: {best_target['distance']:.1f}px")
                                    self.is_aiming = True
                                
                                self.aim_at_target(best_target)
                            
                            else:
                                if self.is_aiming:
                                    self.is_aiming = False
                        
                        # 性能监控
                        frame_end = time.perf_counter()
                        frame_time = (frame_end - frame_start) * 1000
                        self.frame_times.append(frame_time)
                        
                        if self.frame_count % 120 == 0:
                            avg_frame_time = np.mean(self.frame_times)
                            fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
                            if fps < 50:
                                print(f"⚠️ 性能: {fps:.1f}FPS - 考虑降低设置")
                            self.frame_times = []
                        
                except Exception as e:
                    print(f"AI循环错误: {e}")

            def is_target_in_aiming_zone(self, target_x, target_y):
                """检查目标是否在瞄准区域内"""
                distance = np.sqrt((target_x - self.center_x)**2 + (target_y - self.center_y)**2)
                return distance <= self.aim_radius

            def detect_targets_in_aiming_zone(self, screenshot):
                """检测在瞄准区域内的目标"""
                targets_in_zone = []
                
                try:
                    img_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                    
                    results = self.model.predict(
                        source=img_rgb, 
                        conf=0.3,
                        imgsz=640,
                        verbose=False
                    )[0]
                    
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # 计算目标中心点
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        
                        # 检查是否在瞄准区域内
                        if self.is_target_in_aiming_zone(x_center, y_center):
                            distance = np.sqrt((x_center - self.center_x)**2 + (y_center - self.center_y)**2)
                            
                            target = {
                                'x': x_center, 
                                'y': y_center,
                                'conf': conf,
                                'distance': distance
                            }
                            targets_in_zone.append(target)

                except Exception as e:
                    pass
                
                return targets_in_zone

            def select_best_target_in_zone(self, targets):
                """选择瞄准区域内最佳目标"""
                if not targets:
                    return None
                # 选择距离最近的目标
                return min(targets, key=lambda x: x['distance'])

            def aim_at_target(self, target):
                """瞄准目标"""
                try:
                    if not self.is_right_mouse_pressed():
                        return
                    
                    success = self.input_controller.send_mouse_input(
                        target['x'], target['y'], self.game_window, self.config
                    )
                    
                except Exception as e:
                    pass

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
                if self.thread:
                    self.thread.join(timeout=1.0)

        class ConfigEditor:
            def __init__(self):
                self.window = tk.Tk()
                self.window.title("THE FINALS AI助手 - 完整调节版")
                self.window.geometry("700x800")
                self.window.configure(bg='#1a1a1a')
                self.config = self.load_config()
                self.game_ai = GameAI(self.config)
                self.ai_running = False
                self.create_widgets()
                print("🎮 THE FINALS AI助手已启动! (完整调节版)")

            def load_config(self):
                try:
                    with open("finals_config.json", "r", encoding="utf-8") as f:
                        return json.load(f)
                except FileNotFoundError:
                    return {
                        "x_speed": "50",
                        "y_speed": "50", 
                        "aim_radius": "150",
                        "y_offset": "0",
                        "deadzone": "5",
                        "exact_window_title": ""
                    }

            def create_widgets(self):
                main_frame = tk.Frame(self.window, bg='#1a1a1a')
                main_frame.pack(fill='both', expand=True, padx=20, pady=20)
                
                # 标题
                title_label = tk.Label(main_frame, text="🎯 THE FINALS AI助手 - 完整调节版",
                                      font=("Arial", 18, "bold"), bg='#1a1a1a', fg='white')
                title_label.pack(pady=(0, 20))
                
                # 创建选项卡
                notebook = ttk.Notebook(main_frame)
                notebook.pack(fill='both', expand=True)
                
                # 基本设置选项卡
                basic_frame = tk.Frame(notebook, bg='#1a1a1a')
                notebook.add(basic_frame, text="🎯 基本设置")
                
                # 灵敏度设置
                sens_frame = tk.LabelFrame(basic_frame, text="🎮 瞄准灵敏度",
                                         font=("Arial", 12, "bold"),
                                         bg='#2d2d2d', fg='white', padx=15, pady=15)
                sens_frame.pack(fill='x', pady=10, padx=10)
                
                # X轴速度滑块
                x_frame = tk.Frame(sens_frame, bg='#2d2d2d')
                x_frame.pack(fill='x', pady=8)
                tk.Label(x_frame, text="X轴速度:", bg='#2d2d2d', fg='white', width=10).pack(side='left')
                self.x_speed_var = tk.IntVar(value=int(self.config.get("x_speed", "50")))
                x_scale = tk.Scale(x_frame, from_=1, to=100, orient='horizontal', 
                                 variable=self.x_speed_var, bg='#2d2d2d', fg='white',
                                 highlightbackground='#2d2d2d', length=300)
                x_scale.pack(side='left', fill='x', expand=True)
                self.x_value_label = tk.Label(x_frame, text="50", bg='#2d2d2d', fg='#3498db', width=4)
                self.x_value_label.pack(side='right')
                
                # Y轴速度滑块
                y_frame = tk.Frame(sens_frame, bg='#2d2d2d')
                y_frame.pack(fill='x', pady=8)
                tk.Label(y_frame, text="Y轴速度:", bg='#2d2d2d', fg='white', width=10).pack(side='left')
                self.y_speed_var = tk.IntVar(value=int(self.config.get("y_speed", "50")))
                y_scale = tk.Scale(y_frame, from_=1, to=100, orient='horizontal',
                                 variable=self.y_speed_var, bg='#2d2d2d', fg='white',
                                 highlightbackground='#2d2d2d', length=300)
                y_scale.pack(side='left', fill='x', expand=True)
                self.y_value_label = tk.Label(y_frame, text="50", bg='#2d2d2d', fg='#3498db', width=4)
                self.y_value_label.pack(side='right')
                
                # 绑定滑块事件
                x_scale.configure(command=self.on_x_speed_change)
                y_scale.configure(command=self.on_y_speed_change)
                
                # 高级设置选项卡
                advanced_frame = tk.Frame(notebook, bg='#1a1a1a')
                notebook.add(advanced_frame, text="⚙️ 高级设置")
                
                # 瞄准设置
                aim_frame = tk.LabelFrame(advanced_frame, text="🎯 瞄准设置",
                                        font=("Arial", 12, "bold"),
                                        bg='#2d2d2d', fg='white', padx=15, pady=15)
                aim_frame.pack(fill='x', pady=10, padx=10)
                
                # 瞄准半径滑块
                radius_frame = tk.Frame(aim_frame, bg='#2d2d2d')
                radius_frame.pack(fill='x', pady=8)
                tk.Label(radius_frame, text="瞄准半径:", bg='#2d2d2d', fg='white', width=12).pack(side='left')
                self.radius_var = tk.IntVar(value=int(self.config.get("aim_radius", "150")))
                radius_scale = tk.Scale(radius_frame, from_=50, to=300, orient='horizontal',
                                      variable=self.radius_var, bg='#2d2d2d', fg='white',
                                      highlightbackground='#2d2d2d', length=300)
                radius_scale.pack(side='left', fill='x', expand=True)
                self.radius_value_label = tk.Label(radius_frame, text="150", bg='#2d2d2d', fg='#e67e22', width=4)
                self.radius_value_label.pack(side='right')
                radius_scale.configure(command=self.on_radius_change)
                
                # Y轴偏移滑块
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
                offset_scale.configure(command=self.on_offset_change)
                
                # 死区设置
                deadzone_frame = tk.Frame(aim_frame, bg='#2d2d2d')
                deadzone_frame.pack(fill='x', pady=8)
                tk.Label(deadzone_frame, text="瞄准死区:", bg='#2d2d2d', fg='white', width=12).pack(side='left')
                self.deadzone_var = tk.IntVar(value=int(self.config.get("deadzone", "5")))
                deadzone_scale = tk.Scale(deadzone_frame, from_=0, to=20, orient='horizontal',
                                        variable=self.deadzone_var, bg='#2d2d2d', fg='white',
                                        highlightbackground='#2d2d2d', length=300)
                deadzone_scale.pack(side='left', fill='x', expand=True)
                self.deadzone_value_label = tk.Label(deadzone_frame, text="5", bg='#2d2d2d', fg='#e67e22', width=4)
                self.deadzone_value_label.pack(side='right')
                deadzone_scale.configure(command=self.on_deadzone_change)
                
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
                self.window_entry.insert(0, self.config.get("exact_window_title", ""))
                self.window_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
                tk.Button(window_input_frame, text="自动检测", command=self.auto_detect_window,
                         bg='#e67e22', fg='white', width=10).pack(side='right')
                
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
                info_text = """🎮 完整调节指南:

🎯 基本设置:
• X轴速度: 水平移动速度 (推荐: 40-60)
• Y轴速度: 垂直移动速度 (推荐: 40-60)

⚙️ 高级设置:
• 瞄准半径: 检测范围像素 (推荐: 120-180)
• Y轴偏移: 瞄准点上下调整 (爆头: +20~+30)
• 瞄准死区: 防抖动范围 (推荐: 3-8)

🚀 调试步骤:
1. 设置X/Y速度为40
2. 瞄准半径设为150  
3. 启动AI测试基础功能
4. 逐步调整参数到最佳状态

💡 爆头技巧:
• 增加Y轴偏移+20~+30
• 适当降低Y轴速度
• 减少瞄准死区"""
                
                info_label = tk.Label(main_frame, text=info_text,
                                     justify="left", fg="#cccccc", font=("Arial", 9),
                                     bg='#1a1a1a')
                info_label.pack(pady=10)

            def on_x_speed_change(self, value):
                self.x_value_label.config(text=value)

            def on_y_speed_change(self, value):
                self.y_value_label.config(text=value)

            def on_radius_change(self, value):
                self.radius_value_label.config(text=value)

            def on_offset_change(self, value):
                self.offset_value_label.config(text=value)

            def on_deadzone_change(self, value):
                self.deadzone_value_label.config(text=value)

            def auto_detect_window(self):
                all_windows = gw.getAllWindows()
                for window in all_windows:
                    if window.title and "THE FINALS" in window.title:
                        self.window_entry.delete(0, tk.END)
                        self.window_entry.insert(0, window.title)
                        self.status_label.config(text=f"✅ 已检测到窗口", fg="#00ff00")
                        return
                self.status_label.config(text="❌ 未找到THE FINALS窗口", fg="red")

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
                    self.game_ai.aim_radius = int(config.get("aim_radius", "150"))
                    current_time = time.strftime("%H:%M:%S")
                    self.status_label.config(text=f"✅ 保存成功：{current_time}", fg="#00ff00")
                except Exception as e:
                    self.status_label.config(text=f"❌ 保存失败：{str(e)}", fg="red")

            def reset_defaults(self):
                self.x_speed_var.set(50)
                self.y_speed_var.set(50)
                self.radius_var.set(150)
                self.offset_var.set(0)
                self.deadzone_var.set(5)
                self.x_value_label.config(text="50")
                self.y_value_label.config(text="50")
                self.radius_value_label.config(text="150")
                self.offset_value_label.config(text="0")
                self.deadzone_value_label.config(text="5")
                self.status_label.config(text="✅ 已重置为默认设置", fg="#00ff00")

            def toggle_ai(self):
                if not self.ai_running:
                    self.save_config()
                    if self.game_ai.start_ai():
                        self.ai_running = True
                        self.ai_btn.config(text="🛑 停止AI", bg="#c0392b")
                        x_speed = self.x_speed_var.get()
                        y_speed = self.y_speed_var.get()
                        self.status_label.config(text=f"🎯 AI运行中 - X:{x_speed} Y:{y_speed}", fg="#3498db")
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