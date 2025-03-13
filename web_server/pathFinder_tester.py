import tkinter as tk
from tkinter import messagebox, ttk
import math
from web_server.utils.pathFinder import PathFinder
from web_server.utils.consts import Direction, WIDTH, HEIGHT

class PathfindingTestUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pathfinding Test Tool")
        
        # 地图参数
        self.grid_size = 22
        self.cell_size = 30
        self.robot_start = (1, 1)
        self.robot_dir = Direction.NORTH
        # 修改障碍物数据结构：键为坐标，值为字典，包含障碍物信息
        self.obstacles = {}  # {(x,y): {"x": x, "y": y, "d": direction, "id": unique_id}}
        self.obstacle_counter = 0
        
        # UI控件初始化
        self.create_widgets()
        self.draw_grid()
        
    def create_widgets(self):
        # 控制面板
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 机器人起始位置输入
        tk.Label(control_frame, text="Robot Start (x,y):").grid(row=0, column=0)
        self.entry_x = tk.Entry(control_frame, width=5)
        self.entry_y = tk.Entry(control_frame, width=5)
        self.entry_x.insert(0, "1")
        self.entry_y.insert(0, "1")
        self.entry_x.grid(row=0, column=1)
        self.entry_y.grid(row=0, column=2)
        
        # 机器人方向选择
        tk.Label(control_frame, text="Direction:").grid(row=1, column=0)
        self.dir_combo = ttk.Combobox(control_frame, values=[d.name for d in Direction])
        self.dir_combo.current(0)
        self.dir_combo.grid(row=1, column=1, columnspan=2)
        
        # 操作按钮
        tk.Button(control_frame, text="Set Start", command=self.set_start).grid(row=2, column=0, columnspan=3)
        tk.Button(control_frame, text="Calculate Path", command=self.calculate_path).grid(row=3, column=0, columnspan=3)
        tk.Button(control_frame, text="Clear All", command=self.clear_all).grid(row=4, column=0, columnspan=3)
        
        # 显示命令的文本框
        tk.Label(control_frame, text="Generated Commands:").grid(row=5, column=0, columnspan=3, pady=(10,0))
        self.commands_text = tk.Text(control_frame, width=30, height=30)
        self.commands_text.grid(row=6, column=0, columnspan=3)
        
        # 新增按钮：绘制实际行驶轨迹
        tk.Button(control_frame, text="Show Actual Route", command=self.show_actual_route).grid(row=7, column=0, columnspan=3, pady=(10,0))
        
        # 画布初始化
        self.canvas = tk.Canvas(self.master, 
                              width=self.grid_size*self.cell_size, 
                              height=self.grid_size*self.cell_size)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.add_obstacle)
    
    def draw_grid(self):
        self.canvas.delete("all")
        # 绘制网格线
        for i in range(self.grid_size+1):
            self.canvas.create_line(0, i*self.cell_size, 
                                  self.grid_size*self.cell_size, i*self.cell_size)
            self.canvas.create_line(i*self.cell_size, 0,
                                  i*self.cell_size, self.grid_size*self.cell_size)
        
        for i in range(self.grid_size):
            x_pos = i * self.cell_size + self.cell_size / 2
            self.canvas.create_text(x_pos, 10, text=str(i), font=("Arial", 10), fill="black")
        # Y轴：每个格子中心显示行号（注意转换坐标：上边缘对应行号最大）
        for j in range(self.grid_size):
            y_pos = (self.grid_size - j - 1) * self.cell_size + self.cell_size / 2
            self.canvas.create_text(10, y_pos, text=str(j), font=("Arial", 10), fill="black")

            
        # 绘制机器人起始位置
        x, y = self.robot_start
        self.draw_robot(x, y)
        
        # 绘制障碍物
        for obstacle in self.obstacles.values():
            self.draw_obstacle(obstacle["x"], obstacle["y"], obstacle["d"])
    def draw_robot(self, x, y):
        # 转换坐标系（左下角为原点）
        canvas_y = (self.grid_size - y - 1) * self.cell_size
        self.canvas.create_oval(
            x*self.cell_size + 5, canvas_y + 5,
            (x+1)*self.cell_size -5, canvas_y + self.cell_size -5,
            fill="blue", tags="robot"
        )
    def draw_obstacle(self, x, y, direction):
        # 转换坐标系
        canvas_y = (self.grid_size - y - 1) * self.cell_size
        # 绘制障碍物主体
        self.canvas.create_rectangle(
            x*self.cell_size, canvas_y,
            (x+1)*self.cell_size, canvas_y + self.cell_size,
            fill="red", tags="obstacle"
        )
        # 绘制方向指示
        dir_symbol = {
            Direction.NORTH: "↑",
            Direction.EAST: "→",
            Direction.SOUTH: "↓",
            Direction.WEST: "←"
        }.get(direction, "")
        self.canvas.create_text(
            (x+0.5)*self.cell_size, canvas_y + 0.5*self.cell_size,
            text=dir_symbol, font=("Arial", 12), fill="white"
        )
    def set_start(self):
        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.robot_start = (x, y)
                self.robot_dir = Direction[self.dir_combo.get()]
                self.draw_grid()
            else:
                messagebox.showerror("Error", "Invalid coordinates!")
        except Exception as e:
            messagebox.showerror("Error", "Invalid input!")
    def add_obstacle(self, event):
        # 转换坐标系
        x = event.x // self.cell_size
        y = self.grid_size - (event.y // self.cell_size) - 1
        
        # 弹出方向选择菜单
        menu = tk.Menu(self.master, tearoff=0)
        for d in [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]:
            menu.add_command(
                label=d.name,
                command=lambda d=d: self._save_obstacle(x, y, d)
            )
        menu.tk_popup(event.x_root, event.y_root)
    def _save_obstacle(self, x, y, direction):
        key = (x, y)
        if key in self.obstacles:
            # 删除障碍物
            del self.obstacles[key]
        else:
            # 添加新的障碍物，同时生成唯一id
            self.obstacles[key] = {"x": x, "y": y, "d": direction, "id": self.obstacle_counter}
            self.obstacle_counter += 1
        self.draw_grid()
    def clear_all(self):
        self.obstacles.clear()
        self.draw_grid()
        self.commands_text.delete("1.0", tk.END)
        self.canvas.delete("actual_route")
    
    def calculate_path(self):

        # Initialize path finder
        path_finder = PathFinder(
            size_x=self.grid_size,
            size_y=self.grid_size,
            robot_x=self.robot_start[0],
            robot_y=self.robot_start[1],
            robot_direction=self.robot_dir
        )
        
        # Add obstacles
        for obstacle in self.obstacles.values():
            path_finder.add_obstacle(obstacle["x"], obstacle["y"], obstacle["d"], obstacle_id=obstacle["id"])
        
        try:
            
            # Calculate path
            optimal_path, distance = path_finder.get_optimal_order_dp(retrying=False)
            self.draw_path(optimal_path)
            
            # Generate commands
            from web_server.utils.helper import commandGenerator
            obstacles_list = list(self.obstacles.values())
            self.commands = commandGenerator(optimal_path, obstacles_list)
        
            # Display commands
            self.commands_text.delete("1.0", tk.END)
            self.commands_text.insert(tk.END, "\n".join(self.commands))
            messagebox.showinfo("Result", f"Path found! Total distance: {distance}")
        except Exception as e:
            messagebox.showerror("Error", f"No valid path found!\n{str(e)}")
    
    def draw_path(self, path):
        self.canvas.delete("path")
        prev_point = None
        arrow_length = 10  # 箭头长度
        for state in path:
            # 转换坐标系
            x = state.x
            y = self.grid_size - state.y - 1
            cx = (x + 0.5) * self.cell_size
            cy = (y + 0.5) * self.cell_size
            
            # # 绘制路径点
            # self.canvas.create_oval(
            #     cx-3, cy-3, cx+3, cy+3,
            #     fill="green", tags="path"
            # )
            
            # 绘制连接线
            if prev_point:
                self.canvas.create_line(
                    prev_point[0], prev_point[1], cx, cy,
                    fill="green", width=1, tags="path"
                )
            prev_point = (cx, cy)
            
            # 绘制方向箭头（假设state中有direction属性）
            if hasattr(state, 'direction'):
                if state.direction == Direction.NORTH:
                    angle = math.radians(90)
                elif state.direction == Direction.EAST:
                    angle = math.radians(0)
                elif state.direction == Direction.SOUTH:
                    angle = math.radians(270)
                elif state.direction == Direction.WEST:
                    angle = math.radians(180)
                else:
                    angle = 0
                # 计算箭头终点（注意Tkinter的y轴方向）
                x2 = cx + arrow_length * math.cos(angle)
                y2 = cy - arrow_length * math.sin(angle)
                self.canvas.create_line(
                    cx, cy, x2, y2,
                    arrow=tk.LAST, fill="green", width=3, tags="path"
                )
    # 新增：根据生成的命令绘制小车的实际行驶轨迹（包括转弯圆弧和标出小车头部）
    def draw_actual_route(self, commands):
        self.canvas.delete("actual_route")
        # 设定初始状态，位置取网格中心
        x = self.robot_start[0] + 0.5
        y = self.robot_start[1] + 0.5
        # 将初始方向转换为弧度（约定0°为东，90°为北）
        if self.robot_dir == Direction.NORTH:
            theta = math.radians(90)
        elif self.robot_dir == Direction.EAST:
            theta = math.radians(0)
        elif self.robot_dir == Direction.SOUTH:
            theta = math.radians(270)
        elif self.robot_dir == Direction.WEST:
            theta = math.radians(180)
        else:
            theta = 0
        points = []  # 记录轨迹上的关键点（世界坐标系）
        points.append((x, y))
        for cmd in commands:
            if cmd.startswith("SF") or cmd.startswith("SB"):
                # 直行命令，提取距离（单位为和地图单位一致）
                dist = int(cmd[2:]) / 10
                # 如果是倒退则距离取负
                if cmd.startswith("SB"):
                    dist = -dist
                # 更新位置
                x_new = x + dist * math.cos(theta)
                y_new = y + dist * math.sin(theta)
                # 绘制直线段
                self.draw_line_world(x, y, x_new, y_new, tag="actual_route", color="purple")
                x, y = x_new, y_new
                points.append((x, y))
                self.draw_point_world(x, y, tag="actual_route", color="red", size=4)
            elif cmd[:2] in ["RF", "LF", "RB", "LB"]:
                # 固定转角90度，转弯半径为1
                turn_angle = math.radians(90)
                # 判断前进或倒车，并设置转向sign（前进时直接使用，倒车时反转sign）
                if cmd.startswith("RF"):
                    forward = True
                    sign = -1  # forward right: 顺时针转
                elif cmd.startswith("LF"):
                    forward = True
                    sign = 1   # forward left: 逆时针转
                elif cmd.startswith("RB"):
                    forward = False
                    sign = 1   # backward right: 这里与前进相反，倒车右转用正号
                elif cmd.startswith("LB"):
                    forward = False
                    sign = -1  # backward left: 这里用负号
                else:
                    continue

                # 根据是否前进确定实际运动方向
                motion_theta = theta if forward else (theta + math.pi)

                # 根据运动方向计算转弯圆心（右转：偏移向量为 (sin(motion_theta), -cos(motion_theta)），左转取反）
                if sign == -1:
                    offset_x = math.sin(motion_theta)
                    offset_y = -math.cos(motion_theta)
                else:
                    offset_x = -math.sin(motion_theta)
                    offset_y = math.cos(motion_theta)
                center_x = x + offset_x
                center_y = y + offset_y

                # 绘制圆弧
                radius = 1
                c1x, c1y = self.world_to_canvas(center_x - radius, center_y - radius)
                c2x, c2y = self.world_to_canvas(center_x + radius, center_y + radius)
                # 计算起始角度（以东为0°，逆时针增加）
                start_angle = math.degrees(math.atan2(y - center_y, x - center_x))
                # Tkinter的create_arc要求，右转用负角度，左转用正角度
                extent = -90 if sign == -1 else 90
                self.canvas.create_arc(c1x, min(c1y, c2y), c2x, max(c1y, c2y),
                                    start=start_angle, extent=extent,
                                    style=tk.ARC, outline="purple", width=2, tags="actual_route")

                # 计算圆弧终点：以圆心为参考，旋转90度后得到新位置
                start_angle_rad = math.atan2(y - center_y, x - center_x)
                new_angle = start_angle_rad + sign * math.pi/2
                x = center_x + radius * math.cos(new_angle)
                y = center_y + radius * math.sin(new_angle)
                # 更新机器人的朝向（无论前进还是倒车，都按照转向命令旋转90°）
                theta = theta + sign * turn_angle

                # Not sure for this 
                move_dist = 2 if forward else -2   # Forward dis after turning 
               
                x_new = x + move_dist * math.cos(theta)
                y_new = y + move_dist * math.sin(theta)
                self.draw_line_world(x, y, x_new, y_new, tag="actual_route", color="purple")
                x, y = x_new, y_new

                points.append((x, y))
                self.draw_point_world(x, y, tag="actual_route", color="red", size=4)


            elif cmd.startswith("SNAP"):
                # SNAP命令，不改变位置，可在当前点标记拍照
                self.draw_point_world(x, y, tag="actual_route", color="orange", size=8)
            elif cmd == "FIN":
                break
        # 标记小车头部：在最终位置绘制一个三角形表示前方
        self.draw_car_head(x, y, theta, tag="actual_route")
    
    
    # 辅助：将世界坐标转换为画布坐标
    def world_to_canvas(self, wx, wy):
        # 假设世界坐标原点在左下角，与网格一致
        cx = wx * self.cell_size
        cy = (self.grid_size - wy) * self.cell_size
        return cx, cy
    # 辅助：在世界坐标中绘制直线
    def draw_line_world(self, x1, y1, x2, y2, tag, color):
        cx1, cy1 = self.world_to_canvas(x1, y1)
        cx2, cy2 = self.world_to_canvas(x2, y2)
        self.canvas.create_line(cx1, cy1, cx2, cy2, fill=color, width=2, tags=tag)
    # 辅助：在世界坐标中绘制一个小圆点
    def draw_point_world(self, x, y, tag, color, size=3):
        cx, cy = self.world_to_canvas(x, y)
        self.canvas.create_oval(cx-size, cy-size, cx+size, cy+size,outline=color, fill="", tags=tag)
    # 辅助：绘制小车头部（三角形）
    def draw_car_head(self, x, y, theta, tag):
        # 车体长度
        L = 1.0  # 可调整
        # 小车前部的三个顶点
        # 以当前位置为三角形中心，朝向theta指示小车前部
        front = (x + L * math.cos(theta), y + L * math.sin(theta))
        left = (x + 0.5 * L * math.cos(theta + math.radians(150)),
                y + 0.5 * L * math.sin(theta + math.radians(150)))
        right = (x + 0.5 * L * math.cos(theta - math.radians(150)),
                 y + 0.5 * L * math.sin(theta - math.radians(150)))
        points = []
        for pt in (front, left, right):
            cx, cy = self.world_to_canvas(pt[0], pt[1])
            points.extend([cx, cy])
        self.canvas.create_polygon(points, fill="magenta", tags=tag)
    # 新增：点击按钮时调用该方法显示实际路线
    def show_actual_route(self):
        if hasattr(self, "commands"):
            self.draw_actual_route(self.commands)
        else:
            messagebox.showerror("Error", "请先生成路径并命令!")
if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingTestUI(root)
    root.mainloop()