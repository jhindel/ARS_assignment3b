"""
author: Diego Di Benedetto
"""

from tkinter import *
import time
import numpy as np
from PIL import ImageTk, Image
import matplotlib.pyplot as plt


class World:
    # size of the environment
    min_X = 0
    max_X = 800
    min_Y = 0
    max_Y = 800

    dust_steps = 0

    # dictionary with different rooms
    obstacles = dict(rec=[[100, 100, 700, 100, 700, 700, 100, 700]],
                     double_rec=[[100, 100, 700, 100, 700, 700, 100, 700], [300, 300, 500, 300, 500, 500, 300, 500]],
                     room=[[100, 100, 700, 100, 700, 700, 100, 700],
                           [100, 200, 300, 200, 300, 250, 100, 250],
                           [500, 200, 600, 200, 600, 300, 500, 300],
                           [350, 500, 400, 500, 400, 700, 350, 700]],
                     star=[[400, 50, 500, 300, 700, 300, 525, 425, 600, 700, 400, 500, 200, 700, 275, 425, 100, 300, 300, 300]],
                     trapezoid=[[100, 100, 700, 200, 700, 600, 100, 700]],
                     double_trapezoid=[[100, 100, 700, 200, 700, 600, 100, 700], [300, 300, 500, 300, 500, 500, 300, 500]])

    def __init__(self, GUI, config=0):
        '''
        the configuration of the world is chosen with the parameter config(set in EA)
        GUI is a boolean variable, when set to true the world is displayed. the variable is dynamically changed
        in EA.run_generation
        '''
        self.world_configuration = config
        self.key = list(self.obstacles.keys())[self.world_configuration]
        self.display = GUI
        self.dust = np.ones((self.max_X, self.max_Y)) * 255
        # self.dust = np.full((self.max_X, self.max_Y), 1., dtype=int)
        if GUI:
            self.window = Tk()
            # self.window = Toplevel()
            self.window.title('World simulation')
            self.canvas = Canvas(self.window)
            self.draw_obstacles()
            self.init_dust()
            self.canvas.configure(width=self.max_X, height=self.max_Y)
            # self.canvas.pack(fill="both", expand=True)
            # print(self.canvas)

    def change_display(self, bool):
        self.display = bool
        if self.display:
            self.__init__(self.display, self.world_configuration)

    def init_agent(self, agent):
        self.canvas.pack()
        self.draw_robot(agent.get_position(), agent.get_radius())
        self.draw_sensors(agent.get_position(), agent.get_sensors())
        self.draw_velocity(agent.get_position())

    def get_dust_value(self):
        if self.display:
            try:
                #TODO change
                # pass
                self.window.destroy()
            except:
                pass
        # print("dust", int(np.count_nonzero(self.dust == 100)) / (self.max_Y * self.max_X))
        return int(np.count_nonzero(self.dust == 100)) / (self.max_Y * self.max_X)

    def init_dust(self):
        self.background_label = self.canvas.create_image(0, 0, image=None, anchor='nw', tag="A")
        self.canvas.tag_lower("A")

    def get_obstacles(self):
        return self.obstacles

    def draw_obstacles(self):
        for i, obstacle in enumerate(self.obstacles[self.key]):
            self.canvas.create_polygon(obstacle, outline='black', fill=("" if i == 0 else 'white'), width=1, tag=f"polygon{i}")
            self.canvas.tag_lower(f"polygon{i}")
            # print(i, obstacle)


    def draw_robot(self, position, radius):
        # print(position, radius)
        x0 = position[0] - radius
        y0 = position[1] - radius
        x1 = position[0] + radius
        y1 = position[1] + radius
        x2 = position[0] + radius * np.cos(position[2])
        y2 = position[1] + radius * np.sin(position[2])
        return [self.canvas.create_oval(x0, y0, x1, y1, tags='robot', fill="#C2DFFF"),
                self.canvas.create_line(position[0], position[1], x2, y2, tags='line')]

    def draw_sensors(self, position, sensors):
        for i, [x2, y2, value, _, _, _] in enumerate(sensors):
            # 16/20 at current position and 4/20 at end_point sensors
            self.canvas.create_text((16 / 20 * position[0] + 4 / 20 * x2), (16 / 20 * position[1] + 4 / 20 * y2),
                                    fill="black", font="Times 7",
                                    text=round(value), tags=f'sensor{i}')
            self.canvas.lift(f'sensor{i}')

    def draw_velocity(self, position):
        # draw the velocity values of the the motors
        self.canvas.create_text(position[0], position[1] + 12, fill="black", font="Times 10",
                                text="0", tags='velocity_l')
        self.canvas.create_text(position[0], position[1] - 12, fill="black", font="Times 10",
                                text="0", tags='velocity_r')

    def update_angle(self, direction_coord):
        # update line indicating direction from passed coordinates
        self.canvas.coords('line', direction_coord[0], direction_coord[1], direction_coord[2], direction_coord[3])
        self.canvas.lift('line')

    def update_sensors(self, position, sensors):
        for i, [x2, y2, value, _, _, _] in enumerate(sensors):
            # 16/20 at current position and 4/20 at end_point sensors
            self.canvas.itemconfig(f'sensor{i}', text=np.round(value))
            self.canvas.coords(f'sensor{i}', (16 / 20 * position[0] + 4 / 20 * x2),
                               (16 / 20 * position[1] + 4 / 20 * y2))
            # self.canvas.lift(f'sensor{i}')

    def update_velocity_display(self, position, velocity):
        # velocity left
        # add label 12 pixel from center
        x1 = position[0] + 12 * np.cos(position[2] - np.pi / 2)
        y1 = position[1] + 12 * np.sin(position[2] - np.pi / 2)
        self.canvas.itemconfig('velocity_l', text=np.round(velocity[1], 0))
        self.canvas.coords('velocity_l', x1, y1)
        # self.canvas.lift('velocity_l')
        # velocity right
        # add label 12 pixel from center
        x1 = position[0] + 12 * np.cos(position[2] + np.pi / 2)
        y1 = position[1] + 12 * np.sin(position[2] + np.pi / 2)
        self.canvas.itemconfig('velocity_r', text=np.round(velocity[0], 0))
        self.canvas.coords('velocity_r', x1, y1)
        # self.canvas.lift('velocity_r')

    def move_agent(self, x, y, agent):
        # update all objects in environment
        if self.dust_steps % 20 == 0:
            # print(self.dust_steps)
            self.draw_dust()
        self.dust_steps += 1
        self.canvas.move('robot', x, y)
        self.update_angle(agent.get_direction_coords())
        self.update_sensors(agent.get_position(), agent.get_sensors())
        self.update_velocity_display(agent.get_position(), agent.get_velocity())
        self.canvas.update_idletasks()
        self.canvas.after(1)

    def update_dust(self, position, radius=20):
        # calculate the amount of dust collected at current position
        top = round(max(self.min_Y, position[0] - radius))
        bottom = round(min(self.max_Y, position[0] + radius))
        left = round(max(self.min_X, position[1] - radius))
        right = round(min(self.max_X, position[1] + radius))
        for y in range(top, bottom):
            for x in range(left, right):
                if self.inside_circle(position, [y, x], radius):
                    self.dust[y, x] = 100
        # self.dust = self.dust.T
        # print(self.dust)

    def inside_circle(self, position, point, radius):
        dx = position[0] - point[0]
        dy = position[1] - point[1]

        distance_squared = dx * dx + dy * dy
        return distance_squared <= radius * radius

    def draw_dust(self):
        dust = self.dust.copy()
        self.im = ImageTk.PhotoImage(image=Image.fromarray(dust.T), master=self.window)
        self.canvas.itemconfigure(self.background_label, image=self.im)
        # self.background_label.configure(image=self.im)

    def display_dust(self):
        fig, ax = plt.subplots()
        ax.matshow(self.dust.T, cmap=plt.cm.Blues)
        plt.title(f"world {self.key} dust {self.get_dust_value()}")
        plt.show()
        time.sleep(3)
        plt.close(fig)