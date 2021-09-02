"""
author: Diego Di Benedetto, Julia Hindel
"""

import math
import vector_calculation
import numpy as np
import random
from shapely.geometry import Point, Polygon


def chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class Robot:
    # robot features
    radius = 20.
    n_sensors = 12
    sensor_range = 200
    distance_threshold = 100
    sensor_threshold_wall_collision = 10
    init_random = True
    init_position = np.array([150., 325., 0.])
    # movement features
    max_speed = 10

    def __init__(self, world):

        # parameters used for calculating rotated position
        self.w = 0
        self.icc = np.zeros((2, 0))
        self.l = 20
        self.r = 0
        # list of detected walls
        self.detected_walls = []
        self.world = world
        # position
        self.position = self.init_position
        if self.init_random:
            self.init_robot_position()
        # coordinates of line indicating direction
        self.direction_coords = np.array([self.position[0], self.position[1], self.position[0] + 20, self.position[1]])
        # velocity of left and right wheel
        self.velocity = np.array([0., 0.])
        # list of sensors
        self.sensors = self.set_sensors()
        # boolean determining if checks for intersection are needed
        self.intersection = False
        if self.world.display:
            self.world.init_agent(self)

    def init_robot_position(self):
        # initialize random position in current environment
        # successively checks whether the robot position is contained/intersected with obstacles
        obstacles = []
        found = False
        coords = self.world.obstacles[self.world.key]
        for i in range(len(coords)):
            # print(len(coords[i]))
            coords_formatted = list(chunks(coords[i], 2))
            # print('formatted', coords_formatted)
            poly = Polygon(coords_formatted)
            obstacles.append(poly)
        while not found:
            found = True
            x_coord = random.randint(self.world.min_X + self.radius, self.world.max_X - self.radius)
            y_coord = random.randint(self.world.min_Y + self.radius, self.world.max_Y - self.radius)
            point = Point(x_coord, y_coord)
            circle = point.buffer(self.radius*4)
            # print(circle)
            if len(obstacles) == 1:
                if not obstacles[0].contains(circle):
                    found = False
                continue
            for i in range(len(obstacles)-1):
                # print(obstacles[i+1], obstacles[0].contains(circle), obstacles[i+1].contains(circle), obstacles[i+1].intersects(circle), x_coord, y_coord)
                if (not obstacles[0].contains(circle)) or (obstacles[i+1].contains(circle)) or (obstacles[i+1].intersects(circle)):
                    found = False
                    # print("setting to False")
        self.position = [x_coord, y_coord, 0.]

    def get_position(self):
        return self.position

    def get_radius(self):
        return self.radius

    def get_direction_coords(self):
        return self.direction_coords

    def get_velocity(self):
        return self.velocity

    def get_sensors(self):
        return self.sensors

    def set_sensors(self):
        # create list of sensors including
        partition = 360 / self.n_sensors
        sensors = []
        for sensor_n in range(self.n_sensors):
            angle = partition * sensor_n
            x2 = self.position[0] + (self.sensor_range + self.radius) * np.cos(angle * np.pi / 180)
            y2 = self.position[1] + (self.sensor_range + self.radius) * np.sin(angle * np.pi / 180)
            # end_coordinates of sensor (x, y), distance to next wall, intersection point of next wall and sensor,
            # sensor_id
            sensors.append([x2, y2, self.sensor_range, -1, -1, sensor_n])
        return np.array(sensors)

    def update_velocity(self, left_wheel, right_wheel):
        # print("update weights", left_wheel, right_wheel)
        self.velocity[1] = left_wheel * self.max_speed
        self.velocity[0] = right_wheel * self.max_speed
        self.update()
        return self.punish_collisions()

    """
    in every timestep apply certain transformations
    """

    def update(self):
        # calculate new position and update sensor values
        new_position = self.timestep_movement()
        temp_sensors = self.check_sensors(new_position)
        temp_sensors = self.update_boundaries(self.world.obstacles[self.world.key], temp_sensors, new_position)
        # in case of an intersection
        if self.intersection:
            # calculate intermediate time step until robot touches wall
            time = self.time_to_wall(new_position)
            # if timestep between 0 and 1
            if np.isfinite(time) and (0 <= time <= 1.):
                # calculate intermediate position and timestep t
                intermediate_position = self.timestep_movement(time)
                # calculate new position with sliding and update sensors
                new_position = self.apply_transformation(intermediate_position)
                temp_sensors = self.check_sensors(new_position)
                temp_sensors = self.update_boundaries(self.world.obstacles[self.world.key], temp_sensors, new_position)
            self.intersection = False
        # update coordinates of position, sensors and direction indicator
        self.update_direction_coords(new_position)
        old_position = self.position.copy()
        self.position = new_position.copy()
        self.sensors = temp_sensors.copy()
        # make change in GUI application
        self.world.update_dust(self.get_position(), self.get_radius())
        if self.world.display:
            self.world.move_agent(self.position[0] - old_position[0], self.position[1] - old_position[1], self)

    def punish_collisions(self):
        temp_fitness = 0
        filtered_sensors = [i for i in self.sensors if i[2] < self.sensor_threshold_wall_collision]
        temp_fitness += ((self.n_sensors - len(filtered_sensors))/self.n_sensors)*2
        if (self.velocity[0] > 0 and self.velocity[1] < 0) or (self.velocity[0] < 0 and self.velocity[1] > 0):
            temp_fitness += ((self.max_speed*2) - abs(self.velocity[0] - self.velocity[1]))/(self.max_speed*2)
        else:
            temp_fitness += 1
        if (self.velocity[0] < 0) and (self.velocity[1] < 0):
            temp_fitness += ((self.max_speed*2) - abs(self.velocity[0] + self.velocity[1]))/(self.max_speed*2)
        else:
            temp_fitness += 1
        if temp_fitness > 4:
            print("______ERROR: punish > 3______")
            print("sensors", (self.n_sensors - len(filtered_sensors))/self.n_sensors)
            print("velocity", ((self.max_speed*2) - abs(self.velocity[0] - self.velocity[1]))/(self.max_speed*2))
            print("velocity2", ((self.max_speed*2) - abs(self.velocity[0] + self.velocity[1]))/(self.max_speed*2))
        # print(punish/3)
        return temp_fitness/4

    """
    reaction to wall collision
    """

    def find_neighbouring_sensor(self, sensor_point, sensor_new):
        # find neighbouring sensor which is also below the distance_threshold and return neighbour with smaller distance
        x = sensor_point[-1]
        x_increment = (x + 1) % self.n_sensors
        x_decrement = (x - 1) % self.n_sensors
        increment_sensor = np.array([])
        decrement_sensor = np.array([])
        sensor_new = np.squeeze(sensor_new)
        if np.any(sensor_new[:, 5] == x_increment):
            increment_sensor = np.squeeze(self.sensors[self.sensors[:, 5] == x_increment])[3:6]
        if np.any(sensor_new[:, 5] == x_decrement):
            decrement_sensor = np.squeeze(self.sensors[self.sensors[:, 5] == x_decrement])[3:6]
        if increment_sensor.size != 0:
            if decrement_sensor.size != 0:
                if increment_sensor[2] < decrement_sensor[2]:
                    return increment_sensor
                else:
                    return decrement_sensor
            else:
                return increment_sensor
        else:
            return decrement_sensor

    def time_to_wall(self, new_position):
        # calculate time until robot touches a wall
        time_list = [np.inf]
        self.detected_walls = []
        # sort sensors according to distance from wall
        new_sensors = self.sensors.copy()
        new_sensors = np.array([i for i in new_sensors if i[2] < self.distance_threshold])
        sensor_new_sorted = np.array(sorted(new_sensors, key=lambda x: x[2]))
        while len(sensor_new_sorted) > 2:
            # remove one sensor from list
            sensor_point0 = sensor_new_sorted[0, 3:6]
            sensor_new_sorted = np.delete(sensor_new_sorted, 0, 0)
            # find neighbour sensor which is also below the threshold
            sensor_point01 = self.find_neighbouring_sensor(sensor_point0, sensor_new_sorted)
            # if it exists, calculate time until robot hits wall consisting of both sensor coordinates
            if sensor_point01.size != 0:
                index = np.where(sensor_new_sorted[:, -1] == sensor_point01[-1])
                sensor_new_sorted = np.delete(sensor_new_sorted, index, 0)
                time1 = self.time_calculation(sensor_point0, sensor_point01, new_position)
                if time1 < 0 or time1 > 1:
                    time1 = np.inf
                else:
                    self.detected_walls.append((sensor_point0, sensor_point01))
                time_list.append(time1)
        return min(time_list)

    def time_calculation(self, sensor_point0, sensor_point1, new_position):
        # calculate time until robot hits the wall
        point_on_wall = vector_calculation.get_point(sensor_point0, sensor_point1, self.position)
        v_normal = vector_calculation.normalize(vector_calculation.vector_between_points(point_on_wall, self.position))
        dx = new_position[:2] - self.position[:2]
        # print('v_norm ',v_normal, ' self pos ',self.position[:2], ' new pos ',new_position, ' sensor ', sensor_point0[:2], dx, np.dot(dx, v_normal))
        time = 0
        with np.errstate(divide='raise'):
            try:
                time = (self.radius - np.dot(self.position[:2], v_normal) + np.dot(sensor_point0[:2], v_normal)) \
               / np.dot(dx, v_normal)
            except FloatingPointError:
                pass
        return time

    def check_sensors(self, new_position):
        # update the position of the sensors according to the new angle and position
        temp_sensors = []
        angle = new_position[2]
        old_angle = self.position[2]
        for i in range(self.n_sensors):
            coordinates = self.sensors[i]
            old_sensor_angle = math.atan2(coordinates[1] - self.position[1], coordinates[0] - self.position[0])
            x2 = new_position[0] + ((self.sensor_range + self.radius) * np.cos(angle - old_angle + old_sensor_angle))
            y2 = new_position[1] + ((self.sensor_range + self.radius) * np.sin(angle - old_angle + old_sensor_angle))
            temp_sensors.append([x2, y2, self.sensor_range, -1, -1, i])
        return np.array(temp_sensors)

    def update_boundaries(self, obstacles, sensors, new_position):
        # create boundary lines of objects
        lines = []
        for obstacle in obstacles:
            lines.append(obstacle[-2:] + obstacle[:2])
            for i in range(len(obstacle) - 2):
                if i % 2 == 0:
                    lines.append(obstacle[i:i + 4])


        for i in range(len(sensors)):
            for line in lines:
                point = vector_calculation.line_intersect(*line, new_position[0], new_position[1], sensors[i][0],
                                                          sensors[i][1])
                if point:
                    # distance from new_position to line
                    d = np.sqrt(pow(point[0] - new_position[0], 2) + pow(point[1] - new_position[1], 2)) - self.radius
                    if d < sensors[i][2]:
                        sensors[i][2] = d
                        # safe coordinates of line intersection if smaller than threshold
                        if d < self.distance_threshold:
                            sensors[i][3] = point[0]
                            sensors[i][4] = point[1]
                            self.intersection = True
        return sensors

    def apply_transformation(self, intermediate_position):
        # apply transformation (if touching wall in next step)
        if len(self.detected_walls) == 1:
            return self.calculate_transformed_position(self.detected_walls[0][0], self.detected_walls[0][1],
                                                       intermediate_position)
        else:
            pos = self.calculate_transformed_position(self.detected_walls[0][0], self.detected_walls[0][1],
                                                      intermediate_position)
            return self.calculate_transformed_position(self.detected_walls[1][0], self.detected_walls[1][1], pos)

    def calculate_transformed_position(self, sensor_point0, sensor_point1, intermediate_position):
        # calculate transformed position
        point_perpendicular_to_center = vector_calculation.get_point(sensor_point0, sensor_point1, self.position)
        vector_center_to_wall_perpendicular = vector_calculation.vector_between_points(self.position,
                                                                                       point_perpendicular_to_center)
        point_perpendicular_to_new_center = vector_calculation.get_point(sensor_point0, sensor_point1,
                                                                         intermediate_position)
        new_center_transformed = vector_calculation.shift_point_with_vector(point_perpendicular_to_new_center,
                                                                            np.negative(
                                                                                vector_center_to_wall_perpendicular))
        return np.append(new_center_transformed, intermediate_position[2])

    """
    used for calculation of new position
    """

    def timestep_movement(self, time=1):
        # calculate forces and change velocity of both motors
        self.update_r()
        self.update_w()
        self.calculate_icc()
        return self.calculate_position(time)

    def update_r(self):
        if self.velocity[1] != self.velocity[0]:
            self.r = (self.l / 2) * ((self.velocity[0] + self.velocity[1]) / (self.velocity[1] - self.velocity[0]))
        else:
            self.r = 99999999

    def update_w(self):
        self.w = (self.velocity[1] - self.velocity[0]) / self.l

    def calculate_icc(self):
        self.icc = [self.position[0] - self.r * np.sin(self.position[2]),
                    self.position[1] + self.r * np.cos(self.position[2])]

    def calculate_position(self, time):
        # according to formula
        temp_position = self.position.copy()
        delta = 1
        if self.w == 0:
            temp_position[0] = np.round(
                self.position[0] + ((self.velocity[0] + self.velocity[1]) / 2) * np.cos(self.position[2]))
            temp_position[1] = np.round(
                self.position[1] + ((self.velocity[0] + self.velocity[1]) / 2) * np.sin(self.position[2]))
        else:
            temp_position = np.array([[np.cos(self.w * delta * time), -np.sin(self.w * delta * time), 0],
                                      [np.sin(self.w * delta * time), np.cos(self.w * delta * time), 0],
                                      [0, 0, 1]]).dot(
                np.array([[self.position[0] - self.icc[0]], [self.position[1] - self.icc[1]],
                          [self.position[2]]])) \
                            + np.array([[self.icc[0]], [self.icc[1]], [self.w * delta * time]])
        temp_position = np.squeeze(temp_position)
        return temp_position

    """
    update line indicating direction
    """

    def update_direction_coords(self, new_position):
        x0 = new_position[0]
        y0 = new_position[1]
        x1 = x0 + self.radius * np.cos(new_position[2])
        y1 = y0 + self.radius * np.sin(new_position[2])
        self.direction_coords = [x0, y0, x1, y1]
