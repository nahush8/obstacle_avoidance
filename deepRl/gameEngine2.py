#!/usr/bin/python
import random
import math
import numpy as np
import time
import pygame
from pygame.color import THECOLORS
import pymunk
from pymunk.vec2d import Vec2d
#from pymunk.pygame_util import draw
import pymunk.pygame_util
# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True



class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False
        self.car_velocity = 50
        self.numOflasersData = 10
        self.spread = 10
        self.distance = 10

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(150, 150, 0.5)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['white']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        #counter-clockwise starting from left bottom
        a1 = 100
        b1 = 100

        a2 = 950
        b2 = 100

        a3 = 950
        b3 = 650

        a4 = 100
        b4 = 650

        self.obstacles = []
        self.obstacles.append(self.create_obstacle(a4,b4 ,a1,b1, 10)) #left vertical
        self.obstacles.append(self.create_obstacle(a1,b1 ,a2,b2, 10)) #bottom horizontal
        self.obstacles.append(self.create_obstacle(a2,b2 ,a3,b3, 10)) #right vertical
        self.obstacles.append(self.create_obstacle(a3,b3 ,a4,b4, 10)) #top horizontal
        

        gap = 100
        self.obstacles.append(self.create_obstacle(a4+gap,b4-gap ,a1+gap,b1+gap, 10)) #left vertical
        self.obstacles.append(self.create_obstacle(a2-gap,b2+gap ,a3-gap,b3-gap, 10)) #right vertical

        self.obstacles.append(self.create_obstacle(a1+gap,b1+gap ,a1+gap+200,b2+gap, 10)) #bottom horizontal left
        self.obstacles.append(self.create_obstacle(a1+gap+400,b1+gap ,a2-gap,b2+gap, 10)) #bottom horizontal right

        self.obstacles.append(self.create_obstacle(a1+gap,b4-gap ,a1+gap+200,b3-gap, 10)) #top horizontal left
        self.obstacles.append(self.create_obstacle(a1+gap+400,b4-gap ,a2-gap,b3-gap, 10)) #top horizontal right

        self.obstacles.append(self.create_obstacle(a1+gap+200,b2+gap ,a1+gap+200,b4-gap, 10)) #verticle mid left
        self.obstacles.append(self.create_obstacle(a1+gap+400,b2+gap ,a1+gap+400,b3-gap, 10)) #verticle mid right
        #self.obstacles.append(self.create_obstacle(a1+gap+200,b2+gap ,a1+gap+400,b2+gap, 10)) #bottom mid horizontal
        self.obstacles.append(self.create_obstacle(a1+gap+200,b2+gap ,a1+gap+400,b2, 10)) #diagonal
        #self.obstacles.append(self.create_obstacle(a2-gap,b3-gap ,a2,b3-gap, 10))
        #self.obstacles.append(self.create_obstacle(700, 200, 125))
        #self.obstacles.append(self.create_obstacle(600, 600, 35))

        # Create a cat.
        #self.create_cat()
        '''
        self.create_incentive()
        '''
        
    def create_obstacle(self, x1,y1, x2,y2, r):
        #c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        '''
        c_body = pymunk.Body(1000, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["white"]
        self.space.add(c_body, c_shape)
        return c_body
        '''
        cp = pymunk.body.Body()
        # STATIC OBSTACLE, IGNORE MASS AND MOMENT
        c_body = pymunk.Body(10000, pymunk.inf,body_type=cp.STATIC)
        #vertices = [(200.0,200.0),(700.0,200.0),(700.0,600.0),(200.0,600.0)]
        c_shape = pymunk.Segment(c_body,(x1,y1),(x2,y2),radius=r)
        c_shape.elasticity = 0.0
        #c_body.position = x, y
        c_shape.color = THECOLORS["white"]
        self.space.add(c_body, c_shape)
        return c_body
    '''
    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)
    '''
    '''
    def create_incentive(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.incentive_body = pymunk.Body(1, inertia)
        self.incentive_body.position = 100, height - 200
        self.incentive_shape = pymunk.Circle(self.incentive_body, 10)
        self.incentive_shape.color = THECOLORS["blue"]
        self.incentive_shape.elasticity = 1.0
        self.incentive_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.incentive_body.angle)
        self.space.add(self.incentive_body, self.incentive_shape)
    '''

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 10)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 0.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        #self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        # Get the current location and the readings there.

        if action == 0:  # Turn left.
            self.car_body.angle -= .2
            self.car_velocity = 40
        elif action == 1:  # Turn right.
            self.car_body.angle += .2
            self.car_velocity = 40
        elif action == 2:  # Turn right.
            self.car_body.angle += 0.0
            self.car_velocity = 40
        elif action == 3:
            self.car_velocity = -20
        '''
        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()
        '''
        # Move cat
        '''
        if self.num_steps % 5 == 0:
            self.move_cat()
        '''

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        #self.car_body.velocity = 100 * driving_direction
        self.car_body.velocity = self.car_velocity * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        #pymunk.pygame_util.draw(screen, self.space)
        options = pymunk.pygame_util.DrawOptions(screen)
        self.space.debug_draw(options)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        state = np.array([readings])

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            #reward = -500
            reward = -5
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            #reward = -5 + int(self.sum_readings(readings) / 10)
            reward = int(self.sum_readings(readings))
            #reward = 1
        self.num_steps += 1
        return reward, state

    '''
    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            #speed = 0
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction
    '''
    '''
    def move_cat(self):
        speed = random.randint(20, 200)
        #speed = random.randint(0, 10)
        #speed = 0 
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction
    '''
    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1: #or readings[3] == 1 or readings[4] == 1 or readings[5] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            #self.car_body.velocity = -100 * driving_direction
            #self.car_body.velocity = -self.car_velocity * driving_direction
            #self.car_body.position = 200,200
            self.crashed = False
            
            for i in range(10):
                #self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["red"])  # Red is scary!
                #draw(screen, self.space)
                options = pymunk.pygame_util.DrawOptions(screen)
                self.space.debug_draw(options)
                self.space.step(1./10)
                #if draw_screen:
                   #pygame.display.flip()
                clock.tick()
            

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle1 = arm_left
        #arm_middle2 = arm_middle1
        #arm_middle3 = arm_middle2
        #arm_middle4 = arm_middle3
        arm_right = arm_middle1

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle1, x, y, angle, 0.0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))
        #readings.append(self.get_arm_distance(arm_middle3, x, y, angle, -0.25))
        #readings.append(self.get_arm_distance(arm_middle4, x, y, angle, -0.5))
        #readings.append(self.get_arm_distance(arm_right, x, y, angle, -1.0))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = self.spread  # Default spread.
        distance = self.distance  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(0, self.numOflasersData):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    prev_state = [[2,2,2]]
    while True:
        currReward, state = game_state.frame_step((random.randint(0, 2)))
        time.sleep(0.5)
        print "prev State"
        print prev_state
        print "current_state:"
        print state
        print "------"

        prev_state = state

