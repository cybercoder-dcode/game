import pygame
import sys
import math
import random

pygame.init()

#vars
TRACK_WIDTH = 100
TRACK_RADIUS = TRACK_WIDTH // 2


screen_width, screen_height = 1200, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Racing")

WHITE = (255, 255, 255)
RED = (255,0,0)
clock = pygame.time.Clock()

#Car Class
class Car:
    def __init__(self, x, y):
        #Just plain old constants
        self.max_speed = 200
        self.accel = 3
        self.brake = 25
        self.friction = 3
        self.turn_rate = 0.1
        self.x = x
        self.speed = 0
        self.y = y
        self.angle = 0
        self.width = 40
        self.height = 20
    
    def draw(self, surface):
        # Create a rect surface
        car_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        car_surf.fill(RED)
        # Rotate it
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        # Center it on self.x, self.y
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect)

    def move(self, keys, dt):
        #simple movement script
        if not keys[pygame.K_w] and not keys[pygame.K_s]:
            if self.speed > 0:
                self.speed = max(0, self.speed - self.friction)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.friction)

        if keys[pygame.K_w]:
            self.speed += self.accel 
        if keys[pygame.K_s]:
            self.speed -= self.brake 

        self.speed = max(-self.max_speed/2, min(self.speed, self.max_speed))

        if abs(self.speed) > 0.1:
            turn = 0
            if keys[pygame.K_a]:
                turn -= 1
            if keys[pygame.K_d]:
                turn += 1
            self.angle += turn * self.turn_rate * (abs(self.speed) / self.max_speed)#radians

        self.x += math.cos(self.angle) * self.speed *dt
        self.y += math.sin(self.angle) * self.speed *dt 

font = pygame.font.SysFont(None, 24) #For looking at speed

car = Car(40,20)

def generate_points(center, radius_x, radius_y, number, variability):
    cx, cy = center
    points = []

    for i in range(number):
        angle = (2 * math.pi * i) / number
        radius_offset = random.uniform(-variability, variability)

        x = cx + math.cos(angle) * (radius_x + radius_offset)
        y = cy + math.sin(angle) * (radius_y + radius_offset)

        points.append((x, y))
    return points

points = generate_points(
    center=(600, 300),
    radius_x=450,
    radius_y=200,
    number=24,
    variability=50
)

def smooth_lines(points, samples = 10): 
    smooth = []
    n = len(points)

    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]

        for j in range(samples):
            t = j / samples
            t2 = t * t
            t3 = t2 * t

            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
                )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
            )

            smooth.append((x, y))
    return smooth

    #need some sort of interpolation -> Catmul something, Like beizer but for more points

smooth_points = smooth_lines(points, samples=20)

# Build the track once, before the loop
track_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

#Turns
for x, y in smooth_points:
    pygame.draw.circle(
        track_surface,
        (100, 100, 100),
        (round(x), round(y)),
        TRACK_RADIUS
    )


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    dt = clock.tick(60) / 1000.0   # only once
    keys = pygame.key.get_pressed()
    car.move(keys, dt)              # update first
    screen.fill(WHITE)
    screen.blit(track_surface, (0, 0))

    for point in points: #Junctions
        pygame.draw.circle(screen, RED, (int(point[0]), int(point[1])), 5)

        
    car.draw(screen)                # then draw
    speed_text = font.render(f"Speed: {car.speed:.1f}", True, (0, 0, 0))
    screen.blit(speed_text, (10, 10))
    pygame.display.flip()