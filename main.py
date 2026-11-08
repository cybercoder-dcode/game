import pygame
import sys
import math

pygame.init()

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
        self.max_speed = 400
        self.accel = 5
        self.brake = 25
        self.friction = 10
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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    dt = clock.tick(60) / 1000.0   # only once
    keys = pygame.key.get_pressed()
    car.move(keys, dt)              # update first
    screen.fill(WHITE)
    car.draw(screen)                # then draw
    speed_text = font.render(f"Speed: {car.speed:.1f}", True, (0, 0, 0))
    screen.blit(speed_text, (10, 10))
    pygame.display.flip()