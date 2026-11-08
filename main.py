import pygame
import sys
import math
import random

pygame.init()

# vars
TRACK_WIDTH = 80
TRACK_RADIUS = TRACK_WIDTH // 2
TRACK_COLOR = (100, 100, 100)
CHECKPOINT_COLOR = (255, 220, 0)

WORLD_WIDTH, WORLD_HEIGHT = 2400, 1200
ZOOM = 1.5

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()
view_width = int(screen_width / ZOOM)
view_height = int(screen_height / ZOOM)

pygame.display.set_caption("Car Racing")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
clock = pygame.time.Clock()


# Cam
def get_camera_offset(target_x, target_y):
    cam_x = target_x - view_width // 2
    cam_y = target_y - view_height // 2
    # Clamp so the view never leaves the world
    cam_x = max(0, min(cam_x, WORLD_WIDTH - view_width))
    cam_y = max(0, min(cam_y, WORLD_HEIGHT - view_height))
    return cam_x, cam_y


def world_to_screen(point, cam_x, cam_y):
    x, y = point
    return (int((x - cam_x) * ZOOM), int((y - cam_y) * ZOOM))


# Car Class
class Car:
    def __init__(self, x, y):
        # Just plain old constants
        self.max_speed = 300
        self.accel = 6
        self.brake = 10
        self.friction = 4
        self.turn_rate = 0.1
        self.x = x
        self.speed = 0
        self.y = y
        self.angle = 0
        self.width = 40
        self.height = 20

    def get_corners(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        half_width = self.width / 2
        half_height = self.height / 2
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        corners = []

        for local_x, local_y in [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]:
            world_x = x + local_x * cos_angle - local_y * sin_angle
            world_y = y + local_x * sin_angle + local_y * cos_angle
            corners.append((world_x, world_y))

        return corners

    def draw(self, surface, cam_x, cam_y):
        # Create a rect surface
        car_surf = pygame.Surface(
            (int(self.width * ZOOM), int(self.height * ZOOM)),
            pygame.SRCALPHA
        )
        car_surf.fill(RED)

        # Rotate it
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.angle))

        # Center it on self.x, self.y
        rect = rotated.get_rect(
            center=(
                int((self.x - cam_x) * ZOOM),
                int((self.y - cam_y) * ZOOM)
            )
        )
        surface.blit(rotated, rect)

    def move(self, keys, dt):
        # simple movement script
        if not keys[pygame.K_w] and not keys[pygame.K_s]:
            if self.speed > 0:
                self.speed = max(0, self.speed - self.friction)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.friction)

        if keys[pygame.K_w]:
            self.speed += self.accel
        if keys[pygame.K_s]:
            self.speed -= self.brake

        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))

        if abs(self.speed) > 0.1:
            turn = 0
            if keys[pygame.K_a]:
                turn -= 1
            if keys[pygame.K_d]:
                turn += 1
            self.angle += turn * self.turn_rate * (abs(self.speed) / self.max_speed)  # radians

        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt


font = pygame.font.SysFont(None, 24)  # For looking at speed


def generate_points(center, radius_x, radius_y, number, variability):
    cx, cy = center
    points = []
    for i in range(number):
        angle = (2 * math.pi * i) / number
        # First point is the fixed start point
        if i == 0:
            radius_offset = 0
        else:
            radius_offset = random.uniform(-variability, variability)
        x = cx + math.cos(angle) * (radius_x + radius_offset)
        y = cy + math.sin(angle) * (radius_y + radius_offset)
        points.append((x, y))
    return points


points = generate_points(
    center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2),
    radius_x=850,
    radius_y=380,
    number=24,
    variability=35
)


def smooth_lines(points, samples=20):
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
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            smooth.append((x, y))
    return smooth
    # need some sort of interpolation -> Catmul something, Like beizer but for more points


def distance_to_segment(point, segment_start, segment_end):
    px, py = point
    ax, ay = segment_start
    bx, by = segment_end
    ab_x = bx - ax
    ab_y = by - ay
    ab_length_squared = ab_x * ab_x + ab_y * ab_y

    if ab_length_squared == 0:
        return math.hypot(px - ax, py - ay)

    projection = ((px - ax) * ab_x + (py - ay) * ab_y) / ab_length_squared
    projection = max(0, min(1, projection))
    closest_x = ax + projection * ab_x
    closest_y = ay + projection * ab_y
    return math.hypot(px - closest_x, py - closest_y)


def point_is_on_track(point, centerline):
    for i in range(len(centerline)):
        start = centerline[i]
        end = centerline[(i + 1) % len(centerline)]
        distance = distance_to_segment(point, start, end)
        if distance <= TRACK_RADIUS:
            return True
    return False


def car_is_on_track(car, centerline):
    # Check the car's corners and centre
    test_points = car.get_corners()
    test_points.append((car.x, car.y))
    for point in test_points:
        if not point_is_on_track(point, centerline):
            return False
    return True


smooth_points = smooth_lines(points, samples=20)

# Starting position of the car by defining the certain point it always starts from
start_point = points[0]
next_point = points[1]
start_angle = math.atan2(next_point[1] - start_point[1], next_point[0] - start_point[0])

car = Car(start_point[0], start_point[1])
car.angle = start_angle
checkpoint_angle = start_angle + math.pi / 2

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    dt = clock.tick(60) / 1000.0   # only once
    keys = pygame.key.get_pressed()
    previous_x = car.x
    previous_y = car.y
    car.move(keys, dt)              # update first

    # simple check
    if not car_is_on_track(car, smooth_points):
        car.x = previous_x
        car.y = previous_y
        car.speed = 0

    cam_x, cam_y = get_camera_offset(car.x, car.y)
    screen.fill(WHITE)

    # Convert the track points to camera coordinates
    screen_points = [world_to_screen(point, cam_x, cam_y) for point in smooth_points]

    # Draw the track
    pygame.draw.lines(screen, TRACK_COLOR, True, screen_points, int(TRACK_WIDTH * ZOOM))

    # Round the track joins
    for point in screen_points:
        pygame.draw.circle(screen, TRACK_COLOR, point, int(TRACK_RADIUS * ZOOM))

    # Draw initialization checkpoint
    checkpoint_x = math.cos(checkpoint_angle) * TRACK_RADIUS
    checkpoint_y = math.sin(checkpoint_angle) * TRACK_RADIUS
    checkpoint_start = (start_point[0] - checkpoint_x, start_point[1] - checkpoint_y)
    checkpoint_end = (start_point[0] + checkpoint_x, start_point[1] + checkpoint_y)

    pygame.draw.line(
        screen,
        CHECKPOINT_COLOR,
        world_to_screen(checkpoint_start, cam_x, cam_y),
        world_to_screen(checkpoint_end, cam_x, cam_y),
        max(2, int(5 * ZOOM))
    )
    car.draw(screen, cam_x, cam_y)
    speed_text = font.render(f"Speed: {car.speed:.1f}", True, (0, 0, 0))
    screen.blit(speed_text, (10, 10))
    pygame.display.flip()