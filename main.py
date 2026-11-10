import pygame
import sys
import math
import random

pygame.init()

'''
So out code will output our ai code will have inputs being:
    7 raycasts at 0,180,90 and 30,60 both sides
    Our velocity
    angle
    distance to next waypoint

    Output will simply be left right up down

    We will rank fitness by checkpoints reached in my average time with a bonus for high speed?

    so task 1 - modify to accept inputs from left right forward back
        Task 1 Done!
    will use argmax in code so as to accept only 1 input
'''

# vars
TRACK_WIDTH = 120
TRACK_RADIUS = TRACK_WIDTH // 2
TRACK_COLOR = (100, 100, 100)
CHECKPOINT_COLOR = (255, 220, 0)
next_waypoint_index = 1   
laps_completed = 0
WAYPOINT_RADIUS = TRACK_RADIUS  
current_speed = 0
last_waypoint_index = 0   
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
        self.max_speed = 500
        self.accel = 3
        self.brake = 4
        self.friction = 1
        self.turn_rate = 0.1
        self.x = x
        #self.speed = 0
        self.y = y
        self.angle = 0
        self.width = 40
        self.height = 20
        self.vx = 0
        self.vy = 0
        self.grip = 0.05
    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)
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
            (-half_width, half_height)]:
            
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

    def move(self, move_state, dt):
        # simple movement script
        turn = 0
        if move_state["left"]:
            turn -= 1
        if move_state["right"]:
            turn += 1

        current_speed = math.hypot(self.vx, self.vy)
        if abs(current_speed) > 0.1:
                    self.angle += turn * self.turn_rate * (abs(current_speed) / self.max_speed)  # radians

        forward_x = math.cos(self.angle)
        forward_y = math.sin(self.angle)

        # We break into two components the angle and increase both by accel
        # Vese toh not needed but makes rest simpler via 2 components and is essential for drifting

        if move_state["accel"]:
            self.vx += forward_x * self.accel
            self.vy += forward_y * self.accel
        if move_state["brake"]:
            self.vx -= forward_x * self.brake
            self.vy -= forward_y * self.brake

        #friction
        if not move_state["accel"] and not move_state["brake"]:
            self.vx *= 0.95
            self.vy *= 0.95
        # Scaling the speeds instead of simply clamping it
        max_speed = self.max_speed
        if current_speed > max_speed:
            scale = max_speed / current_speed
            self.vx *= scale
            self.vy *= scale
            current_speed = max_speed

        if current_speed > 150: #less grip at high speed
            effective_grip = self.grip
        else: 
            effective_grip = self.grip * 0.4

        current_speed = math.hypot(self.vx, self.vy)

        if abs(current_speed) > 0.1:
            desired_vx = forward_x * current_speed
            desired_vy = forward_y * current_speed
            self.vx += (desired_vx - self.vx) * effective_grip
            self.vy += (desired_vy - self.vy) * effective_grip

        self.x += self.vx * dt
        self.y += self.vy * dt


font = pygame.font.SysFont(None, 24)  # For looking at speed + Progress


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

def draw():
    screen.fill(WHITE)

    # Convert the track points to camera coordinates
    screen_points = [world_to_screen(point, cam_x, cam_y) for point in smooth_points]

    # Draw the track
    pygame.draw.lines(screen, TRACK_COLOR, True, screen_points, int(TRACK_WIDTH * ZOOM))

    # Round the track joins
    for point in screen_points:
        pygame.draw.circle(screen, TRACK_COLOR, point, int(TRACK_RADIUS * ZOOM))

    pygame.draw.circle(
        screen, (0, 0, 255),
        world_to_screen(points[next_waypoint_index], cam_x, cam_y),
        int(10 * ZOOM)
    )    
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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_r:
                
                waypoint = points[last_waypoint_index]
                towards = points[(last_waypoint_index+1) % len(points)]

                car.x, car.y = waypoint
                car.angle = math.atan2(towards[1] - waypoint[1], towards[0] - waypoint[0])
                car.vx = car.vy = 0

                next_waypoint_index = (last_waypoint_index + 1) % len(points)
            if event.key == pygame.K_RETURN:
                
                points = generate_points(
                    center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2),
                    radius_x=850,
                    radius_y=380,
                    number=24,
                    variability=35
                )
                cam_x, cam_y = get_camera_offset(car.x, car.y)
                smooth_points = smooth_lines(points, samples=20)
                start_point = points[0]
                next_point = points[1]
                start_angle = math.atan2(
                    next_point[1] - start_point[1],
                    next_point[0] - start_point[0]
                )

                car = Car(start_point[0], start_point[1])
                car.angle = start_angle
                checkpoint_angle = start_angle + math.pi / 2
                car.x, car.y = start_point
                car.angle = start_angle
                last_waypoint_index = 0

                car.vx = car.vy = 0

                next_waypoint_index = 1
                laps_completed = 0
    dt = clock.tick(60) / 1000.0   # only once
    keys = pygame.key.get_pressed()
    move_state = {
        "left":  keys[pygame.K_a],
        "right": keys[pygame.K_d],
        "accel": keys[pygame.K_w],
        "brake": keys[pygame.K_s],
    }
    
    
    previous_x = car.x
    previous_y = car.y
    car.move(move_state, dt)              # update first

    # simple check
    if not car_is_on_track(car, smooth_points):
        car.x = previous_x
        car.y = previous_y
        car.vx = 0
        car.vy = 0

#For Waypoints
    
    target = points[next_waypoint_index]

    if math.hypot(car.x - target[0], car.y - target[1]) < WAYPOINT_RADIUS:
        last_waypoint_index = next_waypoint_index
        next_waypoint_index = (next_waypoint_index + 1) % len(points)
        if next_waypoint_index == 1:
            laps_completed += 1
    

    cam_x, cam_y = get_camera_offset(car.x, car.y)
    draw()  
    car.draw(screen, cam_x, cam_y)

    progress = laps_completed * len(points) + next_waypoint_index
    speed_text = font.render(f"Speed: {car.speed:.1f}", True, (0, 0, 0))
    progress_text = font.render(f"Progress: {progress}", True, (0, 0, 0))
    screen.blit(speed_text, (10, 10))
    screen.blit(progress_text, (10, 30))
    pygame.display.flip()