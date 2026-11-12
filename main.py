import pygame
import sys
import math
import random
import pickle
import argparse
import neat
from neat.nn import FeedForwardNetwork
import nn_viz

random.seed(19)

pygame.init()

'''
 So out code will output our ai code will have inputs being:
    7 raycasts at 0,180,90 and 30,60 both sides
    current speed
    angle offset to angle to next waypoint
    distance to next waypoint

    Output will simply be left right up down

    We will rank fitness by checkpoints reached in my average time with a bonus for high speed? + moving along the track

    so task 1 - modify to accept inputs from left right forward back
        Task 1 Done!
    Task 2 form raycasts
        Task 2 Done!
    Task 3 Record speed, current heading to next waypoint and distance to next waypoint
        angle done, speed done, dist is just hypot
        Task 3 Done!
    Task 4 implement Fitness
        Done!
    Task 5 Implement Neat + On crash reset
          Yes SUCCESSFULLY TRAINED 99% of thw way there!!

    Neat eveolves both the weights and the structure ie layers etc so is perfect for something so random

    will use argmax in code so as to accept only 1 input

    Treat turning and throttle as two separate argmax/binary decisions (one output group for {left, right, none}, another for {accel, brake, none}) — lets you combine them
'''

TRACK_WIDTH = 130
TRACK_RADIUS = TRACK_WIDTH // 2
TRACK_COLOR = (72, 72, 78)          # darker asphalt grey, reads better against grass
CHECKPOINT_COLOR = (255, 220, 0)    # kept for backwards-compat, checkered line used for drawing now
next_waypoint_index = 1
laps_completed = 0
WAYPOINT_RADIUS = TRACK_RADIUS

last_waypoint_index = 0

WORLD_WIDTH, WORLD_HEIGHT = 2400, 1200
ZOOM = 1.5
RAY_ANGLES = [-90, -45, 0, 45, 90, 180]
MAX_RAY_DIST = 400
NEAR_WINDOW = 20  # how many centerline segments either side of the car's nearest point to check

# Names for each slot of the observation vector / output vector, in the exact
# order ai_values()/translator() use them. Fed into the NN viz panel so each
# neuron gets a readable label instead of just a coloured dot.
INPUT_LABELS = [f"Ray {a}°" for a in RAY_ANGLES] + ["Angle→WP", "Speed", "Dist→WP"]
OUTPUT_LABELS = ["Left", "Straight", "Right", "Accel", "Coast", "Brake"]

car_near_index = 0
net_speed = 0
frame_count = 0
frame_since_waypoint = 0
STALL = 200
episode_fitness = None
can_move = True
episode_over = False
crashed = False
MAX_FRAMES = 12000
RENDER = False
DEBUG = False
DEBUG_EVERY = 30  # print every N frames

FULLSCREEN = False
WINDOWED_SIZE = (1200, 800)

def set_display_mode(fullscreen):
    global screen, screen_width, screen_height, view_width, view_height, FULLSCREEN
    FULLSCREEN = fullscreen
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)  # (0,0) = use native desktop res
    else:
        screen = pygame.display.set_mode(WINDOWED_SIZE)
    screen_width, screen_height = screen.get_size()
    view_width = int(screen_width / ZOOM)
    view_height = int(screen_height / ZOOM)

set_display_mode(False)
pygame.display.set_caption("Car Racing")

WHITE = (255, 255, 255)
RED = (255, 0, 0)              # champion / default car colour
PLAYER_BLUE = (40, 110, 240)   # player's car + HUD label colour
GRASS_GREEN = (56, 142, 60)
LANE_WHITE = (235, 235, 235)
clock = pygame.time.Clock()


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
class Car:
    def __init__(self, x, y, color=None):
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
        self.grip = 0.07
        self.color = color if color is not None else RED  # lets replay tell cars apart

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
        car_surf.fill(self.color)

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
        if can_move == True:
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
    #Oh shit this is just the distance formula ez
    projection = ((px - ax) * ab_x + (py - ay) * ab_y) / ab_length_squared
    projection = max(0, min(1, projection))
    closest_x = ax + projection * ab_x
    closest_y = ay + projection * ab_y
    return math.hypot(px - closest_x, py - closest_y)

def find_nearest_index(point, centerline):
    px, py = point
    best_i = 0 #More optimization as waypoints was causing this to slow down a lot
    best_dist_sq = float("inf")
    for i, (cx, cy) in enumerate(centerline):
        dist_sq = (px - cx) ** 2 + (py - cy) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_i = i
    return best_i

def point_is_on_track(point, centerline, near_index=None, window=NEAR_WINDOW):
    n = len(centerline)
    if near_index is None:
        indices = range(n)
    else:
        indices = [(near_index + offset) % n for offset in range(-window, window + 1)] #Simply sample

    for i in indices:
        start = centerline[i]
        end = centerline[(i + 1) % n]
        distance = distance_to_segment(point, start, end)
        if distance <= TRACK_RADIUS:
            return True
    return False

def car_is_on_track(car, centerline, near_index=None):
    # Check the car's corners and centre
    test_points = car.get_corners()
    test_points.append((car.x, car.y))
    for point in test_points:
        if not point_is_on_track(point, centerline, near_index):
            return False
    return True
def cast_ray(car, angle_offset_deg, centerline, near_index, max_dist=MAX_RAY_DIST, step=10):
    angle = car.angle + math.radians(angle_offset_deg)
    dx = math.cos(angle)
    dy = math.sin(angle)
    #throw the rays outward
    dist = 0
    while dist < max_dist:
        test_x = car.x + dx * dist
        test_y = car.y + dy * dist
        if not point_is_on_track((test_x, test_y), centerline, near_index):
            return dist
        dist += step
    return max_dist

#Call ray_cats
def get_raycast(car, centreline, near_index):
    #near_index = find_nearest_index((car.x, car.y), centreline)  # NEW: computed once, reused for all rays
    return [cast_ray(car, a, centreline, near_index) / MAX_RAY_DIST for a in RAY_ANGLES]

smooth_points = smooth_lines(points, samples=20)

# Starting position of the car by defining the certain point it always starts from
start_point = points[0]
next_point = points[1]
start_angle = math.atan2(next_point[1] - start_point[1], next_point[0] - start_point[0])

car = Car(start_point[0], start_point[1])
car.angle = start_angle
checkpoint_angle = start_angle + math.pi / 2
def draw_track(surface, cam_x, cam_y, waypoint_index):
    surface.fill(GRASS_GREEN)

    screen_points = [world_to_screen(point, cam_x, cam_y) for point in smooth_points]

    # Road surface + rounded joins
    pygame.draw.lines(surface, TRACK_COLOR, True, screen_points, int(TRACK_WIDTH * ZOOM))
    for point in screen_points:
        pygame.draw.circle(surface, TRACK_COLOR, point, int(TRACK_RADIUS * ZOOM))

    # Dashed centerline, purely cosmetic
    for i in range(0, len(screen_points), 4):
        pygame.draw.circle(surface, LANE_WHITE, screen_points[i], max(2, int(3 * ZOOM)))

    # Next waypoint marker
    pygame.draw.circle(surface, (0, 0, 255),
                       world_to_screen(points[waypoint_index], cam_x, cam_y), int(10 * ZOOM))

    draw_start_line(surface, cam_x, cam_y)

def draw_start_line(surface, cam_x, cam_y, squares=6):
    # Checkered start/finish flag instead of a single yellow bar
    cp_x = math.cos(checkpoint_angle) * TRACK_RADIUS
    cp_y = math.sin(checkpoint_angle) * TRACK_RADIUS
    seg_a = (start_point[0] - cp_x, start_point[1] - cp_y)
    seg_b = (start_point[0] + cp_x, start_point[1] + cp_y)

    for i in range(squares):
        t0, t1 = i / squares, (i + 1) / squares
        p0 = (seg_a[0] + (seg_b[0] - seg_a[0]) * t0, seg_a[1] + (seg_b[1] - seg_a[1]) * t0)
        p1 = (seg_a[0] + (seg_b[0] - seg_a[0]) * t1, seg_a[1] + (seg_b[1] - seg_a[1]) * t1)
        color = WHITE if i % 2 == 0 else (30, 30, 30)
        pygame.draw.line(surface, color, world_to_screen(p0, cam_x, cam_y),
                         world_to_screen(p1, cam_x, cam_y), max(3, int(6 * ZOOM)))

def draw_hud_panel(surface, x, y, w, h, alpha=150):
    # A translucent rounded box to put HUD text on, so it stays readable over any track colour
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, alpha), panel.get_rect(), border_radius=12)
    surface.blit(panel, (x, y))

def render_text_outline(font_obj, text, color, outline_color=(0, 0, 0), px=3):
    # Crude outlined text (render the text N times offset in black, then once on top)
    base = font_obj.render(text, True, color)
    outline = font_obj.render(text, True, outline_color)
    surf = pygame.Surface((base.get_width() + px * 2, base.get_height() + px * 2), pygame.SRCALPHA)
    for dx in (-px, 0, px):
        for dy in (-px, 0, px):
            if dx or dy:
                surf.blit(outline, (dx + px, dy + px))
    surf.blit(base, (px, px))
    return surf

def draw():
    draw_track(screen, cam_x, cam_y, next_waypoint_index)

def ai_values(near_index, targets, car_obj=None):
    if car_obj is None:
        car_obj = car  # default keeps training / single-player code unchanged

    readings = get_raycast(car_obj, smooth_points, near_index)

    x_var = car_obj.x
    y_var = car_obj.y
    angle = math.atan2(targets[1] - y_var, targets[0] - x_var) - car_obj.angle
    angle = (angle + math.pi) % (2 * math.pi) - math.pi  # Wrapping it to range of -pi to pi
    angle = angle / math.pi
    angle = round(angle, 2)

    speed = round(round(car_obj.speed) / (car_obj.max_speed + 3), 3)

    dist = round(math.hypot(targets[0] - car_obj.x, targets[1] - car_obj.y) / 250, 3)
    dist = max(0, min(dist, 1))

    readings.extend([angle, speed, dist])
    return readings

def reset_episode():
    global car, can_move, crashed, episode_over, next_waypoint_index
    global last_waypoint_index, laps_completed, net_speed, frame_count
    global frame_since_waypoint, episode_fitness

    car = Car(start_point[0], start_point[1])
    car.angle = start_angle
    can_move = True
    crashed = False
    episode_over = False
    next_waypoint_index = 1
    last_waypoint_index = 0
    laps_completed = 0
    net_speed = 0
    frame_count = 0
    frame_since_waypoint = 0
    episode_fitness = None

def translator(neat_output):
    steer_choice = 0
    for i in range(1, 3):
        if neat_output[i] > neat_output[steer_choice]:
            steer_choice = i

    throttle_choice = 3
    for i in range(4, 6):
        if neat_output[i] > neat_output[throttle_choice]:
            throttle_choice = i

    move_state = {
        "left":  steer_choice == 0,
        "right": steer_choice == 2,
        "accel": throttle_choice == 3,
        "brake": throttle_choice == 5,
    }
    return move_state

def eval_one_genome(genome_id, genome, config):

    global car_near_index, net_speed, frame_count, frame_since_waypoint
    global crashed, can_move, last_waypoint_index, next_waypoint_index
    global laps_completed, cam_x, cam_y

    net = FeedForwardNetwork.create(genome, config)

    reset_episode()                                   # fresh car
    dt = 1 / 60 #Max processing speed
    frame = 0
    if DEBUG:
        print(f"= car #{genome_id} start =")

    while True:
        car_near_index = find_nearest_index((car.x, car.y), smooth_points)
        target = points[next_waypoint_index]
        obs = ai_values(car_near_index, target)
        outputs = net.activate(obs)
        move_state = translator(outputs)

        car.move(move_state, dt)
        net_speed += car.speed
        frame_count += 1
        frame_since_waypoint += 1

        if not car_is_on_track(car, smooth_points, car_near_index):
            crashed = True
            can_move = False
            car.vx = 0
            car.vy = 0

        if math.hypot(car.x - target[0], car.y - target[1]) < WAYPOINT_RADIUS:
            last_waypoint_index = next_waypoint_index
            next_waypoint_index = (next_waypoint_index + 1) % len(points)
            frame_since_waypoint = 0
            if next_waypoint_index == 0:
                laps_completed += 1
        stalled = frame_since_waypoint >= STALL

        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            cam_x, cam_y = get_camera_offset(car.x, car.y)
            draw()
            car.draw(screen, cam_x, cam_y)
            pygame.display.flip()
            pygame.event.pump()
            clock.tick(60)

        if DEBUG and (frame % DEBUG_EVERY == 0):
            print(f"  #{genome_id} f={frame:4d} x={car.x:8.1f} y={car.y:8.1f} "
                  f"spd={car.speed:6.1f} wp={next_waypoint_index:2d} lap={laps_completed} "
                  f"dist2wp={math.hypot(target[0]-car.x, target[1]-car.y):7.1f}")

        frame += 1
        if crashed or stalled or frame > MAX_FRAMES:
            break

    progress = laps_completed * len(points) + next_waypoint_index
    avg_speed = (net_speed / frame_count) / car.max_speed if frame_count > 0 else 0
    if DEBUG:
        print(f"  #{genome_id} DONE crashed={crashed} stalled={stalled} "
              f"fitness={progress + avg_speed * 0.5:.2f}")
    return progress + avg_speed * 0.5

def eval_genomes(genomes, config):
    global points, smooth_points, start_point, next_point, start_angle
    points = generate_points(
        center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2),
        radius_x=850,
        radius_y=380,
        number=24,
        variability=35
    )
    smooth_points = smooth_lines(points, samples=20)
    start_point = points[0]
    next_point = points[1]
    start_angle = math.atan2(next_point[1] - start_point[1],
                             next_point[0] - start_point[0])

    for genome_id, genome in genomes:
        genome.fitness = eval_one_genome(genome_id, genome, config)

class BestSaver(neat.reporting.BaseReporter):
    """Saves the all-time best genome (plus the exact track it trained on) to
    disk after every generation, so the champion survives crashes and reruns."""
    def __init__(self, path="best_genome.pkl"):
        self.path = path
        self.best_fitness = -1e9

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome.fitness > self.best_fitness:   # all-time best, not just this gen
            self.best_fitness = best_genome.fitness
            with open(self.path, "wb") as f:
                pickle.dump({"genome": best_genome, "track": points}, f)
            print(f"  [Saved new champion -> {self.path}  fitness={best_genome.fitness:.2f}]")

def load_config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       "config-feedforward.txt")

def train():
    config = load_config()
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(generation_interval=2, filename_prefix="neat-checkpoint-"))
    pop.add_reporter(BestSaver("best_genome.pkl"))
    pop.run(eval_genomes, 20)

def replay_champion(path="best_genome.pkl"):
    global RENDER, DEBUG, points, smooth_points, start_point, next_point, start_angle

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        genome = data["genome"]
        saved_track = data.get("track")
    else:
        genome = data
        saved_track = None

    if saved_track is not None:
        points = saved_track
        smooth_points = smooth_lines(points, samples=20)
        start_point = points[0]
        next_point = points[1]
        start_angle = math.atan2(next_point[1] - start_point[1],
                                 next_point[0] - start_point[0])

    RENDER = True
    DEBUG = False

    cfg = load_config()
    net = FeedForwardNetwork.create(genome, cfg)

    def apply_replay_view():
        # Track fills the left 2/3 of the window, NN panel the right 1/3.
        # The camera's visible world size comes from the track area only, so
        # world units map 1:1 to on-screen pixels (no stretching).
        global view_width, view_height
        panel_w = screen_width // 3
        view_width = int((screen_width - panel_w) / ZOOM)
        view_height = int(screen_height / ZOOM)
        return panel_w

    def build_viz(panel_w):
        # Rebuilt whenever the window size changes (e.g. F11 fullscreen toggle)
        # so the NN panel always matches the current viewport exactly.
        return nn_viz.compute_layout(genome, cfg, panel_w=panel_w, panel_h=screen_height,
                                     input_labels=INPUT_LABELS, output_labels=OUTPUT_LABELS)

    panel_w = apply_replay_view()
    viz = build_viz(panel_w)

    print(f"Replaying {path} | fitness={genome.fitness:.2f} | size={genome.size()} | ESC/close to exit")
    print("Controls: WASD drive | R reset to checkpoint | F11 fullscreen | ESC quit")

    player_laps = 0
    champ_laps = 0
    player_wp = 1
    champ_wp = 1
    player_last_wp = 0     # last waypoint the player actually reached, for the R-key reset
    player_crashed = False
    champ_crashed = False

    player_car = Car(start_point[0], start_point[1], color=PLAYER_BLUE)
    player_car.angle = start_angle

    # offset the champ car sideways relative to the track direction, not
    # just along world-x (old code: start_point[0] + 30), so it reliably
    # spawns next to the player regardless of track orientation
    side_x = -math.sin(start_angle) * 25
    side_y = math.cos(start_angle) * 25
    champ_car = Car(start_point[0] + side_x, start_point[1] + side_y, color=RED)
    champ_car.angle = start_angle

    dt = 1 / 60
    clock = pygame.time.Clock()
    big_font = pygame.font.SysFont(None, 160)
    lap_font = pygame.font.SysFont(None, 30)
    hint_font = pygame.font.SysFont(None, 22)

    COUNTDOWN_SECS = 3
    countdown_start = pygame.time.get_ticks()
    racing = False
    running = True
    champ_vals = {}  # neuron activations for the NN panel; filled once racing starts

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F11:
                    set_display_mode(not FULLSCREEN)
                    panel_w = apply_replay_view()
                    viz = build_viz(panel_w)
                elif event.key == pygame.K_r:
                    # Reset ONLY the player's car back to the last waypoint they reached.
                    # The champ car / champ progress / lap counts are untouched.
                    wp_point = points[player_last_wp]
                    next_pt = points[(player_last_wp + 1) % len(points)]
                    player_car.x, player_car.y = wp_point
                    player_car.angle = math.atan2(next_pt[1] - wp_point[1], next_pt[0] - wp_point[0])
                    player_car.vx = player_car.vy = 0
                    player_crashed = False

        elapsed = (pygame.time.get_ticks() - countdown_start) / 1000.0
        if not racing and elapsed >= COUNTDOWN_SECS:
            racing = True

        if racing:
            if not champ_crashed:
                champ_near = find_nearest_index((champ_car.x, champ_car.y), smooth_points)
                champ_target = points[champ_wp]
                champ_obs = ai_values(champ_near, champ_target, champ_car)
                champ_vals = nn_viz.get_activations(net, champ_obs)
                champ_out = [champ_vals[o] for o in net.output_nodes]

                champ_move = translator(champ_out)
                champ_car.move(champ_move, dt)

                if not car_is_on_track(champ_car, smooth_points, champ_near):
                    champ_crashed = True
                    champ_car.vx = 0
                    champ_car.vy = 0
                elif math.hypot(champ_car.x - champ_target[0], champ_car.y - champ_target[1]) < WAYPOINT_RADIUS:
                    champ_wp = (champ_wp + 1) % len(points)
                    if champ_wp == 0:
                        champ_laps += 1

            if not player_crashed:
                keys = pygame.key.get_pressed()
                player_move = {
                    "left":  keys[pygame.K_a],
                    "right": keys[pygame.K_d],
                    "accel": keys[pygame.K_w],
                    "brake": keys[pygame.K_s],
                }
                player_near = find_nearest_index((player_car.x, player_car.y), smooth_points)
                player_target = points[player_wp]
                player_car.move(player_move, dt)

                if not car_is_on_track(player_car, smooth_points, player_near):
                    player_crashed = True
                    player_car.vx = 0
                    player_car.vy = 0
                elif math.hypot(player_car.x - player_target[0], player_car.y - player_target[1]) < WAYPOINT_RADIUS:
                    player_last_wp = player_wp
                    player_wp = (player_wp + 1) % len(points)
                    if player_wp == 0:
                        player_laps += 1

        cam_x, cam_y = get_camera_offset(player_car.x, player_car.y)

        world_surf = pygame.Surface((screen_width, screen_height))

        draw_track(world_surf, cam_x, cam_y, player_wp)
        player_car.draw(world_surf, cam_x, cam_y)
        champ_car.draw(world_surf, cam_x, cam_y)

        # HUD: translucent rounded box + colour-matched labels (blue = player, red = champ)
        draw_hud_panel(world_surf, 10, 10, 240, 120)
        world_surf.blit(lap_font.render(
            f"Player: {player_laps}/3" + (" (crashed)" if player_crashed else ""), True, PLAYER_BLUE), (22, 18))
        world_surf.blit(lap_font.render(f"Speed: {player_car.speed:4.0f}", True, PLAYER_BLUE), (22, 48))
        world_surf.blit(lap_font.render(
            f"Champ: {champ_laps}/3" + (" (crashed)" if champ_crashed else ""), True, RED), (22, 78))
        world_surf.blit(lap_font.render(f"Speed: {champ_car.speed:4.0f}", True, RED), (22, 108))

        hint = hint_font.render("WASD drive   R reset   F11 fullscreen   ESC quit", True, (255, 255, 255))
        world_surf.blit(hint, (12, screen_height - 26))

        if not racing:
            remaining = max(0, COUNTDOWN_SECS - elapsed)
            label = "GO!" if remaining <= 0.001 else str(int(remaining) + 1)
            cd_surf = render_text_outline(big_font, label, (255, 215, 0))
            cd_rect = cd_surf.get_rect(center=(screen_width // 2, screen_height // 2))
            world_surf.blit(cd_surf, cd_rect)

        nn_viz.draw_network(world_surf, viz, champ_vals, pos=(screen_width - panel_w, 0))

        screen.blit(world_surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

        # End conditions
        if racing and (player_laps >= 3 or champ_laps >= 3 or (player_crashed and champ_crashed)):
            running = False

    print(f"Race over — Player laps: {player_laps}, Champ laps: {champ_laps}")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NEAT car racing")
    ap.add_argument("--replay", nargs="?", const="best_genome.pkl", metavar="FILE",
                    help="replay a saved champion with rendering (default: best_genome.pkl)")
    ap.add_argument("--debug", action="store_true",
                    help="print per-car telemetry during training (slow, noisy)")
    ap.add_argument("--fullscreen", action="store_true",
                    help="start in fullscreen mode (toggle anytime with F11)")
    args = ap.parse_args()

    DEBUG = args.debug
    RENDER = False

    if args.fullscreen:
        set_display_mode(True)

    if args.replay:
        replay_champion(args.replay)   # renders the champion, then exits

    train()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_F11:
                    set_display_mode(not FULLSCREEN)
                if event.key == pygame.K_r:
                    episode_over = False
                    can_move = True
                    crashed = False
                    net_speed = 0
                    frame_count = 0
                    frame_since_waypoint = 0
                    episode_fitness = None
                    waypoint = points[last_waypoint_index]
                    towards = points[(last_waypoint_index+1) % len(points)]

                    car.x, car.y = waypoint
                    car.angle = math.atan2(towards[1] - waypoint[1], towards[0] - waypoint[0])
                    car.vx = car.vy = 0

                    next_waypoint_index = (last_waypoint_index + 1) % len(points)
                if event.key == pygame.K_RETURN:
                    episode_over = False
                    can_move = True
                    crashed = False
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
                    reset_episode()

        dt = clock.tick(60) / 1000.0   # only once
        keys = pygame.key.get_pressed()
        move_state = {
            "left":  keys[pygame.K_a],
            "right": keys[pygame.K_d],
            "accel": keys[pygame.K_w],
            "brake": keys[pygame.K_s],
        }

        car.move(move_state, dt)              # update first

        net_speed += car.speed
        frame_count += 1
        frame_since_waypoint += 1

        # NEW: find the car's nearest centerline index once per frame, reuse everywhere below
        car_near_index = find_nearest_index((car.x, car.y), smooth_points)

        # simple check
        if not car_is_on_track(car, smooth_points, car_near_index):
            crashed = True
            can_move = False
            car.vx = 0
            car.vy = 0

        #For Waypoints
        target = points[next_waypoint_index]
        if math.hypot(car.x - target[0], car.y - target[1]) < WAYPOINT_RADIUS:
            last_waypoint_index = next_waypoint_index
            next_waypoint_index = (next_waypoint_index + 1) % len(points)
            frame_since_waypoint = 0

            if next_waypoint_index == 0:
                laps_completed += 1

        cam_x, cam_y = get_camera_offset(car.x, car.y)
        draw()
        car.draw(screen, cam_x, cam_y)

        #crashed = not car_is_on_track(car, smooth_points, car_near_index)
        stalled = frame_since_waypoint >= STALL
        if crashed or stalled:
            if not episode_over:
                episode_over = True
                progress = laps_completed * len(points) + next_waypoint_index
                avg_speed_normalized = (net_speed / frame_count) / car.max_speed if frame_count > 0 else 0
                episode_fitness = progress + avg_speed_normalized * 0.5   #f
                print(f"Episode ended — crashed={crashed}, stalled={stalled}, fitness={episode_fitness:.3f}")

        progress = laps_completed * len(points) + next_waypoint_index
        draw_hud_panel(screen, 10, 10, 220, 70)
        speed_text = font.render(f"Speed: {car.speed:.1f}", True, WHITE)
        progress_text = font.render(f"Progress: {progress}", True, WHITE)
        screen.blit(speed_text, (22, 18))
        screen.blit(progress_text, (22, 44))
        pygame.display.flip()