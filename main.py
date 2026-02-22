import pygame
import random
import numpy as np
import sys
import warnings
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

pygame.init()

# --- Display Settings ---
WIDTH = 800
HEIGHT = 600
FPS = 60

# --- Colors (Polished Palette) ---
WHITE = (245, 245, 250)
BLACK = (15, 15, 20)
RED = (255, 65, 85)
DARK_RED = (150, 20, 30)
BLUE = (65, 200, 255)
DARK_BLUE = (20, 80, 150)
LIGHT_GRAY = (150, 150, 160)
GREEN = (50, 255, 120)
BG_COLOR = (20, 20, 28)
GRID_COLOR = (35, 35, 45)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Dodge Arena: Hardcore Edition")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Segoe UI", 28, bold=True)
small_font = pygame.font.SysFont("Segoe UI", 16)
large_font = pygame.font.SysFont("Segoe UI", 72, bold=True)

class Trail:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(int(x), int(y), width, height)
        self.color = color
        self.alpha = 150

    def update(self, dt):
        self.alpha -= 600 * dt # Fades out quickly

    def draw(self, surface):
        if self.alpha > 0:
            s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            r, g, b = self.color
            pygame.draw.rect(s, (r, g, b, int(self.alpha)), s.get_rect(), border_radius=10)
            surface.blit(s, (self.rect.x, self.rect.y))

class Player:
    def __init__(self):
        self.width = 44
        self.height = 44
        self.x = float(WIDTH // 2 - self.width // 2)
        self.y = float(HEIGHT - self.height - 30)
        self.speed = 550.0 
        self.rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)
        self.trails = []

    def move(self, keys, dt):
        old_x = self.x
        if keys[pygame.K_LEFT] and self.x > 0:
            self.x -= self.speed * dt
        if keys[pygame.K_RIGHT] and self.x < WIDTH - self.width:
            self.x += self.speed * dt
        
        self.rect.x = int(self.x)

        # Add trail if moving
        if abs(self.x - old_x) > 0.1:
            self.trails.append(Trail(self.x, self.y, self.width, self.height, BLUE))

        # Update trails
        for t in self.trails[:]:
            t.update(dt)
            if t.alpha <= 0:
                self.trails.remove(t)

    def draw(self, surface):
        for t in self.trails:
            t.draw(surface)

        # Glow/Shadow
        shadow_rect = self.rect.copy()
        shadow_rect.inflate_ip(8, 8)
        pygame.draw.rect(surface, DARK_BLUE, shadow_rect, border_radius=12)
        
        # Main body
        pygame.draw.rect(surface, BLUE, self.rect, border_radius=10)
        pygame.draw.rect(surface, WHITE, self.rect, width=2, border_radius=10)

class Block:
    def __init__(self, x, y, base_speed):
        self.width = random.randint(35, 75)
        self.height = random.randint(35, 75)
        self.x = float(x)
        self.y = float(y)
        # Difficulty tweak: Variable speeds so they overlap and create complex patterns
        self.speed = base_speed * random.uniform(0.8, 1.4) 
        self.rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    def update(self, dt):
        self.y += self.speed * dt
        self.rect.y = int(self.y)

    def draw(self, surface):
        # Glow
        glow_rect = self.rect.copy()
        glow_rect.inflate_ip(10, 10)
        pygame.draw.rect(surface, DARK_RED, glow_rect, border_radius=10)
        
        # Main body
        pygame.draw.rect(surface, RED, self.rect, border_radius=8)
        pygame.draw.rect(surface, WHITE, self.rect, width=2, border_radius=8)

def draw_background(surface, scroll_y):
    surface.fill(BG_COLOR)
    grid_size = 50
    offset_y = scroll_y % grid_size
    
    for x in range(0, WIDTH, grid_size):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(int(offset_y) - grid_size, HEIGHT, grid_size):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WIDTH, y))

def main():
    player = Player()
    blocks = []
    
    score = 0
    frame_count = 0
    bg_scroll = 0.0
    
    # Difficulty Mechanics
    base_speed = 400.0 # Increased starting speed
    spawn_timer = 0.0
    spawn_interval = 0.45 # Much faster spawns
    
    # ML Data
    player_history_x = []
    MAX_HISTORY = 1000
    
    kmeans = KMeans(n_clusters=3, n_init=10)
    nb = GaussianNB()
    
    models_trained = False
    preferred_zone = -1
    current_prediction = -1
    
    running = True
    game_over = False

    while running:
        dt = clock.tick(FPS) / 1000.0 
        
        # Background Scrolling
        bg_scroll += 100 * dt
        draw_background(screen, bg_scroll)
        
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_over:
                    main()
                    return

        if not game_over:
            player.move(keys, dt)
            
            # Record Data
            if frame_count % 5 == 0:
                player_history_x.append([player.x + player.width / 2])
                if len(player_history_x) > MAX_HISTORY:
                    player_history_x.pop(0)
                
            # Train ML (Faster initial training - 30 instead of 60)
            if frame_count > 0 and frame_count % 120 == 0 and len(player_history_x) > 30:
                try:
                    X_data = np.array(player_history_x[-300:])
                    kmeans.fit(X_data)
                    zones = kmeans.predict(X_data)
                    
                    unique_zones, counts = np.unique(zones, return_counts=True)
                    preferred_zone = unique_zones[np.argmax(counts)]
                    
                    if len(zones) >= 3:
                        X_nb, y_nb = [], []
                        for i in range(len(zones)-2):
                            X_nb.append([zones[i], zones[i+1]])
                            y_nb.append(zones[i+2])
                            
                        if len(set(y_nb)) > 1:
                            nb.fit(X_nb, y_nb)
                            models_trained = True
                except Exception:
                    pass 
            
            # Spawn Blocks
            spawn_timer += dt
            if spawn_timer >= spawn_interval:
                spawn_timer = 0.0
                target_x = random.randint(0, WIDTH - 75)
                
                if models_trained and len(player_history_x) >= 2:
                    try:
                        recent_x = np.array(player_history_x[-2:])
                        recent_zones = kmeans.predict(recent_x)
                        predicted_zone = nb.predict([recent_zones])[0]
                        current_prediction = predicted_zone
                        
                        cluster_center = kmeans.cluster_centers_[predicted_zone][0]
                        target_x = int(cluster_center) + random.randint(-40, 40)
                        target_x = max(0, min(WIDTH - 75, target_x))
                    except Exception:
                        pass 
                
                blocks.append(Block(target_x, -100, base_speed))

            # Update Blocks
            for block in blocks[:]:
                block.update(dt)
                
                # Shrink hitboxes slightly for fair "grazing" dodges
                player_hitbox = player.rect.inflate(-8, -8)
                block_hitbox = block.rect.inflate(-8, -8)
                
                if block_hitbox.colliderect(player_hitbox):
                    game_over = True
                    
                if block.y > HEIGHT:
                    blocks.remove(block)
                    score += 1
                    
                    # Difficulty Curve (Aggressive scaling)
                    if score % 5 == 0:
                        base_speed += 20.0
                        if spawn_interval > 0.15:
                            spawn_interval -= 0.02

            # Draw
            player.draw(screen)
            for block in blocks:
                block.draw(screen)

            # UI Polish
            score_text = font.render(f"SCORE: {score}", True, WHITE)
            screen.blit(score_text, (WIDTH - score_text.get_width() - 20, 20))
            
            # Glassmorphism style ML Panel
            panel_surface = pygame.Surface((340, 95), pygame.SRCALPHA)
            pygame.draw.rect(panel_surface, (20, 20, 30, 200), panel_surface.get_rect(), border_radius=8)
            pygame.draw.rect(panel_surface, (100, 100, 120, 255), panel_surface.get_rect(), width=1, border_radius=8)
            screen.blit(panel_surface, (15, 15))
            
            status_color = GREEN if models_trained else LIGHT_GRAY
            status_str = "SYSTEM ACTIVE" if models_trained else "ANALYZING TARGET..."
            ml_status = font.render(f"{status_str}", True, status_color)
            screen.blit(ml_status, (30, 25))
            
            if models_trained:
                pref_text = small_font.render(f"Favored Zone: [{preferred_zone}]", True, LIGHT_GRAY)
                pred_text = small_font.render(f"Intercepting Zone: [{current_prediction}]", True, RED)
                screen.blit(pref_text, (30, 60))
                screen.blit(pred_text, (30, 80))

            frame_count += 1
            
        else:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (10, 10, 15, 220), overlay.get_rect())
            screen.blit(overlay, (0, 0))
            
            go_text = large_font.render("SYSTEM FAILURE", True, RED)
            score_res = font.render(f"Final Score: {score}", True, WHITE)
            restart_text = small_font.render("[ Press SPACE to Reboot ]", True, BLUE)
            
            screen.blit(go_text, (WIDTH // 2 - go_text.get_width() // 2, HEIGHT // 2 - 100))
            screen.blit(score_res, (WIDTH // 2 - score_res.get_width() // 2, HEIGHT // 2))
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 70))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()