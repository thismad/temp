"""
Micro Agar.io Clone - Server
~50 players, lightweight, real-time
"""

import asyncio
import json
import random
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Game constants
MAP_WIDTH = 2000
MAP_HEIGHT = 2000
FOOD_COUNT = 500
FOOD_SIZE = 10
CACTUS_COUNT = 30
CACTUS_SIZE = 15
PLAYER_START_SIZE = 30
TICK_RATE = 20  # 20 ticks/sec = 50ms
SPEED_BASE = 5
SIZE_SPEED_FACTOR = 0.02  # Bigger = slower
EJECT_SIZE = 12  # Size of ejected mass
EJECT_SPEED = 15  # Speed of ejected mass
EJECT_COST = 15  # Minimum size to eject

class Food:
    def __init__(self, x=None, y=None, color=None, size=FOOD_SIZE):
        self.x = x if x is not None else random.randint(FOOD_SIZE, MAP_WIDTH - FOOD_SIZE)
        self.y = y if y is not None else random.randint(FOOD_SIZE, MAP_HEIGHT - FOOD_SIZE)
        self.size = size
        self.color = color or random.choice([
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ])

class Cactus:
    def __init__(self):
        self.x = random.randint(50, MAP_WIDTH - 50)
        self.y = random.randint(50, MAP_HEIGHT - 50)
        self.size = CACTUS_SIZE

class EjectedMass:
    def __init__(self, x, y, vx, vy, color, size=EJECT_SIZE):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifetime = 0  # Ticks since ejected

    def update(self):
        # Move and slow down
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.92
        self.vy *= 0.92
        self.lifetime += 1
        # Clamp to map
        self.x = max(self.size, min(MAP_WIDTH - self.size, self.x))
        self.y = max(self.size, min(MAP_HEIGHT - self.size, self.y))

BOT_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]

class Player:
    def __init__(self, ws: WebSocket | None, player_id: int, name: str, is_bot: bool = False):
        self.ws = ws
        self.id = player_id
        self.name = name[:15] if name else f"Player{player_id}"
        self.x = random.randint(100, MAP_WIDTH - 100)
        self.y = random.randint(100, MAP_HEIGHT - 100)
        self.size = PLAYER_START_SIZE
        self.color = f"hsl({random.randint(0, 360)}, 70%, 50%)"
        self.target_x = self.x
        self.target_y = self.y
        self.score = 0
        self.is_bot = is_bot

    @property
    def radius(self):
        return self.size

    @property
    def speed(self):
        return max(1, SPEED_BASE - self.size * SIZE_SPEED_FACTOR)

    def move_towards_target(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 5:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

        # Clamp to map bounds
        self.x = max(self.radius, min(MAP_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(MAP_HEIGHT - self.radius, self.y))

    def can_eat(self, other_radius: float) -> bool:
        return self.radius > other_radius * 1.1

    def eat(self, other_size: float):
        # Mass is proportional to area
        self.size = math.sqrt(self.size**2 + other_size**2 * 0.5)
        self.score += int(other_size)

class Game:
    def __init__(self, num_bots: int = 5):
        self.players: dict[int, Player] = {}
        self.food: list[Food] = []
        self.cactus: list[Cactus] = []
        self.ejected: list[EjectedMass] = []
        self.next_id = 1
        self.running = False

        # Spawn initial food
        for _ in range(FOOD_COUNT):
            self.food.append(Food())

        # Spawn cactus
        for _ in range(CACTUS_COUNT):
            self.cactus.append(Cactus())

        # Spawn bots
        for i in range(num_bots):
            self.add_bot(BOT_NAMES[i % len(BOT_NAMES)])

    def add_player(self, ws: WebSocket, name: str) -> Player:
        player = Player(ws, self.next_id, name)
        self.players[self.next_id] = player
        self.next_id += 1
        return player

    def add_bot(self, name: str) -> Player:
        bot = Player(None, self.next_id, f"[BOT] {name}", is_bot=True)
        self.players[self.next_id] = bot
        self.next_id += 1
        return bot

    def remove_player(self, player_id: int):
        if player_id in self.players:
            del self.players[player_id]

    def eject_mass(self, player: Player):
        """Player ejects mass in the direction they're moving"""
        if player.size < EJECT_COST + EJECT_SIZE:
            return  # Too small to eject

        # Direction from player to target
        dx = player.target_x - player.x
        dy = player.target_y - player.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return

        # Normalize direction
        dx /= dist
        dy /= dist

        # Create ejected mass
        eject_x = player.x + dx * (player.radius + EJECT_SIZE)
        eject_y = player.y + dy * (player.radius + EJECT_SIZE)
        vx = dx * EJECT_SPEED
        vy = dy * EJECT_SPEED

        self.ejected.append(EjectedMass(eject_x, eject_y, vx, vy, player.color))

        # Reduce player size
        player.size = math.sqrt(player.size**2 - EJECT_SIZE**2)

    def update_bots(self):
        for player in self.players.values():
            if not player.is_bot:
                continue

            # Find nearest food
            nearest_food = None
            nearest_dist = float('inf')
            for f in self.food:
                dx = f.x - player.x
                dy = f.y - player.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_food = f

            # Check for nearby threats (bigger players)
            flee_target = None
            for other in self.players.values():
                if other.id == player.id:
                    continue
                dx = other.x - player.x
                dy = other.y - player.y
                dist = math.sqrt(dx * dx + dy * dy)
                # Flee if other is bigger and close
                if other.can_eat(player.radius) and dist < 200:
                    # Run away
                    flee_target = (player.x - dx, player.y - dy)
                    break

            # Check for prey (smaller players)
            chase_target = None
            for other in self.players.values():
                if other.id == player.id:
                    continue
                dx = other.x - player.x
                dy = other.y - player.y
                dist = math.sqrt(dx * dx + dy * dy)
                if player.can_eat(other.radius) and dist < 300:
                    chase_target = (other.x, other.y)
                    break

            # Priority: flee > chase > eat food
            if flee_target:
                player.target_x, player.target_y = flee_target
            elif chase_target:
                player.target_x, player.target_y = chase_target
            elif nearest_food:
                player.target_x = nearest_food.x
                player.target_y = nearest_food.y

    def update(self):
        # Update bot AI
        self.update_bots()

        # Move all players
        for player in self.players.values():
            player.move_towards_target()

        # Update ejected masses
        for e in self.ejected:
            e.update()

        # Check food collisions
        for player in self.players.values():
            eaten_food = []
            for i, f in enumerate(self.food):
                dx = player.x - f.x
                dy = player.y - f.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < player.radius:
                    eaten_food.append(i)
                    player.eat(f.size)

            # Remove eaten food and spawn new
            for i in reversed(eaten_food):
                self.food[i] = Food()

        # Check ejected mass collisions
        eaten_ejected = []
        for i, e in enumerate(self.ejected):
            if e.lifetime < 10:  # Can't eat own mass immediately
                continue
            for player in self.players.values():
                dx = player.x - e.x
                dy = player.y - e.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < player.radius:
                    player.eat(e.size)
                    eaten_ejected.append(i)
                    break
        for i in reversed(eaten_ejected):
            self.ejected.pop(i)

        # Check cactus collisions (split player)
        for player in self.players.values():
            for cactus in self.cactus:
                dx = player.x - cactus.x
                dy = player.y - cactus.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < player.radius + cactus.size:
                    # Split: lose half mass as ejected pieces
                    if player.size > PLAYER_START_SIZE * 1.5:
                        pieces = min(5, int(player.size / 20))
                        for _ in range(pieces):
                            angle = random.uniform(0, 2 * math.pi)
                            vx = math.cos(angle) * EJECT_SPEED * 1.5
                            vy = math.sin(angle) * EJECT_SPEED * 1.5
                            self.ejected.append(EjectedMass(
                                player.x, player.y, vx, vy, player.color, EJECT_SIZE
                            ))
                            player.size = math.sqrt(max(PLAYER_START_SIZE**2, player.size**2 - EJECT_SIZE**2))

        # Check player vs player collisions
        players_list = list(self.players.values())
        eaten_players = set()

        for i, p1 in enumerate(players_list):
            for p2 in players_list[i+1:]:
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < max(p1.radius, p2.radius):
                    if p1.can_eat(p2.radius):
                        p1.eat(p2.size)
                        eaten_players.add(p2.id)
                    elif p2.can_eat(p1.radius):
                        p2.eat(p1.size)
                        eaten_players.add(p1.id)

        # Respawn eaten players
        for pid in eaten_players:
            if pid in self.players:
                p = self.players[pid]
                p.x = random.randint(100, MAP_WIDTH - 100)
                p.y = random.randint(100, MAP_HEIGHT - 100)
                p.size = PLAYER_START_SIZE

    def get_state(self, for_player_id: int) -> dict:
        player = self.players.get(for_player_id)
        if not player:
            return {}

        # Get visible area (viewport centered on player)
        view_size = 800 + player.size * 2

        # All players (send all for simplicity, client can cull)
        players_data = [
            [p.id, round(p.x), round(p.y), round(p.size), p.name, p.color]
            for p in self.players.values()
        ]

        # Food near player (optimization: only send nearby food)
        food_data = [
            [round(f.x), round(f.y), f.color]
            for f in self.food
            if abs(f.x - player.x) < view_size and abs(f.y - player.y) < view_size
        ]

        # Cactus data
        cactus_data = [
            [round(c.x), round(c.y), c.size]
            for c in self.cactus
            if abs(c.x - player.x) < view_size and abs(c.y - player.y) < view_size
        ]

        # Ejected mass data
        ejected_data = [
            [round(e.x), round(e.y), round(e.size), e.color]
            for e in self.ejected
            if abs(e.x - player.x) < view_size and abs(e.y - player.y) < view_size
        ]

        # Leaderboard (top 10)
        leaderboard = sorted(
            [(p.name, p.score) for p in self.players.values()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "t": "s",  # state
            "p": players_data,
            "f": food_data,
            "c": cactus_data,
            "e": ejected_data,
            "l": leaderboard,
            "you": for_player_id,
            "map": [MAP_WIDTH, MAP_HEIGHT]
        }

    async def broadcast(self):
        disconnected = []
        for pid, player in self.players.items():
            if player.is_bot:
                continue  # Bots don't need state updates
            try:
                state = self.get_state(pid)
                await player.ws.send_text(json.dumps(state))
            except:
                disconnected.append(pid)

        for pid in disconnected:
            self.remove_player(pid)

    async def game_loop(self):
        self.running = True
        while self.running:
            self.update()
            await self.broadcast()
            await asyncio.sleep(1 / TICK_RATE)

# Global game instance
game = Game()

@app.on_event("startup")
async def startup():
    asyncio.create_task(game.game_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str = ""):
    await websocket.accept()
    player = game.add_player(websocket, name)

    # Send initial welcome
    await websocket.send_text(json.dumps({
        "t": "welcome",
        "id": player.id,
        "map": [MAP_WIDTH, MAP_HEIGHT]
    }))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("t") == "m":  # move
                player.target_x = msg.get("x", player.x)
                player.target_y = msg.get("y", player.y)

            elif msg.get("t") == "e":  # eject mass (spacebar)
                game.eject_mass(player)

    except WebSocketDisconnect:
        game.remove_player(player.id)
    except Exception as e:
        print(f"Error: {e}")
        game.remove_player(player.id)

# Serve the game HTML
@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
