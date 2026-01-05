"""
Micro Agar.io Clone - Server
~50 players, lightweight, real-time
"""

import asyncio
import json
import random
import math
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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
TICK_RATE = 20
SPEED_BASE = 5
SIZE_SPEED_FACTOR = 0.02
EJECT_SIZE = 12
EJECT_SPEED = 15
EJECT_COST = 15
SPLIT_MIN_SIZE = 35  # Minimum size to split
MERGE_TIME = 200  # Ticks before cells can merge (~10 seconds)
MAX_CELLS = 16  # Maximum cells per player

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
        self.lifetime = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.92
        self.vy *= 0.92
        self.lifetime += 1
        self.x = max(self.size, min(MAP_WIDTH - self.size, self.x))
        self.y = max(self.size, min(MAP_HEIGHT - self.size, self.y))

class Cell:
    """A single cell belonging to a player"""
    def __init__(self, x, y, size, vx=0, vy=0):
        self.x = x
        self.y = y
        self.size = size
        self.vx = vx  # Velocity for split movement
        self.vy = vy
        self.merge_time = 0  # Ticks until can merge

    @property
    def radius(self):
        return self.size

    @property
    def speed(self):
        return max(1, SPEED_BASE - self.size * SIZE_SPEED_FACTOR)

    def move_towards(self, target_x, target_y):
        # Apply velocity (for split)
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.9
        self.vy *= 0.9

        # Move towards target
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 5:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

        # Clamp to map
        self.x = max(self.radius, min(MAP_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(MAP_HEIGHT - self.radius, self.y))

        # Decrease merge timer
        if self.merge_time > 0:
            self.merge_time -= 1

BOT_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]

class Player:
    def __init__(self, ws: WebSocket | None, player_id: int, name: str, is_bot: bool = False):
        self.ws = ws
        self.id = player_id
        self.name = name[:15] if name else f"Player{player_id}"
        self.color = f"hsl({random.randint(0, 360)}, 70%, 50%)"
        self.target_x = MAP_WIDTH / 2
        self.target_y = MAP_HEIGHT / 2
        self.score = 0
        self.is_bot = is_bot
        self.cells: list[Cell] = [Cell(
            random.randint(100, MAP_WIDTH - 100),
            random.randint(100, MAP_HEIGHT - 100),
            PLAYER_START_SIZE
        )]

    @property
    def total_size(self):
        return sum(c.size for c in self.cells)

    @property
    def center_x(self):
        if not self.cells:
            return MAP_WIDTH / 2
        return sum(c.x * c.size for c in self.cells) / self.total_size

    @property
    def center_y(self):
        if not self.cells:
            return MAP_HEIGHT / 2
        return sum(c.y * c.size for c in self.cells) / self.total_size

    def can_eat(self, other_radius: float, cell: Cell) -> bool:
        return cell.radius > other_radius * 1.1

    def respawn(self):
        self.cells = [Cell(
            random.randint(100, MAP_WIDTH - 100),
            random.randint(100, MAP_HEIGHT - 100),
            PLAYER_START_SIZE
        )]

class Game:
    def __init__(self, num_bots: int = 5):
        self.players: dict[int, Player] = {}
        self.food: list[Food] = []
        self.cactus: list[Cactus] = []
        self.ejected: list[EjectedMass] = []
        self.next_id = 1
        self.running = False

        for _ in range(FOOD_COUNT):
            self.food.append(Food())
        for _ in range(CACTUS_COUNT):
            self.cactus.append(Cactus())
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

    def feed(self, player: Player):
        """W key - eject mass from all cells"""
        for cell in player.cells:
            if cell.size < EJECT_COST + EJECT_SIZE:
                continue

            dx = player.target_x - cell.x
            dy = player.target_y - cell.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1:
                continue

            dx /= dist
            dy /= dist

            eject_x = cell.x + dx * (cell.radius + EJECT_SIZE)
            eject_y = cell.y + dy * (cell.radius + EJECT_SIZE)
            vx = dx * EJECT_SPEED
            vy = dy * EJECT_SPEED

            self.ejected.append(EjectedMass(eject_x, eject_y, vx, vy, player.color))
            cell.size = math.sqrt(cell.size**2 - EJECT_SIZE**2)

    def split(self, player: Player):
        """Space key - split all cells that are big enough"""
        new_cells = []
        for cell in player.cells:
            if len(player.cells) + len(new_cells) >= MAX_CELLS:
                break
            if cell.size < SPLIT_MIN_SIZE:
                continue

            # Direction towards target
            dx = player.target_x - cell.x
            dy = player.target_y - cell.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1:
                dx, dy = 1, 0
            else:
                dx /= dist
                dy /= dist

            # Split size in half
            new_size = cell.size / math.sqrt(2)
            cell.size = new_size
            cell.merge_time = MERGE_TIME

            # New cell shoots forward
            new_cell = Cell(
                cell.x + dx * cell.radius,
                cell.y + dy * cell.radius,
                new_size,
                dx * 20,  # Initial velocity
                dy * 20
            )
            new_cell.merge_time = MERGE_TIME
            new_cells.append(new_cell)

        player.cells.extend(new_cells)

    def update_bots(self):
        for player in self.players.values():
            if not player.is_bot or not player.cells:
                continue

            main_cell = max(player.cells, key=lambda c: c.size)

            # Find nearest food
            nearest_food = None
            nearest_dist = float('inf')
            for f in self.food:
                dx = f.x - main_cell.x
                dy = f.y - main_cell.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_food = f

            # Check threats
            flee_target = None
            for other in self.players.values():
                if other.id == player.id:
                    continue
                for oc in other.cells:
                    dx = oc.x - main_cell.x
                    dy = oc.y - main_cell.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if oc.size > main_cell.size * 1.1 and dist < 200:
                        flee_target = (main_cell.x - dx, main_cell.y - dy)
                        break

            # Check prey
            chase_target = None
            for other in self.players.values():
                if other.id == player.id:
                    continue
                for oc in other.cells:
                    dx = oc.x - main_cell.x
                    dy = oc.y - main_cell.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if main_cell.size > oc.size * 1.1 and dist < 300:
                        chase_target = (oc.x, oc.y)
                        break

            if flee_target:
                player.target_x, player.target_y = flee_target
            elif chase_target:
                player.target_x, player.target_y = chase_target
            elif nearest_food:
                player.target_x = nearest_food.x
                player.target_y = nearest_food.y

    def merge_cells(self, player: Player):
        """Merge cells that overlap and can merge"""
        if len(player.cells) < 2:
            return

        merged = set()
        for i, c1 in enumerate(player.cells):
            if i in merged:
                continue
            for j, c2 in enumerate(player.cells[i+1:], i+1):
                if j in merged:
                    continue
                if c1.merge_time > 0 or c2.merge_time > 0:
                    continue

                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist = math.sqrt(dx * dx + dy * dy)

                # If overlapping enough, merge
                if dist < max(c1.radius, c2.radius) * 0.8:
                    # Merge into larger cell
                    c1.size = math.sqrt(c1.size**2 + c2.size**2)
                    c1.x = (c1.x * c1.size + c2.x * c2.size) / (c1.size + c2.size)
                    c1.y = (c1.y * c1.size + c2.y * c2.size) / (c1.size + c2.size)
                    merged.add(j)

        player.cells = [c for i, c in enumerate(player.cells) if i not in merged]

    def push_apart_cells(self, player: Player):
        """Push cells of same player apart so they don't overlap"""
        for i, c1 in enumerate(player.cells):
            for c2 in player.cells[i+1:]:
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = c1.radius + c2.radius

                if dist < min_dist and dist > 0:
                    # Push apart
                    overlap = (min_dist - dist) / 2
                    dx /= dist
                    dy /= dist
                    c1.x += dx * overlap * 0.5
                    c1.y += dy * overlap * 0.5
                    c2.x -= dx * overlap * 0.5
                    c2.y -= dy * overlap * 0.5

    def update(self):
        self.update_bots()

        # Move all cells
        for player in self.players.values():
            for cell in player.cells:
                cell.move_towards(player.target_x, player.target_y)
            self.push_apart_cells(player)
            self.merge_cells(player)

        # Update ejected masses
        for e in self.ejected:
            e.update()

        # Check food collisions
        for player in self.players.values():
            for cell in player.cells:
                eaten_food = []
                for i, f in enumerate(self.food):
                    dx = cell.x - f.x
                    dy = cell.y - f.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < cell.radius:
                        eaten_food.append(i)
                        cell.size = math.sqrt(cell.size**2 + f.size**2 * 0.5)
                        player.score += int(f.size)

                for i in reversed(eaten_food):
                    self.food[i] = Food()

        # Check ejected mass collisions
        eaten_ejected = []
        for i, e in enumerate(self.ejected):
            if e.lifetime < 10:
                continue
            for player in self.players.values():
                for cell in player.cells:
                    dx = cell.x - e.x
                    dy = cell.y - e.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < cell.radius:
                        cell.size = math.sqrt(cell.size**2 + e.size**2 * 0.5)
                        player.score += int(e.size)
                        eaten_ejected.append(i)
                        break
                if i in eaten_ejected:
                    break
        for i in reversed(eaten_ejected):
            self.ejected.pop(i)

        # Check cactus collisions
        for player in self.players.values():
            for cell in player.cells:
                for cactus in self.cactus:
                    dx = cell.x - cactus.x
                    dy = cell.y - cactus.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < cell.radius + cactus.size:
                        if cell.size > PLAYER_START_SIZE * 1.5:
                            pieces = min(5, int(cell.size / 20))
                            for _ in range(pieces):
                                angle = random.uniform(0, 2 * math.pi)
                                vx = math.cos(angle) * EJECT_SPEED * 1.5
                                vy = math.sin(angle) * EJECT_SPEED * 1.5
                                self.ejected.append(EjectedMass(
                                    cell.x, cell.y, vx, vy, player.color, EJECT_SIZE
                                ))
                                cell.size = math.sqrt(max(PLAYER_START_SIZE**2, cell.size**2 - EJECT_SIZE**2))

        # Check player vs player collisions
        players_list = list(self.players.values())
        for p1 in players_list:
            for p2 in players_list:
                if p1.id >= p2.id:
                    continue

                cells_to_remove_p1 = []
                cells_to_remove_p2 = []

                for i, c1 in enumerate(p1.cells):
                    for j, c2 in enumerate(p2.cells):
                        dx = c1.x - c2.x
                        dy = c1.y - c2.y
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist < max(c1.radius, c2.radius):
                            if c1.size > c2.size * 1.1:
                                c1.size = math.sqrt(c1.size**2 + c2.size**2 * 0.5)
                                p1.score += int(c2.size)
                                cells_to_remove_p2.append(j)
                            elif c2.size > c1.size * 1.1:
                                c2.size = math.sqrt(c2.size**2 + c1.size**2 * 0.5)
                                p2.score += int(c1.size)
                                cells_to_remove_p1.append(i)

                # Remove eaten cells
                p1.cells = [c for i, c in enumerate(p1.cells) if i not in cells_to_remove_p1]
                p2.cells = [c for i, c in enumerate(p2.cells) if i not in cells_to_remove_p2]

        # Respawn players with no cells
        for player in self.players.values():
            if not player.cells:
                player.respawn()

    def get_state(self, for_player_id: int) -> dict:
        player = self.players.get(for_player_id)
        if not player:
            return {}

        view_size = 800 + player.total_size * 2

        # All cells from all players
        cells_data = []
        for p in self.players.values():
            for cell in p.cells:
                cells_data.append([
                    p.id, round(cell.x), round(cell.y), round(cell.size), p.name, p.color
                ])

        food_data = [
            [round(f.x), round(f.y), f.color]
            for f in self.food
            if abs(f.x - player.center_x) < view_size and abs(f.y - player.center_y) < view_size
        ]

        cactus_data = [
            [round(c.x), round(c.y), c.size]
            for c in self.cactus
            if abs(c.x - player.center_x) < view_size and abs(c.y - player.center_y) < view_size
        ]

        ejected_data = [
            [round(e.x), round(e.y), round(e.size), e.color]
            for e in self.ejected
            if abs(e.x - player.center_x) < view_size and abs(e.y - player.center_y) < view_size
        ]

        leaderboard = sorted(
            [(p.name, p.score) for p in self.players.values()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "t": "s",
            "p": cells_data,
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
                continue
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

# NUM_BOTS=0 in production (Docker), NUM_BOTS=5 in dev
num_bots = int(os.getenv("NUM_BOTS", "5"))
game = Game(num_bots=num_bots)

@app.on_event("startup")
async def startup():
    asyncio.create_task(game.game_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str = ""):
    await websocket.accept()
    player = game.add_player(websocket, name)

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
                player.target_x = msg.get("x", player.target_x)
                player.target_y = msg.get("y", player.target_y)

            elif msg.get("t") == "w":  # W key - feed/eject
                game.feed(player)

            elif msg.get("t") == "s":  # Space - split
                game.split(player)

    except WebSocketDisconnect:
        game.remove_player(player.id)
    except Exception as e:
        print(f"Error: {e}")
        game.remove_player(player.id)

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
