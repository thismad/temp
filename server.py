"""
Micro Agar.io Clone - Server
~50 players, lightweight, real-time
OPTIMIZED: spatial hashing, dist², cached leaderboard, parallel broadcast, orjson
"""

import asyncio
import random
import math
import os
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Fast JSON (orjson is ~3-10x faster)
try:
    import orjson
    def json_dumps(obj): return orjson.dumps(obj).decode()
    def json_loads(s): return orjson.loads(s)
except ImportError:
    import json
    json_dumps = json.dumps
    json_loads = json.loads

app = FastAPI()

# Game constants
BASE_MAP_WIDTH = 2000
BASE_MAP_HEIGHT = 2000
FOOD_COUNT = 150
FOOD_SIZE = 18
CACTUS_COUNT = 10
MAX_PLAYERS = 50
PLAYERS_PER_SCALE = 10
CACTUS_SIZE = 15
PLAYER_START_SIZE = 30
TICK_RATE = 20
SPEED_BASE = 5
SIZE_SPEED_FACTOR = 0.02
EJECT_SIZE = 12
EJECT_SPEED = 15
EJECT_COST = 15
SPLIT_MIN_SIZE = 35
MERGE_TIME = 200
MAX_CELLS = 16
GRID_CELL_SIZE = 100


class SpatialGrid:
    """Spatial hashing for O(1) neighbor lookups instead of O(n²)"""
    __slots__ = ('cell_size', 'grid')

    def __init__(self, cell_size: int = GRID_CELL_SIZE):
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list] = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def insert(self, obj, x: float, y: float):
        key = (int(x // self.cell_size), int(y // self.cell_size))
        self.grid[key].append(obj)

    def query(self, x: float, y: float, radius: float) -> list:
        results = []
        cells_needed = int(radius // self.cell_size) + 1
        cx, cy = int(x // self.cell_size), int(y // self.cell_size)
        for dx in range(-cells_needed, cells_needed + 1):
            for dy in range(-cells_needed, cells_needed + 1):
                bucket = self.grid.get((cx + dx, cy + dy))
                if bucket:
                    results.extend(bucket)
        return results


class Food:
    __slots__ = ('x', 'y', 'size', 'color')

    def __init__(self, x=None, y=None, color=None, size=FOOD_SIZE, map_w=BASE_MAP_WIDTH, map_h=BASE_MAP_HEIGHT):
        self.x = x if x is not None else random.randint(FOOD_SIZE, map_w - FOOD_SIZE)
        self.y = y if y is not None else random.randint(FOOD_SIZE, map_h - FOOD_SIZE)
        self.size = size
        self.color = color or random.choice([
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ])


class Cactus:
    __slots__ = ('x', 'y', 'size')

    def __init__(self, map_w=BASE_MAP_WIDTH, map_h=BASE_MAP_HEIGHT):
        self.x = random.randint(50, map_w - 50)
        self.y = random.randint(50, map_h - 50)
        self.size = CACTUS_SIZE


class EjectedMass:
    __slots__ = ('x', 'y', 'vx', 'vy', 'color', 'size', 'lifetime', 'map_w', 'map_h')

    def __init__(self, x, y, vx, vy, color, size=EJECT_SIZE, map_w=BASE_MAP_WIDTH, map_h=BASE_MAP_HEIGHT):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifetime = 0
        self.map_w = map_w
        self.map_h = map_h

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.92
        self.vy *= 0.92
        self.lifetime += 1
        self.x = max(self.size, min(self.map_w - self.size, self.x))
        self.y = max(self.size, min(self.map_h - self.size, self.y))


class Cell:
    __slots__ = ('x', 'y', 'size', 'vx', 'vy', 'merge_time', 'map_w', 'map_h')

    def __init__(self, x, y, size, vx=0, vy=0, map_w=BASE_MAP_WIDTH, map_h=BASE_MAP_HEIGHT):
        self.x = x
        self.y = y
        self.size = size
        self.vx = vx
        self.vy = vy
        self.merge_time = 0
        self.map_w = map_w
        self.map_h = map_h

    @property
    def radius(self):
        return self.size

    @property
    def speed(self):
        return max(1, SPEED_BASE - self.size * SIZE_SPEED_FACTOR)

    def move_towards(self, target_x, target_y):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.9
        self.vy *= 0.9

        dx = target_x - self.x
        dy = target_y - self.y
        dist_sq = dx * dx + dy * dy

        if dist_sq > 25:
            dist = math.sqrt(dist_sq)
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

        self.x = max(self.radius, min(self.map_w - self.radius, self.x))
        self.y = max(self.radius, min(self.map_h - self.radius, self.y))

        if self.merge_time > 0:
            self.merge_time -= 1


BOT_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]


class Player:
    __slots__ = ('ws', 'id', 'name', 'color', 'target_x', 'target_y', 'score', 'is_bot', 'map_w', 'map_h', 'cells')

    def __init__(self, ws, player_id: int, name: str, is_bot: bool = False, map_w: int = BASE_MAP_WIDTH, map_h: int = BASE_MAP_HEIGHT):
        self.ws = ws
        self.id = player_id
        self.name = name[:15] if name else f"Player{player_id}"
        self.color = f"hsl({random.randint(0, 360)}, 70%, 50%)"
        self.target_x = map_w / 2
        self.target_y = map_h / 2
        self.score = 0
        self.is_bot = is_bot
        self.map_w = map_w
        self.map_h = map_h
        self.cells: list[Cell] = [Cell(
            random.randint(100, map_w - 100),
            random.randint(100, map_h - 100),
            PLAYER_START_SIZE,
            map_w=map_w, map_h=map_h
        )]

    @property
    def total_size(self):
        return sum(c.size for c in self.cells)

    @property
    def center_x(self):
        if not self.cells:
            return self.map_w / 2
        total = self.total_size
        return sum(c.x * c.size for c in self.cells) / total

    @property
    def center_y(self):
        if not self.cells:
            return self.map_h / 2
        total = self.total_size
        return sum(c.y * c.size for c in self.cells) / total

    def respawn(self, map_w: int = None, map_h: int = None):
        w = map_w or self.map_w
        h = map_h or self.map_h
        self.map_w = w
        self.map_h = h
        self.cells = [Cell(
            random.randint(100, w - 100),
            random.randint(100, h - 100),
            PLAYER_START_SIZE,
            map_w=w, map_h=h
        )]


class Game:
    def __init__(self, num_bots: int = 5):
        self.players: dict[int, Player] = {}
        self.food: list[Food] = []
        self.cactus: list[Cactus] = []
        self.ejected: list[EjectedMass] = []
        self.next_id = 1
        self.running = False
        self.map_width = BASE_MAP_WIDTH
        self.map_height = BASE_MAP_HEIGHT
        self.scale_level = 0

        self.food_grid = SpatialGrid()
        self.cell_grid = SpatialGrid()
        self.ejected_grid = SpatialGrid()
        self.cactus_grid = SpatialGrid()

        self._cached_leaderboard = []

        for _ in range(FOOD_COUNT):
            self.food.append(Food(map_w=self.map_width, map_h=self.map_height))
        for _ in range(CACTUS_COUNT):
            self.cactus.append(Cactus(map_w=self.map_width, map_h=self.map_height))
        for i in range(num_bots):
            self.add_bot(BOT_NAMES[i % len(BOT_NAMES)])

    def _rebuild_grids(self):
        self.food_grid.clear()
        for i, f in enumerate(self.food):
            self.food_grid.insert((i, f), f.x, f.y)

        self.cell_grid.clear()
        for player in self.players.values():
            for cell in player.cells:
                self.cell_grid.insert((player, cell), cell.x, cell.y)

        self.ejected_grid.clear()
        for i, e in enumerate(self.ejected):
            self.ejected_grid.insert((i, e), e.x, e.y)

        self.cactus_grid.clear()
        for cactus in self.cactus:
            self.cactus_grid.insert(cactus, cactus.x, cactus.y)

    def _update_leaderboard(self):
        self._cached_leaderboard = sorted(
            [(p.name, p.score) for p in self.players.values()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

    def player_count(self):
        return sum(1 for p in self.players.values() if not p.is_bot)

    def check_scale_map(self):
        real_players = self.player_count()
        needed_scale = real_players // PLAYERS_PER_SCALE

        while self.scale_level < needed_scale:
            self.scale_level += 1
            self.map_width = BASE_MAP_WIDTH * (2 ** self.scale_level)
            self.map_height = BASE_MAP_HEIGHT * (2 ** self.scale_level)

            new_food = FOOD_COUNT * (2 ** self.scale_level) - len(self.food)
            new_cactus = CACTUS_COUNT * (2 ** self.scale_level) - len(self.cactus)

            for _ in range(max(0, new_food)):
                self.food.append(Food(map_w=self.map_width, map_h=self.map_height))
            for _ in range(max(0, new_cactus)):
                self.cactus.append(Cactus(map_w=self.map_width, map_h=self.map_height))

    def add_player(self, ws: WebSocket, name: str) -> Player | None:
        if self.player_count() >= MAX_PLAYERS:
            return None

        player = Player(ws, self.next_id, name, map_w=self.map_width, map_h=self.map_height)
        self.players[self.next_id] = player
        self.next_id += 1
        self.check_scale_map()
        return player

    def add_bot(self, name: str) -> Player:
        bot = Player(None, self.next_id, f"[BOT] {name}", is_bot=True, map_w=self.map_width, map_h=self.map_height)
        self.players[self.next_id] = bot
        self.next_id += 1
        return bot

    def remove_player(self, player_id: int):
        if player_id in self.players:
            del self.players[player_id]

    def feed(self, player: Player):
        for cell in player.cells:
            if cell.size < EJECT_COST + EJECT_SIZE:
                continue

            dx = player.target_x - cell.x
            dy = player.target_y - cell.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1:
                continue

            dist = math.sqrt(dist_sq)
            dx /= dist
            dy /= dist

            eject_x = cell.x + dx * (cell.radius + EJECT_SIZE)
            eject_y = cell.y + dy * (cell.radius + EJECT_SIZE)

            self.ejected.append(EjectedMass(
                eject_x, eject_y, dx * EJECT_SPEED, dy * EJECT_SPEED,
                player.color, map_w=self.map_width, map_h=self.map_height
            ))
            cell.size = math.sqrt(cell.size**2 - EJECT_SIZE**2)

    def split(self, player: Player):
        new_cells = []
        for cell in player.cells:
            if len(player.cells) + len(new_cells) >= MAX_CELLS:
                break
            if cell.size < SPLIT_MIN_SIZE:
                continue

            dx = player.target_x - cell.x
            dy = player.target_y - cell.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1:
                dx, dy = 1, 0
            else:
                dist = math.sqrt(dist_sq)
                dx /= dist
                dy /= dist

            new_size = cell.size / 1.4142135623730951
            cell.size = new_size
            cell.merge_time = MERGE_TIME

            new_cell = Cell(
                cell.x + dx * cell.radius,
                cell.y + dy * cell.radius,
                new_size, dx * 20, dy * 20,
                map_w=self.map_width, map_h=self.map_height
            )
            new_cell.merge_time = MERGE_TIME
            new_cells.append(new_cell)

        player.cells.extend(new_cells)

    def update_bots(self):
        for player in self.players.values():
            if not player.is_bot or not player.cells:
                continue

            main_cell = max(player.cells, key=lambda c: c.size)

            nearby_food = self.food_grid.query(main_cell.x, main_cell.y, 500)
            nearest_food = None
            nearest_dist_sq = float('inf')
            for _, f in nearby_food:
                dx = f.x - main_cell.x
                dy = f.y - main_cell.y
                dist_sq = dx * dx + dy * dy
                if dist_sq < nearest_dist_sq:
                    nearest_dist_sq = dist_sq
                    nearest_food = f

            flee_target = None
            nearby_cells = self.cell_grid.query(main_cell.x, main_cell.y, 200)
            for other_player, oc in nearby_cells:
                if other_player.id == player.id:
                    continue
                dx = oc.x - main_cell.x
                dy = oc.y - main_cell.y
                dist_sq = dx * dx + dy * dy
                if oc.size > main_cell.size * 1.1 and dist_sq < 40000:
                    flee_target = (main_cell.x - dx, main_cell.y - dy)
                    break

            chase_target = None
            if not flee_target:
                nearby_cells = self.cell_grid.query(main_cell.x, main_cell.y, 300)
                for other_player, oc in nearby_cells:
                    if other_player.id == player.id:
                        continue
                    dx = oc.x - main_cell.x
                    dy = oc.y - main_cell.y
                    dist_sq = dx * dx + dy * dy
                    if main_cell.size > oc.size * 1.1 and dist_sq < 90000:
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
        if len(player.cells) < 2:
            return

        merged = set()
        for i, c1 in enumerate(player.cells):
            if i in merged:
                continue
            for j, c2 in enumerate(player.cells[i+1:], i+1):
                if j in merged or c1.merge_time > 0 or c2.merge_time > 0:
                    continue

                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist_sq = dx * dx + dy * dy
                threshold = max(c1.radius, c2.radius) * 0.8
                threshold_sq = threshold * threshold

                if dist_sq < threshold_sq:
                    c1.size = math.sqrt(c1.size**2 + c2.size**2)
                    total = c1.size + c2.size
                    c1.x = (c1.x * c1.size + c2.x * c2.size) / total
                    c1.y = (c1.y * c1.size + c2.y * c2.size) / total
                    merged.add(j)

        player.cells = [c for i, c in enumerate(player.cells) if i not in merged]

    def push_apart_cells(self, player: Player):
        for i, c1 in enumerate(player.cells):
            for c2 in player.cells[i+1:]:
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist_sq = dx * dx + dy * dy
                min_dist = c1.radius + c2.radius
                min_dist_sq = min_dist * min_dist

                if dist_sq < min_dist_sq and dist_sq > 0:
                    dist = math.sqrt(dist_sq)
                    overlap = (min_dist - dist) / 2
                    dx /= dist
                    dy /= dist
                    c1.x += dx * overlap * 0.5
                    c1.y += dy * overlap * 0.5
                    c2.x -= dx * overlap * 0.5
                    c2.y -= dy * overlap * 0.5

    def update(self):
        self._rebuild_grids()
        self.update_bots()

        for player in self.players.values():
            for cell in player.cells:
                cell.move_towards(player.target_x, player.target_y)
            self.push_apart_cells(player)
            self.merge_cells(player)

        for e in self.ejected:
            e.update()

        for player in self.players.values():
            for cell in player.cells:
                nearby = self.food_grid.query(cell.x, cell.y, cell.radius + FOOD_SIZE)
                eaten_indices = []
                for idx, f in nearby:
                    dx = cell.x - f.x
                    dy = cell.y - f.y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < cell.radius * cell.radius:
                        eaten_indices.append(idx)
                        cell.size = math.sqrt(cell.size**2 + f.size**2 * 0.5)
                        player.score += int(f.size)

                for idx in eaten_indices:
                    self.food[idx] = Food(map_w=self.map_width, map_h=self.map_height)

        eaten_ejected = set()
        for player in self.players.values():
            for cell in player.cells:
                nearby = self.ejected_grid.query(cell.x, cell.y, cell.radius + EJECT_SIZE)
                for idx, e in nearby:
                    if idx in eaten_ejected or e.lifetime < 10:
                        continue
                    dx = cell.x - e.x
                    dy = cell.y - e.y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < cell.radius * cell.radius:
                        cell.size = math.sqrt(cell.size**2 + e.size**2 * 0.5)
                        player.score += int(e.size)
                        eaten_ejected.add(idx)

        for idx in sorted(eaten_ejected, reverse=True):
            self.ejected.pop(idx)

        for player in self.players.values():
            for cell in player.cells:
                nearby = self.cactus_grid.query(cell.x, cell.y, cell.radius + CACTUS_SIZE)
                for cactus in nearby:
                    dx = cell.x - cactus.x
                    dy = cell.y - cactus.y
                    dist_sq = dx * dx + dy * dy
                    threshold = cell.radius + cactus.size
                    if dist_sq < threshold * threshold:
                        if cell.size > PLAYER_START_SIZE * 1.5:
                            pieces = min(5, int(cell.size / 20))
                            for _ in range(pieces):
                                angle = random.uniform(0, 6.283185307179586)
                                vx = math.cos(angle) * EJECT_SPEED * 1.5
                                vy = math.sin(angle) * EJECT_SPEED * 1.5
                                self.ejected.append(EjectedMass(
                                    cell.x, cell.y, vx, vy, player.color, EJECT_SIZE,
                                    map_w=self.map_width, map_h=self.map_height
                                ))
                                cell.size = math.sqrt(max(PLAYER_START_SIZE**2, cell.size**2 - EJECT_SIZE**2))

        players_list = list(self.players.values())
        for p1 in players_list:
            for p2 in players_list:
                if p1.id >= p2.id:
                    continue

                cells_to_remove_p1 = set()
                cells_to_remove_p2 = set()

                for i, c1 in enumerate(p1.cells):
                    for j, c2 in enumerate(p2.cells):
                        dx = c1.x - c2.x
                        dy = c1.y - c2.y
                        dist_sq = dx * dx + dy * dy
                        threshold = max(c1.radius, c2.radius)

                        if dist_sq < threshold * threshold:
                            if c1.size > c2.size * 1.1:
                                c1.size = math.sqrt(c1.size**2 + c2.size**2 * 0.5)
                                p1.score += int(c2.size)
                                cells_to_remove_p2.add(j)
                            elif c2.size > c1.size * 1.1:
                                c2.size = math.sqrt(c2.size**2 + c1.size**2 * 0.5)
                                p2.score += int(c1.size)
                                cells_to_remove_p1.add(i)

                p1.cells = [c for i, c in enumerate(p1.cells) if i not in cells_to_remove_p1]
                p2.cells = [c for i, c in enumerate(p2.cells) if i not in cells_to_remove_p2]

        for player in self.players.values():
            if not player.cells:
                player.respawn(self.map_width, self.map_height)

        self._update_leaderboard()

    def get_state(self, for_player_id: int) -> dict:
        player = self.players.get(for_player_id)
        if not player:
            return {}

        view_size = 800 + player.total_size * 2
        px, py = player.center_x, player.center_y

        cells_data = []
        for p in self.players.values():
            for cell in p.cells:
                cells_data.append([
                    p.id, round(cell.x), round(cell.y), round(cell.size), p.name, p.color
                ])

        food_data = [
            [round(f.x), round(f.y), f.color]
            for f in self.food
            if abs(f.x - px) < view_size and abs(f.y - py) < view_size
        ]

        cactus_data = [
            [round(c.x), round(c.y), c.size]
            for c in self.cactus
            if abs(c.x - px) < view_size and abs(c.y - py) < view_size
        ]

        ejected_data = [
            [round(e.x), round(e.y), round(e.size), e.color]
            for e in self.ejected
            if abs(e.x - px) < view_size and abs(e.y - py) < view_size
        ]

        return {
            "t": "s",
            "p": cells_data,
            "f": food_data,
            "c": cactus_data,
            "e": ejected_data,
            "l": self._cached_leaderboard,
            "you": for_player_id,
            "map": [self.map_width, self.map_height]
        }

    async def _send_state(self, pid: int, player: Player) -> int | None:
        try:
            state = self.get_state(pid)
            await player.ws.send_text(json_dumps(state))
            return None
        except:
            return pid

    async def broadcast(self):
        players_snapshot = [(pid, p) for pid, p in self.players.items() if not p.is_bot]
        tasks = [self._send_state(pid, player) for pid, player in players_snapshot]
        results = await asyncio.gather(*tasks)

        for pid in results:
            if pid is not None:
                self.remove_player(pid)

    async def game_loop(self):
        self.running = True
        while self.running:
            self.update()
            await self.broadcast()
            await asyncio.sleep(1 / TICK_RATE)


num_bots = int(os.getenv("NUM_BOTS", "5"))
game = Game(num_bots=num_bots)


@app.on_event("startup")
async def startup():
    asyncio.create_task(game.game_loop())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str = ""):
    await websocket.accept()
    player = game.add_player(websocket, name)

    if player is None:
        await websocket.send_text(json_dumps({
            "t": "full",
            "msg": "Server full (max 50 players)"
        }))
        await websocket.close()
        return

    await websocket.send_text(json_dumps({
        "t": "welcome",
        "id": player.id,
        "map": [game.map_width, game.map_height]
    }))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json_loads(data)

            if msg.get("t") == "m":
                player.target_x = msg.get("x", player.target_x)
                player.target_y = msg.get("y", player.target_y)
            elif msg.get("t") == "w":
                game.feed(player)
            elif msg.get("t") == "s":
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
