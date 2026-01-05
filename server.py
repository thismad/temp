"""
Micro Agar.io Clone - Server
~50 players, lightweight, real-time
OPTIMIZED v2: spatial grid for all collisions, cached state data, static cactus grid
"""

import asyncio
import random
import math
import os
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

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
SPEED_BASE = 7
SIZE_SPEED_FACTOR = 0.02
EJECT_SIZE = 12
EJECT_SPEED = 15
EJECT_COST = 15
SPLIT_MIN_SIZE = 35
MERGE_TIME = 200
MAX_CELLS = 16
GRID_CELL_SIZE = 150  # Larger cells = fewer buckets to check


class SpatialGrid:
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
        grid = self.grid
        for dx in range(-cells_needed, cells_needed + 1):
            for dy in range(-cells_needed, cells_needed + 1):
                bucket = grid.get((cx + dx, cy + dy))
                if bucket:
                    results.extend(bucket)
        return results

    def query_rect(self, x: float, y: float, half_w: float, half_h: float) -> list:
        """Query a rectangular area - more efficient for viewport culling"""
        results = []
        x1, y1 = int((x - half_w) // self.cell_size), int((y - half_h) // self.cell_size)
        x2, y2 = int((x + half_w) // self.cell_size), int((y + half_h) // self.cell_size)
        grid = self.grid
        for gx in range(x1, x2 + 1):
            for gy in range(y1, y2 + 1):
                bucket = grid.get((gx, gy))
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
            spd = self.speed
            self.x += (dx / dist) * spd
            self.y += (dy / dist) * spd

        r = self.size
        self.x = max(r, min(self.map_w - r, self.x))
        self.y = max(r, min(self.map_h - r, self.y))

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

    def get_center(self):
        """Return (center_x, center_y, total_size) in one pass"""
        if not self.cells:
            return self.map_w / 2, self.map_h / 2, 0
        total = 0
        wx = wy = 0
        for c in self.cells:
            total += c.size
            wx += c.x * c.size
            wy += c.y * c.size
        return wx / total, wy / total, total

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

        # Spatial grids
        self.food_grid = SpatialGrid()
        self.cell_grid = SpatialGrid()
        self.ejected_grid = SpatialGrid()
        self.cactus_grid = SpatialGrid()
        self._cactus_grid_dirty = True  # Only rebuild when cactus added

        # Cached data (updated once per tick)
        self._cached_leaderboard = []
        self._cached_cells_data = []

        for _ in range(FOOD_COUNT):
            self.food.append(Food(map_w=self.map_width, map_h=self.map_height))
        for _ in range(CACTUS_COUNT):
            self.cactus.append(Cactus(map_w=self.map_width, map_h=self.map_height))
        for i in range(num_bots):
            self.add_bot(BOT_NAMES[i % len(BOT_NAMES)])

    def _rebuild_dynamic_grids(self):
        """Rebuild only grids that change every tick"""
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

        # Cactus grid only rebuilt when dirty (new cactus added)
        if self._cactus_grid_dirty:
            self.cactus_grid.clear()
            for cactus in self.cactus:
                self.cactus_grid.insert(cactus, cactus.x, cactus.y)
            self._cactus_grid_dirty = False

    def _update_cached_data(self):
        """Update all cached data once per tick"""
        players = self.players.values()

        # Leaderboard (all players)
        self._cached_leaderboard = sorted(
            [(p.name, p.score) for p in players],
            key=lambda x: x[1],
            reverse=True
        )

        # All cells data (for all players to see)
        cells = []
        for p in players:
            pid, name, color = p.id, p.name, p.color
            for cell in p.cells:
                cells.append([pid, round(cell.x), round(cell.y), round(cell.size), name, color])
        self._cached_cells_data = cells

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
                self._cactus_grid_dirty = True

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
        self.players.pop(player_id, None)

    def feed(self, player: Player):
        mw, mh = self.map_width, self.map_height
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

            self.ejected.append(EjectedMass(
                cell.x + dx * (cell.size + EJECT_SIZE),
                cell.y + dy * (cell.size + EJECT_SIZE),
                dx * EJECT_SPEED, dy * EJECT_SPEED,
                player.color, map_w=mw, map_h=mh
            ))
            cell.size = math.sqrt(cell.size**2 - EJECT_SIZE**2)

    def split(self, player: Player):
        new_cells = []
        mw, mh = self.map_width, self.map_height
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

            new_size = cell.size * 0.7071067811865476  # 1/sqrt(2)
            cell.size = new_size
            cell.merge_time = MERGE_TIME

            new_cell = Cell(
                cell.x + dx * cell.size,
                cell.y + dy * cell.size,
                new_size, dx * 20, dy * 20,
                map_w=mw, map_h=mh
            )
            new_cell.merge_time = MERGE_TIME
            new_cells.append(new_cell)

        player.cells.extend(new_cells)

    def update_bots(self):
        for player in self.players.values():
            if not player.is_bot or not player.cells:
                continue

            main_cell = max(player.cells, key=lambda c: c.size)
            mx, my, ms = main_cell.x, main_cell.y, main_cell.size

            # Combined query for threats and prey (300 radius covers both)
            nearby_cells = self.cell_grid.query(mx, my, 300)
            
            flee_target = chase_target = None
            for other_player, oc in nearby_cells:
                if other_player.id == player.id:
                    continue
                dx = oc.x - mx
                dy = oc.y - my
                dist_sq = dx * dx + dy * dy
                
                # Threat detection (flee)
                if not flee_target and oc.size > ms * 1.1 and dist_sq < 40000:
                    flee_target = (mx - dx, my - dy)
                # Prey detection (chase)
                elif not chase_target and ms > oc.size * 1.1 and dist_sq < 90000:
                    chase_target = (oc.x, oc.y)

            if flee_target:
                player.target_x, player.target_y = flee_target
            elif chase_target:
                player.target_x, player.target_y = chase_target
            else:
                # Find nearest food
                nearby_food = self.food_grid.query(mx, my, 400)
                if nearby_food:
                    nearest = min(nearby_food, key=lambda f: (f[1].x - mx)**2 + (f[1].y - my)**2)
                    player.target_x, player.target_y = nearest[1].x, nearest[1].y

    def merge_cells(self, player: Player):
        cells = player.cells
        if len(cells) < 2:
            return

        merged = set()
        for i in range(len(cells)):
            if i in merged:
                continue
            c1 = cells[i]
            if c1.merge_time > 0:
                continue
            for j in range(i + 1, len(cells)):
                if j in merged:
                    continue
                c2 = cells[j]
                if c2.merge_time > 0:
                    continue

                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist_sq = dx * dx + dy * dy
                threshold = max(c1.size, c2.size) * 0.8

                if dist_sq < threshold * threshold:
                    c1.size = math.sqrt(c1.size**2 + c2.size**2)
                    total = c1.size + c2.size
                    c1.x = (c1.x * c1.size + c2.x * c2.size) / total
                    c1.y = (c1.y * c1.size + c2.y * c2.size) / total
                    merged.add(j)

        if merged:
            player.cells = [c for i, c in enumerate(cells) if i not in merged]

    def push_apart_cells(self, player: Player):
        cells = player.cells
        n = len(cells)
        for i in range(n):
            c1 = cells[i]
            for j in range(i + 1, n):
                c2 = cells[j]
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist_sq = dx * dx + dy * dy
                min_dist = c1.size + c2.size

                if dist_sq < min_dist * min_dist and dist_sq > 0:
                    dist = math.sqrt(dist_sq)
                    overlap = (min_dist - dist) * 0.25
                    dx /= dist
                    dy /= dist
                    c1.x += dx * overlap
                    c1.y += dy * overlap
                    c2.x -= dx * overlap
                    c2.y -= dy * overlap

    def update(self):
        self._rebuild_dynamic_grids()
        self.update_bots()

        # Move and process player cells
        for player in self.players.values():
            tx, ty = player.target_x, player.target_y
            for cell in player.cells:
                cell.move_towards(tx, ty)
            self.push_apart_cells(player)
            self.merge_cells(player)

        # Update ejected masses
        for e in self.ejected:
            e.update()

        # Food collisions (spatial grid)
        mw, mh = self.map_width, self.map_height
        for player in self.players.values():
            for cell in player.cells:
                r_sq = cell.size * cell.size
                nearby = self.food_grid.query(cell.x, cell.y, cell.size + FOOD_SIZE)
                eaten = []
                for idx, f in nearby:
                    dx = cell.x - f.x
                    dy = cell.y - f.y
                    if dx * dx + dy * dy < r_sq:
                        eaten.append(idx)
                        cell.size = math.sqrt(cell.size**2 + f.size**2 * 0.5)
                        player.score += int(f.size)
                for idx in eaten:
                    self.food[idx] = Food(map_w=mw, map_h=mh)

        # Ejected mass collisions
        eaten_ejected = set()
        for player in self.players.values():
            for cell in player.cells:
                r_sq = cell.size * cell.size
                nearby = self.ejected_grid.query(cell.x, cell.y, cell.size + EJECT_SIZE)
                for idx, e in nearby:
                    if idx in eaten_ejected or e.lifetime < 10:
                        continue
                    dx = cell.x - e.x
                    dy = cell.y - e.y
                    if dx * dx + dy * dy < r_sq:
                        cell.size = math.sqrt(cell.size**2 + e.size**2 * 0.5)
                        player.score += int(e.size)
                        eaten_ejected.add(idx)

        if eaten_ejected:
            self.ejected = [e for i, e in enumerate(self.ejected) if i not in eaten_ejected]

        # Cactus collisions
        for player in self.players.values():
            for cell in player.cells:
                nearby = self.cactus_grid.query(cell.x, cell.y, cell.size + CACTUS_SIZE)
                for cactus in nearby:
                    dx = cell.x - cactus.x
                    dy = cell.y - cactus.y
                    threshold = cell.size + cactus.size
                    if dx * dx + dy * dy < threshold * threshold:
                        if cell.size > PLAYER_START_SIZE * 1.5:
                            pieces = min(5, int(cell.size / 20))
                            for _ in range(pieces):
                                angle = random.random() * 6.283185307179586
                                self.ejected.append(EjectedMass(
                                    cell.x, cell.y,
                                    math.cos(angle) * EJECT_SPEED * 1.5,
                                    math.sin(angle) * EJECT_SPEED * 1.5,
                                    player.color, EJECT_SIZE, mw, mh
                                ))
                                cell.size = math.sqrt(max(PLAYER_START_SIZE**2, cell.size**2 - EJECT_SIZE**2))

        # Player vs Player collisions - track cells to remove
        cells_to_remove = {}  # player_id -> set of cell indices
        for player in self.players.values():
            for i, cell in enumerate(player.cells):
                if cells_to_remove.get(player.id) and i in cells_to_remove[player.id]:
                    continue
                nearby = self.cell_grid.query(cell.x, cell.y, cell.size * 2)
                for other_player, other_cell in nearby:
                    if other_player.id == player.id:
                        continue
                    try:
                        j = other_player.cells.index(other_cell)
                    except ValueError:
                        continue
                    if cells_to_remove.get(other_player.id) and j in cells_to_remove[other_player.id]:
                        continue

                    dx = cell.x - other_cell.x
                    dy = cell.y - other_cell.y
                    dist_sq = dx * dx + dy * dy
                    threshold = max(cell.size, other_cell.size)

                    if dist_sq < threshold * threshold:
                        if cell.size > other_cell.size * 1.1:
                            cell.size = math.sqrt(cell.size**2 + other_cell.size**2 * 0.5)
                            player.score += int(other_cell.size)
                            if other_player.id not in cells_to_remove:
                                cells_to_remove[other_player.id] = set()
                            cells_to_remove[other_player.id].add(j)
                        elif other_cell.size > cell.size * 1.1:
                            other_cell.size = math.sqrt(other_cell.size**2 + cell.size**2 * 0.5)
                            other_player.score += int(cell.size)
                            if player.id not in cells_to_remove:
                                cells_to_remove[player.id] = set()
                            cells_to_remove[player.id].add(i)
                            break

        # Remove eaten cells
        for pid, indices in cells_to_remove.items():
            if pid in self.players:
                self.players[pid].cells = [c for i, c in enumerate(self.players[pid].cells) if i not in indices]

        # Respawn players with no cells
        for player in self.players.values():
            if not player.cells:
                player.respawn(mw, mh)

        # Update cached data
        self._update_cached_data()

    def get_state(self, for_player_id: int) -> dict:
        player = self.players.get(for_player_id)
        if not player:
            return {}

        px, py, total_size = player.get_center()
        view_size = 1600 + total_size * 4

        # Use spatial grid for viewport culling
        food_data = [
            [round(f.x), round(f.y), f.color]
            for _, f in self.food_grid.query_rect(px, py, view_size, view_size)
        ]

        cactus_data = [
            [round(c.x), round(c.y), c.size]
            for c in self.cactus_grid.query_rect(px, py, view_size, view_size)
        ]

        ejected_data = [
            [round(e.x), round(e.y), round(e.size), e.color]
            for _, e in self.ejected_grid.query_rect(px, py, view_size, view_size)
        ]

        return {
            "t": "s",
            "p": self._cached_cells_data,
            "f": food_data,
            "c": cactus_data,
            "e": ejected_data,
            "l": self._cached_leaderboard,
            "you": for_player_id,
            "map": [self.map_width, self.map_height]
        }

    async def _send_state(self, pid: int, player: Player) -> int | None:
        try:
            await player.ws.send_text(json_dumps(self.get_state(pid)))
            return None
        except:
            return pid

    async def broadcast(self):
        players = [(pid, p) for pid, p in self.players.items() if not p.is_bot]
        if not players:
            return
        results = await asyncio.gather(*[self._send_state(pid, p) for pid, p in players])
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
        await websocket.send_text(json_dumps({"t": "full", "msg": "Server full"}))
        await websocket.close()
        return

    await websocket.send_text(json_dumps({
        "t": "welcome",
        "id": player.id,
        "map": [game.map_width, game.map_height]
    }))

    try:
        while True:
            msg = json_loads(await websocket.receive_text())
            t = msg.get("t")
            if t == "m":
                player.target_x = msg.get("x", player.target_x)
                player.target_y = msg.get("y", player.target_y)
            elif t == "w":
                game.feed(player)
            elif t == "s":
                game.split(player)
    except WebSocketDisconnect:
        game.remove_player(player.id)
    except:
        game.remove_player(player.id)


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
