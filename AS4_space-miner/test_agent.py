import neat
import pickle
import pygame
import os
import math
import json
import shutil
from datetime import datetime

# 🌟 1. 載入老師規定的固定測試環境 (期末考場)
from miner_harness import Spaceship, Mineral, Asteroid 

def relative_angle_to(ship, target):
    dx, dy = relative_position(ship, target, 800, 600)
    target_angle = math.atan2(dy, dx)
    angle_delta = target_angle - ship.angle
    return math.atan2(math.sin(angle_delta), math.cos(angle_delta))


def relative_position(source, target, width, height):
    dx = target.x - source.x
    dy = target.y - source.y
    half_width = width / 2
    half_height = height / 2
    if dx > half_width:
        dx -= width
    elif dx < -half_width:
        dx += width
    if dy > half_height:
        dy -= height
    elif dy < -half_height:
        dy += height
    return dx, dy


def distance_between(source, target, width, height):
    dx, dy = relative_position(source, target, width, height)
    return math.hypot(dx, dy)


def normalized_relative_vector(source, target, width, height):
    dx, dy = relative_position(source, target, width, height)
    return dx / (width / 2), dy / (height / 2)


def asteroid_in_front(ship, asteroid, width, height):
    dx, dy = relative_position(ship, asteroid, width, height)
    distance = math.hypot(dx, dy)
    if distance == 0:
        return 1

    heading_x = math.cos(ship.angle)
    heading_y = math.sin(ship.angle)
    return max(0, (heading_x * dx + heading_y * dy) / distance)


def asteroid_time_to_collision_signal(
    ship,
    asteroid,
    ship_velocity_x,
    ship_velocity_y,
    width,
    height,
    horizon=120
):
    dx, dy = relative_position(ship, asteroid, width, height)
    relative_vx = asteroid.speed_x - ship_velocity_x
    relative_vy = asteroid.speed_y - ship_velocity_y
    relative_speed_sq = relative_vx * relative_vx + relative_vy * relative_vy
    if relative_speed_sq == 0:
        return 0

    time_to_closest = -(
        dx * relative_vx + dy * relative_vy
    ) / relative_speed_sq
    if time_to_closest < 0 or time_to_closest > horizon:
        return 0

    closest_x = dx + relative_vx * time_to_closest
    closest_y = dy + relative_vy * time_to_closest
    closest_distance = math.hypot(closest_x, closest_y)
    danger_radius = ship.radius + asteroid.radius + 20
    if closest_distance > danger_radius:
        return 0

    return 1 - (time_to_closest / horizon)


def next_artifact_number(artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)

    numbers = []
    for name in os.listdir(artifacts_dir):
        if name.startswith("test_"):
            number_part = name[5:].split(".", 1)[0]
            if number_part.isdigit():
                numbers.append(int(number_part))

    return max(numbers, default=0) + 1


def save_test_artifacts(config_file, genome_path, results):
    local_dir = os.path.dirname(__file__)
    artifacts_dir = os.path.join(local_dir, "artifacts")
    test_number = next_artifact_number(artifacts_dir)
    test_dir = os.path.join(artifacts_dir, f"test_{test_number}")
    os.makedirs(test_dir, exist_ok=False)

    config_artifact_path = os.path.join(test_dir, "configuration.txt")
    shutil.copy2(config_file, config_artifact_path)

    artifact_data = {
        "test_number": test_number,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": os.path.abspath(config_file),
        "genome_path": os.path.abspath(genome_path),
        "results": results,
    }

    results_artifact_path = os.path.join(test_dir, "game_results.json")
    with open(results_artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f, indent=2, ensure_ascii=False)

    pickle_artifact_path = os.path.join(test_dir, f"test_{test_number}.pkl")
    with open(pickle_artifact_path, "wb") as f:
        pickle.dump(artifact_data, f)

    return test_dir


def test_best_agent(config_file, genome_path="winner.pkl"):
    if not os.path.isabs(genome_path):
        genome_path = os.path.join(os.path.dirname(__file__), genome_path)

    # --- 載入大腦與設定 ---
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    print(f"📥 正在讀取 AI 存檔: {genome_path} ...")
    with open(genome_path, "rb") as f:
        winner = pickle.load(f)

    # 將存檔轉換成神經網路
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # --- 準備遊戲視窗 ---
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI 最終測試展示")
    clock = pygame.time.Clock()
    max_distance = math.hypot(WIDTH, HEIGHT)

    # 產生固定位置的物件
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    
    alive_time = 0
    running = True
    death_reason = "window_closed"
    ship_velocity_x = 0
    ship_velocity_y = 0

    print("🚀 測試開始！")

    # --- 遊戲主迴圈 ---
    while running:
        alive_time += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 尋找最近的物件
        closest_mineral = min(
            (m for m in minerals),
            key=lambda m: distance_between(ship, m, WIDTH, HEIGHT),
            default=None
        )
        closest_asteroid = min(
            (a for a in asteroids),
            key=lambda a: distance_between(ship, a, WIDTH, HEIGHT)
        )
        
        
        mineral_distance = (
            distance_between(ship, closest_mineral, WIDTH, HEIGHT) / max_distance
            if closest_mineral else 0
        )
        mineral_relative_angle = (
            relative_angle_to(ship, closest_mineral) / math.pi
            if closest_mineral else 0
        )
        asteroid_distance = (
            distance_between(ship, closest_asteroid, WIDTH, HEIGHT) / max_distance
        )
        asteroid_relative_angle = relative_angle_to(ship, closest_asteroid) / math.pi
        mineral_relative_x, mineral_relative_y = (
            normalized_relative_vector(ship, closest_mineral, WIDTH, HEIGHT)
            if closest_mineral else (0, 0)
        )
        asteroid_relative_x, asteroid_relative_y = normalized_relative_vector(
            ship,
            closest_asteroid,
            WIDTH,
            HEIGHT
        )
        asteroid_velocity_x = closest_asteroid.speed_x / ship.speed
        asteroid_velocity_y = closest_asteroid.speed_y / ship.speed

        # Get inputs (handle case where all minerals are collected)
        inputs = [
            mineral_distance,
            mineral_relative_angle,
            asteroid_distance,
            asteroid_relative_angle,
            ship.fuel / 100.0,
            mineral_relative_x,
            mineral_relative_y,
            asteroid_relative_x,
            asteroid_relative_y,
            asteroid_velocity_x,
            asteroid_velocity_y,
            asteroid_in_front(ship, closest_asteroid, WIDTH, HEIGHT),
            asteroid_time_to_collision_signal(
                ship,
                closest_asteroid,
                ship_velocity_x,
                ship_velocity_y,
                WIDTH,
                HEIGHT
            )
        ]
        
        

        # 讓 AI 思考並做出動作
        output = net.activate(inputs)

        # 執行動作
        ship.angle += (output[0] * 2 - 1) * 0.1
        if output[1] > 0.5: 
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            ship.move(dx, dy)
            ship_velocity_x = dx
            ship_velocity_y = dy
        else:
            ship_velocity_x = 0
            ship_velocity_y = 0
        if output[2] > 0.5:
            old_mineral_count = ship.minerals
            old_fuel = ship.fuel

            ship.mine(minerals)

           # 🌟 2. 完美的防呆機制
            if ship.minerals > old_mineral_count:  # 確定真的有吃到礦石
                # 如果吃完礦石後，油量居然跟本來一樣（而且油箱還沒滿），代表老師真的忘記寫了！
                if ship.fuel == old_fuel and ship.fuel < 100.0:
                    ship.fuel = min(100.0, ship.fuel + 10.0)  # 我們自己手動加

            # 確保礦物被吃掉後會再生 (修復老師提到的 Bug)
            if len(minerals) < 3:
                minerals.extend(Mineral() for _ in range(2))

        # 繪製畫面
        screen.fill((0, 0, 0)) # 黑色背景
        for mineral in minerals:
            mineral.draw()
        for asteroid in asteroids:
            asteroid.move()
            asteroid.draw()
        ship.draw()
        
        # 在畫面上顯示即時資訊
        font = pygame.font.SysFont(None, 36)
        screen.blit(font.render(f"Minerals: {ship.minerals}", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(f"Fuel: {ship.fuel:.1f}", True, (255, 255, 255)), (10, 50))
        
        pygame.display.flip()
        clock.tick(30) # 控制在 30 FPS，方便錄影

        # 🌟 3. 死亡條件判定
        asteroid_collision = (
            distance_between(ship, closest_asteroid, WIDTH, HEIGHT)
            < ship.radius + closest_asteroid.radius
        )
        out_of_fuel = ship.fuel <= 0

        if asteroid_collision:
            running = False
            death_reason = "asteroid_collision"
            print("💥 死因：撞到小行星了！")
        elif out_of_fuel:
            running = False
            death_reason = "out_of_fuel"
            print("⛽ 死因：燃料耗盡！")
        elif alive_time > 5000:
            running = False
            death_reason = "time_limit"
            print("⏱️ 死因：時間到！")
        
        if asteroid_collision or out_of_fuel or alive_time > 5000:
            running = False
            print("💥 飛船損毀或燃料耗盡，遊戲結束！")

    # 🌟 4. 根據老師的公式計算最終成績
    final_score = (alive_time / 4) + (ship.minerals * 100)
    print("-" * 30)
    print(f"Time alive: {alive_time}")
    print(f"Minerals gathered: {ship.minerals}")
    print(f"Final Score: {final_score:.2f}")
    print("-" * 30)

    results = {
        "time_alive": alive_time,
        "minerals_gathered": ship.minerals,
        "final_score": final_score,
        "death_reason": death_reason,
        "fuel_remaining": ship.fuel,
    }
    artifact_path = save_test_artifacts(config_file, genome_path, results)
    print(f"Saved test artifacts to: {artifact_path}")

    pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    test_best_agent(config_file)
