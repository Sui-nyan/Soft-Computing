import neat
import pickle
import pygame
import os
import math

# 🌟 1. 載入老師規定的固定測試環境 (期末考場)
from miner_harness import Spaceship, Mineral, Asteroid 

def test_best_agent(config_file, genome_path="winner.pkl"):
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

    # 產生固定位置的物件
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    
    alive_time = 0
    running = True

    print("🚀 測試開始！")

    # --- 遊戲主迴圈 ---
    while running:
        alive_time += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 尋找最近的物件
        closest_mineral = min((m for m in minerals), key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), default=None)
        closest_asteroid = min((a for a in asteroids), key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y))
        
        # 🌟 2. 讓 AI 看環境 (這裡的 5 個 inputs 必須跟訓練時一模一樣)
        inputs = [
            math.hypot(ship.x - closest_mineral.x)/WIDTH if closest_mineral else 0,
            math.atan2(closest_mineral.y-ship.y, closest_mineral.x-ship.x)/math.pi if closest_mineral else 0,
            math.hypot(ship.x - closest_asteroid.x)/WIDTH,
            ship.fuel / 100.0
        ]
        

        # 讓 AI 思考並做出動作
        output = net.activate(inputs)

        # 執行動作
        ship.angle += (output[0] * 2 - 1) * 0.1
        if output[1] > 0.5: 
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            ship.move(dx, dy)
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
        asteroid_collision = math.hypot(ship.x-closest_asteroid.x, ship.y-closest_asteroid.y) < ship.radius + closest_asteroid.radius
        out_of_fuel = ship.fuel <= 0

        if asteroid_collision:
            running = False
            print("💥 死因：撞到小行星了！")
        elif out_of_fuel:
            running = False
            print("⛽ 死因：燃料耗盡！")
        elif alive_time > 5000:
            running = False
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

    pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    test_best_agent(config_file)