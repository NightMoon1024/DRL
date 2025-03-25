from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 預設 Grid Size，可由前端指定 (範圍 5~9)
GRID_SIZE = 5
ACTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
ACTION_SYMBOLS = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
GAMMA = 0.9  # 折扣因子
THRESHOLD = 1e-3  # 迭代收斂閾值
REWARD = -1  # 每一步的懲罰

# 初始化
start_pos = (0, 0)
end_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
obstacles = set()

# 初始化價值函數 & 策略
values = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.random.choice(list(ACTIONS.keys()), (GRID_SIZE, GRID_SIZE))


def value_iteration():
    """ 使用價值迭代計算最佳價值函數與策略 """
    global values, policy
    values = np.zeros((GRID_SIZE, GRID_SIZE))  # 確保 `values` 大小匹配 GRID_SIZE
    policy = np.full((GRID_SIZE, GRID_SIZE), '', dtype='<U1')  # 初始化 policy 為空字串

    new_values = np.copy(values)

    while True:
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) in obstacles or (i, j) == end_pos:
                    continue

                max_value = float('-inf')
                best_action = None
                for action, (di, dj) in ACTIONS.items():
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and (ni, nj) not in obstacles:
                        v = REWARD + GAMMA * values[ni, nj]
                        if v > max_value:
                            max_value = v
                            best_action = action

                new_values[i, j] = max_value
                if best_action is not None:
                    policy[i, j] = ACTION_SYMBOLS[best_action]  # 存入最佳行動
                delta = max(delta, abs(new_values[i, j] - values[i, j]))

        values = np.copy(new_values)
        if delta < THRESHOLD:
            break



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/update', methods=['POST'])
def update_grid():
    """ 接收前端數據，更新網格設定 """
    global start_pos, end_pos, obstacles, GRID_SIZE
    data = request.json
    GRID_SIZE = data['grid_size']
    start_pos = tuple(data['start'])  # 更新起點
    end_pos = tuple(data['end'])  # 更新終點
    obstacles = {tuple(ob) for ob in data['obstacles']}  # 更新障礙物

    print(f"Updated Start: {start_pos}, End: {end_pos}, Obstacles: {obstacles}")

    value_iteration()  # 重新計算最佳策略

    response = {
        'values': values.tolist(),
        'policy': [[policy[i, j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
    }
    print("Updated Value Function:", response['values'])  # Debugging
    return jsonify(response)






if __name__ == '__main__':
    app.run(debug=True)
