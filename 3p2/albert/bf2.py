exchange_rates = [
    [1,    1.45, 0.52, 0.72],  # Snowballs to Snowballs, Pizzas, Silicon Nuggets, SeaShells
    [0.7,  1,    0.31, 0.48],  # Pizzas
    [1.95, 3.1,  1,    1.49],  # Silicon Nuggets
    [1.34, 1.98, 0.64, 1]      # SeaShells
]
src = 3
q = [[1, [3]]] # money, path

max_money = 0
max_money_path = []


while q:
    money, path = q.pop(0)
    print(path)
    print(money)
    print()

    if path[-1] == src:
        if money > max_money:
            print("New max money:", money)
            print("New max money path:", path)
            max_money = money
            max_money_path = path

    if len(path) == 6:
        continue

    for i in range(len(exchange_rates)):
        if i == path[-1]:
            continue

        new_money = money * exchange_rates[path[-1]][i]
        new_path = path.copy()
        new_path.append(i)

        q.append([new_money, new_path])

print("Max money:", max_money)
print("Max money path:", max_money_path)

