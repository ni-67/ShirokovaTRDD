import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# 1. ПАРАМЕТРЫ

R = 8.314462618    # Универсальная газовая постоянная, Дж/(моль*K), ГОСТ Р 8.974-2019
g0 = 9.80665       # Ускорение свободного падения у поверхности, м/с^2, гост 4401-81
M_air = 0.0289644  # Молярная масса сухого воздуха, кг/моль, гост 4401-81

# Параметры аппарата
THRUST = 520000.0     # Тяга двигателя, Н, расчеты
MASS_INITIAL = 26250.0  # Начальная масса, кг, расчеты
O2_FLOW_RATE = 29.398   # расход кислорода, кг/с, расчеты
v0 = 0.1             # Начальная скорость, м/с
A = 0.675            # Площадь входного сечения, м^2, расчеты
C_d = 0.47            # Коэффициент сопротивления
S_ref = 0.675        # Характерная площадь, м^2, расчеты

# Управление углом тангажа
INITIAL_ANGLE = 0.1    # начальный угол, град 
TARGET_ANGLE = 22.0    # оптимальный угол, град
PITCH_RATE = 0.5       # скорость увеличения угла, град/с

# Стандартные условия
p0 = 101308.0       
T0 = 288.15


# 2. ГЛАДКАЯ МОДЕЛЬ АТМОСФЕРЫ БЕЗ РЕКУРСИИ
def create_smooth_atmosphere(max_h=300000, n_points=1000):
   
    h_points = np.linspace(0, max_h, n_points)
    
    # Температурный профиль
    T_points = np.zeros_like(h_points)
    for i, h in enumerate(h_points):
        h_km = h / 1000
        
        if h <= 11000:
            T = T0 - 0.0065 * h
        elif h <= 20000:
            T = 216.65 + 0.001 * (h - 11000) * 0.1
        elif h <= 32000:
            T = 216.65 + 0.001 * (h - 20000)
        elif h <= 47000:
            T = 228.65 + 0.0028 * (h - 32000)
        elif h <= 51000:
            T = 270.65
        elif h <= 71000:
            T = 270.65 - 0.0028 * (h - 51000)
        elif h <= 86000:
            T = 214.65 - 0.002 * (h - 71000)
        else:
            T_base = 186.65
            T_rise = 1000.0
            h_norm = (h - 86000) / 200000
            T = T_base + (T_rise - T_base) * (1 - np.exp(-h_norm))
        
        T_points[i] = T
    
    # Давление интеграл без рекурсии
    p_points = np.zeros_like(h_points)
    p_points[0] = p0
    
    for i in range(1, len(h_points)):
        dh = h_points[i] - h_points[i-1]
        T_avg = (T_points[i] + T_points[i-1]) / 2
        H_avg = R * T_avg / (g0 * M_air)
        p_points[i] = p_points[i-1] * np.exp(-dh / H_avg)
    
    # Плотность
    rho_points = p_points * M_air / (R * T_points)
    
    # Массовая доля кислорода
    o2_frac_points = np.zeros_like(h_points)
    for i, h in enumerate(h_points):
        if h <= 80000:
            o2_frac = 0.2314
        elif h <= 120000:
            x = (h - 80000) / 40000
            o2_frac = 0.2314 * (1 - 0.5 * (1 - np.exp(-x)))
        else:
            decay = np.exp(-(h - 120000) / 80000)
            o2_frac = 0.2314 * 0.5 * decay
        
        o2_frac_points[i] = max(o2_frac, 0.001)
    
    # интерполяционные функции
    h_km_points = h_points / 1000
    
    p_interp = interp1d(h_km_points, p_points, kind='cubic', fill_value='extrapolate')
    T_interp = interp1d(h_km_points, T_points, kind='cubic', fill_value='extrapolate')
    rho_interp = interp1d(h_km_points, rho_points, kind='cubic', fill_value='extrapolate')
    o2_interp = interp1d(h_km_points, o2_frac_points, kind='cubic', fill_value='extrapolate')
    
    return {
        'h_km': h_km_points,
        'p': p_points,
        'T': T_points,
        'rho': rho_points,
        'o2_frac': o2_frac_points,
        'p_interp': p_interp,
        'T_interp': T_interp,
        'rho_interp': rho_interp,
        'o2_interp': o2_interp
    }

# атмосферные профили
atmosphere = create_smooth_atmosphere()

# 3. РАСЧЁТ ТРАЕКТОРИИ
def calculate_smooth_trajectory():

    
    # Расход топлива с увеличенным соотношением топлива
    fuel_ratio = 6.0  # Увеличенное соотношение топлива
    total_fuel_flow = O2_FLOW_RATE * (1 + 1/fuel_ratio)
    fuel_mass = MASS_INITIAL * 0.8
    burn_time = fuel_mass / total_fuel_flow
    
    # Параметры интегрирования
    dt = 0.5
    t_max = min(3000, burn_time + 500)
    n_steps = int(t_max / dt)
    
    # Массивы результатов
    time = np.zeros(n_steps)
    height = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    velocity_x = np.zeros(n_steps)  # Горизонтальная компонента скорости
    velocity_y = np.zeros(n_steps)  # Вертикальная компонента скорости
    acceleration = np.zeros(n_steps)
    mass = np.zeros(n_steps)
    pressure = np.zeros(n_steps)
    o2_available = np.zeros(n_steps)
    drag_force = np.zeros(n_steps)  # Массив для аэродинамического сопротивления
    horizontal_range = np.zeros(n_steps)  # Горизонтальная дальность
    
    # Начальные условия с начальным углом
    angle_rad = np.radians(INITIAL_ANGLE)
    
    time[0] = 0
    height[0] = 0
    horizontal_range[0] = 0
    velocity[0] = v0
    velocity_x[0] = v0 * np.cos(angle_rad)
    velocity_y[0] = v0 * np.sin(angle_rad)
    mass[0] = MASS_INITIAL
    
    # Интерполяционные функции
    p_interp = atmosphere['p_interp']
    rho_interp = atmosphere['rho_interp']
    o2_interp = atmosphere['o2_interp']
    
    for i in range(1, n_steps):
        time[i] = time[i-1] + dt
        
        # Масса
        if time[i] <= burn_time:
            mass[i] = mass[i-1] - total_fuel_flow * dt
        else:
            mass[i] = mass[i-1]
        
        # Атмосферные параметры через интерполяцию
        h_km_prev = height[i-1] / 1000
        
        # гладкая интерполяция
        pressure[i-1] = float(p_interp(h_km_prev))
        rho = float(rho_interp(h_km_prev))
        o2_frac = float(o2_interp(h_km_prev))
        
        # Доступный кислород 
        air_flow = rho * velocity[i-1] * A
        o2_available[i-1] = air_flow * o2_frac
        
        # Силы
        g = g0 * (6371e3 / (6371e3 + height[i-1]))**2
        F_gravity = mass[i-1] * g
        F_drag = 0.5 * rho * velocity[i-1]**2 * C_d * S_ref
        drag_force[i-1] = F_drag
        
        # Текущий угол тангажа (плавно растёт от INITIAL_ANGLE до TARGET_ANGLE)
        current_angle_deg = min(INITIAL_ANGLE + PITCH_RATE * time[i], TARGET_ANGLE)
        angle_rad = np.radians(current_angle_deg)
        
        # Компоненты тяги с текущим углом
        F_thrust = THRUST if time[i] <= burn_time else 0
        F_thrust_x = F_thrust * np.cos(angle_rad)
        F_thrust_y = F_thrust * np.sin(angle_rad)
        
        # Компоненты силы сопротивления направлены против вектора скорости
        if velocity[i-1] > 0:
            F_drag_x = -F_drag * (velocity_x[i-1] / velocity[i-1])
            F_drag_y = -F_drag * (velocity_y[i-1] / velocity[i-1])
        else:
            F_drag_x = 0
            F_drag_y = 0
        
        # Компоненты силы тяжести
        F_gravity_x = 0
        F_gravity_y = -F_gravity
        
        # Компоненты ускорения
        a_x = (F_thrust_x + F_drag_x) / mass[i-1]
        a_y = (F_thrust_y + F_drag_y + F_gravity_y) / mass[i-1]
        
        # Полное ускорение
        acceleration[i-1] = np.sqrt(a_x**2 + a_y**2)
        
        # Интегрирование скоростей
        velocity_x[i] = velocity_x[i-1] + a_x * dt
        velocity_y[i] = velocity_y[i-1] + a_y * dt
        
        # Полная скорость
        velocity[i] = np.sqrt(velocity_x[i]**2 + velocity_y[i]**2)
        
        # Интегрирование координат
        height[i] = height[i-1] + velocity_y[i] * dt
        horizontal_range[i] = horizontal_range[i-1] + velocity_x[i] * dt
        
        if height[i] >= 300000:
            # Обрезка массивов
            time = time[:i+1]
            height = height[:i+1]
            velocity = velocity[:i+1]
            velocity_x = velocity_x[:i+1]
            velocity_y = velocity_y[:i+1]
            acceleration = acceleration[:i]
            mass = mass[:i+1]
            pressure = pressure[:i+1]
            o2_available = o2_available[:i+1]
            drag_force = drag_force[:i+1]
            horizontal_range = horizontal_range[:i+1]
            break
    
    # последние значения
    if len(time) > 0:
        h_km_last = height[-1] / 1000
        pressure[-1] = float(p_interp(h_km_last))
        rho_last = float(rho_interp(h_km_last))
        o2_frac_last = float(o2_interp(h_km_last))
        o2_available[-1] = rho_last * velocity[-1] * A * o2_frac_last
        drag_force[-1] = 0.5 * rho_last * velocity[-1]**2 * C_d * S_ref
    
    return time, height, velocity, acceleration, mass, pressure, o2_available, burn_time, drag_force, horizontal_range, velocity_x, velocity_y

# 4. РАСЧЁТы

time, height, velocity, acceleration, mass, pressure, o2_available, burn_time, drag_force, horizontal_range, velocity_x, velocity_y = calculate_smooth_trajectory()

height_km = height / 1000
time_min = time / 60
horizontal_range_km = horizontal_range / 1000

# Находим все точки пересечения кривых кислорода
intersection_points = []

for i in range(1, len(o2_available)):
    # Проверяем, пересекает ли кривая линию потребности
    if (o2_available[i-1] <= O2_FLOW_RATE and o2_available[i] >= O2_FLOW_RATE) or \
       (o2_available[i-1] >= O2_FLOW_RATE and o2_available[i] <= O2_FLOW_RATE):
        
        # Линейная интерполяция
        h1, h2 = height_km[i-1], height_km[i]
        o1, o2 = o2_available[i-1], o2_available[i]
        t1, t2 = time[i-1], time[i]
        d1, d2 = drag_force[i-1], drag_force[i]
        
        # Интерполяция параметров в точке пересечения
        intersection_height = h1 + (h2 - h1) * (O2_FLOW_RATE - o1) / (o2 - o1)
        intersection_time = t1 + (t2 - t1) * (O2_FLOW_RATE - o1) / (o2 - o1)
        intersection_drag = d1 + (d2 - d1) * (O2_FLOW_RATE - o1) / (o2 - o1)
        
        intersection_points.append({
            'height': intersection_height,
            'time': intersection_time,
            'drag': intersection_drag,
            'o2_available': O2_FLOW_RATE,
            'index': i
        })

# Если точек пересечения нет, находим ближайшую точку
if not intersection_points:
    diff = np.abs(o2_available - O2_FLOW_RATE)
    closest_idx = np.argmin(diff)
    intersection_points.append({
        'height': height_km[closest_idx],
        'time': time[closest_idx],
        'drag': drag_force[closest_idx],
        'o2_available': o2_available[closest_idx],
        'index': closest_idx
    })

# УДАЛЯЕМ ПЕРВУЮ ТОЧКУ ПЕРЕСЕЧЕНИЯ
if len(intersection_points) > 1:
    intersection_points = intersection_points[1:]  # Удаляем первую точку
elif len(intersection_points) == 1:
    # Если только одна точка, проверяем её высоту, если она очень маленькая удаляем
    if intersection_points[0]['height'] < 1.0:  # Если высота меньше 1 км
        intersection_points = []  # Удаляем все точки

# точка максимального сопротивления
max_drag_idx = np.argmax(drag_force)
max_drag_time = time[max_drag_idx]
max_drag_value = drag_force[max_drag_idx]
max_drag_height = height_km[max_drag_idx]

# время достижения 80 км
time_80km_idx = np.argmin(np.abs(height_km - 80))
time_80km = time[time_80km_idx]
drag_80km = drag_force[time_80km_idx] / 1000

# 5. ГРАФИКИ

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))

# График 1: Сопротивление от времени полета
ax1.plot(time, drag_force / 1000, 'b-', linewidth=2.5, alpha=0.8, label='Аэродинамическое сопротивление')
ax1.set_xlabel('Время полёта, с', fontsize=12)
ax1.set_ylabel('Сопротивление, кН', fontsize=12)
ax1.set_title('Зависимость аэродинамического сопротивления от времени полёта', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, time[-1])

# Точка максимального сопротивления
ax1.plot(max_drag_time, max_drag_value / 1000, 'ro', markersize=10, 
         label=f'Макс. сопр.: {max_drag_value/1000:.1f} кН')
ax1.axvline(x=max_drag_time, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Линия 80 км на графике сопротивления
ax1.axvline(x=time_80km, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='80 км')
ax1.plot(time_80km, drag_80km, 'o', color='orange', markersize=8, label=f'Сопр. на 80 км: {drag_80km:.1f} кН')

# Отмечаем оставшиеся точки пересечения на графике сопротивления
for i, point in enumerate(intersection_points):
    if point['time'] <= time[-1]:
        color = 'green'
        marker = 'o'
        label_text = f'Пересечение {i+1}' if len(intersection_points) > 1 else 'Пересечение'
        
        ax1.plot(point['time'], point['drag'] / 1000, color=color, marker=marker, 
                 markersize=10, label=f'{label_text}')
        ax1.axvline(x=point['time'], color=color, linestyle='--', alpha=0.5, linewidth=1)

ax1.legend(loc='upper right', fontsize=9)

# График 2: Высота от расхода кислорода
ax2.plot(height_km, np.ones_like(time) * O2_FLOW_RATE, 'r-', linewidth=2.5, 
         label=f'Потребность двигателя: {O2_FLOW_RATE:.1f} кг/с', alpha=0.8)

# Фильтруем данные для устранения артефакта на 0 км
valid_indices = (height_km > 0.01) & (o2_available > 0)
if np.any(valid_indices):
    ax2.plot(height_km[valid_indices], o2_available[valid_indices], 'b-', linewidth=2.5, 
             label='Доступный кислород в воздухе', alpha=0.8)
else:
    # Если все значения невалидны, используем все данные
    ax2.plot(height_km, o2_available, 'b-', linewidth=2.5, 
             label='Доступный кислород в воздухе', alpha=0.8)

# Используем логарифмическую шкалу
ax2.set_yscale('log')
ax2.set_ylim(1e-2, 1e4)

ax2.set_xlabel('Высота полёта, км', fontsize=12)
ax2.set_ylabel('Расход кислорода, кг/с', fontsize=12)
ax2.set_title('Зависимость расхода кислорода от высоты полёта', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, height_km[-1])

# Линия 80 км
ax2.axvline(x=80, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='80 км')


# Отмечаем оставшиеся точки пересечения на графике кислорода
for i, point in enumerate(intersection_points):
    if 0 <= point['height'] <= height_km[-1]:
        # Добавляем точку пересечения
        ax2.plot(point['height'], O2_FLOW_RATE, color='green', marker='o', markersize=10)
        
        # аннотация с параметрами
        annotation_text = f'Пересечение {i+1}:\nВысота: {point["height"]:.1f} км\nВремя: {point["time"]:.1f} с\nСопр.: {point["drag"]/1000:.1f} кН'
        
        # зависимости от позиции точки
        if point['height'] < height_km[-1] / 2:
            # Если точка в левой половине графика, размещаем справа от нее
            ax2.annotate(annotation_text, 
                        xy=(point['height'], O2_FLOW_RATE),
                        xytext=(10, 40 if i % 2 == 0 else -40),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
        else:
            # Если точка в правой половине графика, размещаем слева от нее
            ax2.annotate(annotation_text, 
                        xy=(point['height'], O2_FLOW_RATE),
                        xytext=(-200, 40 if i % 2 == 0 else -40),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
        
        # вертикальная линия на точке пересечения
        ax2.axvline(x=point['height'], color='green', linestyle='--', alpha=0.5, linewidth=1)

# легенда
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='r', lw=2, label=f'Потребность: {O2_FLOW_RATE} кг/с'),
    Line2D([0], [0], color='b', lw=2, label='Доступный кислород'),
    Line2D([0], [0], color='orange', linestyle='--', lw=1, label='80 км'),
    Line2D([0], [0], marker='o', color='green', lw=0, label='Пересечение', markersize=8)
]

ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)

# аэродинамическое давление q = 0.5 * ρ * V^2 = drag_force / (C_d * S_ref)
q_dyn = drag_force / (C_d * S_ref)  # в Па
q_dyn_kPa = q_dyn / 1000            # в килопа
ax3.plot(height_km, q_dyn_kPa, 'm-', linewidth=2.5, alpha=0.8, label='Аэродинамическое давление')
ax3.set_xlabel('Высота полёта, км', fontsize=12)
ax3.set_ylabel('Давление, кПа', fontsize=12)
ax3.set_title('Зависимость аэродинамического давления от высоты', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(0, height_km[-1])

# Точка на высоте 21.9 км (значение аэродинамического давления)
h_target_219 = 21.9
idx_219 = np.argmin(np.abs(height_km - h_target_219))
q_219 = q_dyn_kPa[idx_219]
ax3.plot(h_target_219, q_219, 'go', markersize=8, label=f'q на H=21.9 км: {q_219:.2f} кПа')
ax3.axvline(x=h_target_219, color='green', linestyle='--', alpha=0.5, linewidth=1)

# Линия 80 км
ax3.axvline(x=80, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='80 км')

ax3.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()


# 6. ВЫВОД РЕЗУЛЬТАТОВ


print(f"{'Начальная масса':<30} {MASS_INITIAL:<30.1f} {'кг':<20}")
print(f"{'Тяга двигателя':<30} {THRUST:<30.0f} {'Н':<20}")
print(f"{'Расход кислорода':<30} {O2_FLOW_RATE:<30.2f} {'кг/с':<20}")
print(f"{'Начальный угол тангажа':<30} {INITIAL_ANGLE:<30.1f} {'°':<20}")
print(f"{'Целевой угол тангажа':<30} {TARGET_ANGLE:<30.1f} {'°':<20}")
print(f"{'Скорость изменения угла':<30} {PITCH_RATE:<30.2f} {'°/с':<20}")
print(f"{'Начальная скорость':<30} {v0:<30.1f} {'м/с':<20}")
print(f"{'Площадь входного сечения':<30} {A:<30.3f} {'м²':<20}")
print(f"{'Характерная площадь':<30} {S_ref:<30.3f} {'м²':<20}")
print(f"{'Коэффициент сопротивления':<30} {C_d:<30.2f} {'-':<20}")
print(f"{'Начальное давление':<30} {p0:<30.0f} {'Па':<20}")
print(f"{'Время работы двигателя':<30} {burn_time:<30.1f} {'с':<20}")
print(f"{'Максимальная высота':<30} {height_km[-1]:<30.1f} {'км':<20}")
print(f"{'Максимальная скорость':<30} {np.max(velocity):<30.1f} {'м/с':<20}")
print(f"{'Горизонтальная дальность':<30} {horizontal_range_km[-1]:<30.1f} {'км':<20}")


print("ТОЧКИ ПЕРЕСЕЧЕНИЯ КРИВЫХ КИСЛОРОДА")


if intersection_points:
    print(f"{'№':<3} {'Высота, км':<12} {'Время, с':<12} {'Сопр., кН':<12} {'O₂, кг/с':<12}")
    
    for i, point in enumerate(intersection_points):
        print(f"{i+1:<3} {point['height']:<12.2f} {point['time']:<12.1f} {point['drag']/1000:<12.2f} {point['o2_available']:<12.2f}")
else:
    print("Точки пересечения не найдены")


print("ТОЧКА МАКСИМАЛЬНОГО СОПРОТИВЛЕНИЯ")

print(f"{'Параметр':<25} {'Значение':<20} {'Единица измерения':<15}")

print(f"{'Максимальное сопротивление':<25} {max_drag_value/1000:<20.2f} {'кН':<15}")
print(f"{'Время':<25} {max_drag_time:<20.1f} {'с':<15}")
print(f"{'Высота':<25} {max_drag_height:<20.2f} {'км':<15}")
print(f"{'Скорость в этой точке':<25} {velocity[max_drag_idx]:<20.1f} {'м/с':<15}")
print(f"{'Ускорение в этой точке':<25} {acceleration[max_drag_idx]:<20.3f} {'м/с²':<15}")


print("ПАРАМЕТРЫ НА ВЫСОТЕ 80 КМ")

print(f"{'Параметр':<25} {'Значение':<20} {'Единица измерения':<15}")

print(f"{'Время достижения 80 км':<25} {time_80km:<20.1f} {'с':<15}")
print(f"{'Сопротивление на 80 км':<25} {drag_80km:<20.2f} {'кН':<15}")
print(f"{'Скорость на 80 км':<25} {velocity[time_80km_idx]:<20.1f} {'м/с':<15}")
print(f"{'Доступный кислород на 80 км':<25} {o2_available[time_80km_idx]:<20.3f} {'кг/с':<15}")


print("КЛЮЧЕВЫЕ ПАРАМЕТРЫ ПО ВЫСОТАМ")

print(f"{'Высота, км':<12} {'Время, с':<12} {'Скорость, м/с':<15} {'Сопротивление, кН':<20} {'O₂ доступный, кг/с':<20}")

key_heights = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]

for h_target in key_heights:
    # ближайшая точка
    idx = np.argmin(np.abs(height_km - h_target))
    
    if idx < len(height):
        h_actual = height_km[idx]
        t_val = time[idx]
        v_val = velocity[idx]
        drag_val = drag_force[idx] / 1000 if idx < len(drag_force) else 0
        o2_avail = o2_available[idx] if idx < len(o2_available) else 0
        
        # вывод
        print(f"{h_actual:<12.1f} {t_val:<12.1f} {v_val:<15.1f} {drag_val:<20.3f} {o2_avail:<20.3f}")
