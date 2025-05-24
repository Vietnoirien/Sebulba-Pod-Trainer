import sys
import math

# Variables globales pour suivre l'historique des checkpoints
checkpoint_history = []
checkpoint_count = 0
lap_completed = False
current_checkpoint_index = 0
last_checkpoint_x = None
last_checkpoint_y = None

# Variables pour la vitesse et la dérive
last_x = None
last_y = None
velocity_x = 0
velocity_y = 0

# Variable pour le boost et le shield
boost_used = False
shield_used = False
shield_cooldown = 0

def control(next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle, x, y, opponent_x, opponent_y):
    global checkpoint_history, checkpoint_count, lap_completed, current_checkpoint_index, last_checkpoint_x, last_checkpoint_y
    global last_x, last_y, velocity_x, velocity_y, boost_used, shield_used, shield_cooldown
    
    # Gestion du cooldown du bouclier
    if shield_cooldown > 0:
        shield_cooldown -= 1
    
    # Calcul de la vitesse actuelle
    if last_x is not None and last_y is not None:
        velocity_x = x - last_x
        velocity_y = y - last_y
    
    last_x = x
    last_y = y
    
    # Détection de changement de checkpoint
    checkpoint_changed = (last_checkpoint_x != next_checkpoint_x or last_checkpoint_y != next_checkpoint_y)
    
    # Enregistrement des checkpoints pendant le premier tour
    if not lap_completed and checkpoint_changed:
        # Vérifier si ce checkpoint est nouveau ou si nous avons bouclé
        is_new_checkpoint = True
        for cp in checkpoint_history:
            if abs(cp[0] - next_checkpoint_x) < 100 and abs(cp[1] - next_checkpoint_y) < 100:
                is_new_checkpoint = False
                break
        
        if is_new_checkpoint:
            checkpoint_history.append((next_checkpoint_x, next_checkpoint_y))
            checkpoint_count += 1
            print(f"Nouveau checkpoint enregistré: {next_checkpoint_x}, {next_checkpoint_y}", file=sys.stderr, flush=True)
        elif len(checkpoint_history) > 1 and checkpoint_count > 0:
            # Si nous revenons au premier checkpoint, le tour est terminé
            if abs(checkpoint_history[0][0] - next_checkpoint_x) < 100 and abs(checkpoint_history[0][1] - next_checkpoint_y) < 100:
                lap_completed = True
                print(f"Premier tour terminé! {len(checkpoint_history)} checkpoints enregistrés.", file=sys.stderr, flush=True)
    
    # Mettre à jour les dernières coordonnées de checkpoint
    last_checkpoint_x = next_checkpoint_x
    last_checkpoint_y = next_checkpoint_y
    
    # Optimisations basées sur l'historique des checkpoints
    if lap_completed:
        current_checkpoint_index = find_current_checkpoint_index(next_checkpoint_x, next_checkpoint_y)
        next_checkpoint_index = (current_checkpoint_index + 1) % len(checkpoint_history)
        
        # Anticipation du prochain checkpoint
        next_next_checkpoint_x, next_next_checkpoint_y = checkpoint_history[next_checkpoint_index]
        
        # Ajustement de la cible pour couper les virages
        if next_checkpoint_dist < 2000:
            # Calcul d'un point intermédiaire entre le checkpoint actuel et le suivant
            target_x = int(next_checkpoint_x * 0.8 + next_next_checkpoint_x * 0.2)
            target_y = int(next_checkpoint_y * 0.8 + next_next_checkpoint_y * 0.2)
        else:
            target_x = next_checkpoint_x
            target_y = next_checkpoint_y
    else:
        target_x = next_checkpoint_x
        target_y = next_checkpoint_y
    
    # Compensation de la dérive
    # Prédire où le pod sera dans le futur en fonction de sa vitesse actuelle
    # Ajuster la compensation en fonction de la distance au checkpoint
    prediction_factor = min(3.0, max(0.5, next_checkpoint_dist / 1000))

    target_x = int(target_x - velocity_x * prediction_factor)
    target_y = int(target_y - velocity_y * prediction_factor)
    
    # Logique de contrôle de la puissance
    next_checkpoint_angle_abs = abs(next_checkpoint_angle)
    
    # Ajustement de la puissance en fonction de l'angle et de la distance
    if next_checkpoint_angle_abs < 30 and next_checkpoint_dist > 1500:
        thrust = 100
    elif next_checkpoint_angle_abs < 60:
        thrust = 90
    elif next_checkpoint_angle_abs < 90:
        thrust = 70
    else:
        thrust = 15
    
    # Utilisation du boost de manière stratégique
    if not boost_used and lap_completed and next_checkpoint_angle_abs < 5 and next_checkpoint_dist > 4000:
        # Utiliser le boost sur la plus longue ligne droite
        longest_distance = 0
        for i in range(len(checkpoint_history)):
            cp1 = checkpoint_history[i]
            cp2 = checkpoint_history[(i + 1) % len(checkpoint_history)]
            dist = math.sqrt((cp2[0] - cp1[0])**2 + (cp2[1] - cp1[1])**2)
            if dist > longest_distance:
                longest_distance = dist
            
            if current_checkpoint_index == i and next_checkpoint_dist > longest_distance * 0.7:
                thrust = "BOOST"
                boost_used = True
    elif not boost_used and next_checkpoint_angle_abs < 5 and next_checkpoint_dist > 3500:
        thrust = "BOOST"
        boost_used = True
    
    # Gestion des collisions avec l'adversaire
    distance_to_opponent = math.sqrt((opponent_x - x)**2 + (opponent_y - y)**2)
    
    # Si l'adversaire est proche et devant nous (entre nous et le checkpoint)
    opponent_to_checkpoint = math.sqrt((opponent_x - next_checkpoint_x)**2 + (opponent_y - next_checkpoint_y)**2)
    
    # Utilisation du bouclier (SHIELD)
    # Conditions pour utiliser le bouclier:
    # 1. Pas de cooldown actif
    # 2. L'adversaire est très proche
    # 3. Nous avons une bonne vitesse (pour maximiser l'impact)
    # 4. Nous ne sommes pas en train de faire un virage serré (pour ne pas perdre trop de temps)
    pod_speed = math.sqrt(velocity_x**2 + velocity_y**2)
    
    if distance_to_opponent < 850 and pod_speed > 350 and next_checkpoint_angle_abs < 90:
        # Calculer l'angle entre notre direction et l'adversaire
        angle_to_opponent = math.degrees(math.atan2(opponent_y - y, opponent_x - x))
        pod_angle = math.degrees(math.atan2(target_y - y, target_x - x))
        relative_angle = (angle_to_opponent - pod_angle) % 360
        
        # Si l'adversaire est dans notre ligne de mire et assez proche, on active le bouclier
        if abs(relative_angle) < 30 and distance_to_opponent < 600:
            print(f"Activation du SHIELD pour collision avec l'adversaire!", file=sys.stderr, flush=True)
            target_x = opponent_x
            target_y = opponent_y
            thrust = "SHIELD"
    
    # Si l'adversaire est proche mais que nous n'utilisons pas le bouclier
    elif distance_to_opponent < 850 and opponent_to_checkpoint < next_checkpoint_dist:
        # Calculer l'angle entre notre direction et l'adversaire
        angle_to_opponent = math.degrees(math.atan2(opponent_y - y, opponent_x - x))
        pod_angle = math.degrees(math.atan2(target_y - y, target_x - x))
        relative_angle = (angle_to_opponent - pod_angle) % 360
        
        # Si l'adversaire est dans notre ligne de mire, on peut essayer de le percuter
        if abs(relative_angle) < 30 and distance_to_opponent < 600:
            print(f"Tentative de collision avec l'adversaire!", file=sys.stderr, flush=True)
            target_x = opponent_x
            target_y = opponent_y
            thrust = 100  # Accélération maximale pour la collision
        # Sinon, on essaie de l'éviter
        elif distance_to_opponent < 400:
            # Déplacer notre cible pour éviter l'adversaire
            offset_x = y - opponent_y
            offset_y = opponent_x - x
            magnitude = math.sqrt(offset_x**2 + offset_y**2)
            if magnitude > 0:
                offset_x = offset_x / magnitude * 300
                offset_y = offset_y / magnitude * 300
                target_x = int(next_checkpoint_x + offset_x)
                target_y = int(next_checkpoint_y + offset_y)

    return str(target_x) + " " + str(target_y) + " " + str(thrust)

def find_current_checkpoint_index(checkpoint_x, checkpoint_y):
    """Trouve l'index du checkpoint actuel dans l'historique"""
    for i, (x, y) in enumerate(checkpoint_history):
        if abs(x - checkpoint_x) < 100 and abs(y - checkpoint_y) < 100:
            return i
    return 0

# game loop
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]

    # Debug information
    if checkpoint_history:
        print(f"Checkpoints: {checkpoint_history}, Current: {current_checkpoint_index}", file=sys.stderr, flush=True)
    print(f"Distance au checkpoint: {next_checkpoint_dist}, Angle: {next_checkpoint_angle}", file=sys.stderr, flush=True)
    if last_x is not None:
        print(f"Vitesse: ({velocity_x}, {velocity_y}), magnitude: {math.sqrt(velocity_x**2 + velocity_y**2)}", file=sys.stderr, flush=True)
    if shield_cooldown > 0:
        print(f"Bouclier en cooldown: {shield_cooldown} tours restants", file=sys.stderr, flush=True)

    # You have to output the target position
    # followed by the power (0 <= thrust <= 100)
    # i.e.: "x y thrust"
    print(control(next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle, x, y, opponent_x, opponent_y))
