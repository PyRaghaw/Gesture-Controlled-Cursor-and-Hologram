import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import os
from collections import deque
import json
from datetime import datetime

class HologramController:
    def __init__(self):
        # MediaPipe with HIGH accuracy
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 2 hands for zoom gesture
            model_complexity=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480
        
        # Tracking
        self.cursor_history = deque(maxlen=5)
        self.gesture_history = deque(maxlen=6)
        self.finger_history = deque(maxlen=4)
        
        self.prev_hand_x, self.prev_hand_y = None, None
        self.prev_hand_pos = None
        self.prev_wrist = None
        self.click_cooldown = 0
        self.cursor_speed = 3.2
        
        # Modes
        self.current_mode = 'normal'
        self.drag_active = False
        
        # Drawing mode
        self.drawing_canvas = None
        self.drawing_color = (0, 255, 0)
        self.drawing_thickness = 3
        self.drawing_points = []
        
        # Gaming mode
        self.gaming_keys = {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}
        
        # HOLOGRAM MODE - IRON MAN STYLE! 🦾
        self.available_objects = {
            'cube': self.generate_cube(),
            'sphere': self.generate_sphere(),
            'pyramid': self.generate_pyramid(),
            'torus': self.generate_torus(),
            'cylinder': self.generate_cylinder(),
            'diamond': self.generate_diamond(),
            'helix': self.generate_helix(),
            'star': self.generate_star()
        }
        
        # MULTIPLE SPAWNED OBJECTS!
        self.spawned_objects = []  # List of {obj, rotation, position, scale, id, selected}
        self.next_object_id = 0
        
        self.selected_object_id = None
        self.dragging_object_id = None
        
        self.menu_items = []
        self.init_menu()
        
        # DELETE ZONE (Trash bin area)
        self.delete_zone = {
            'x': self.cam_width - 100,
            'y': 50,
            'size': 60,
            'active': False
        }
        
        # Colors
        self.hologram_color = (0, 200, 255)  # Cyan
        self.glow_color = (100, 230, 255)
        self.selected_color = (0, 255, 100)  # Green when selected
        self.delete_color = (0, 0, 255)  # Red for delete zone
        
        # Zoom & Gestures
        self.pinch_threshold = 35
        self.two_hand_prev_distance = None
        self.zoom_cooldown = 0
        
        # Stats & Analytics
        self.total_gestures = 0
        self.gesture_counts = {}
        self.session_start = time.time()
        self.cursor_distance = 0
        self.click_positions = []
        
        self.show_notifications = True
        
        pyautogui.FAILSAFE = False
        
        self.init_drawing_canvas()
    
    def init_drawing_canvas(self):
        self.drawing_canvas = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
    
    # ============ 3D OBJECT GENERATORS ============
    def generate_cube(self):
        size = 1
        vertices = [
            [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
            [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        return {'vertices': vertices, 'edges': edges, 'name': 'CUBE'}
    
    def generate_sphere(self):
        vertices, edges = [], []
        radius, rings, segments = 1, 10, 14
        
        for i in range(rings):
            theta = (i * math.pi) / rings
            for j in range(segments):
                phi = (j * 2 * math.pi) / segments
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)
                vertices.append([x, y, z])
        
        for i in range(rings - 1):
            for j in range(segments):
                current = i * segments + j
                next_ring = (i + 1) * segments + j
                next_segment = i * segments + (j + 1) % segments
                edges.append((current, next_ring))
                edges.append((current, next_segment))
        
        return {'vertices': vertices, 'edges': edges, 'name': 'SPHERE'}
    
    def generate_pyramid(self):
        vertices = [[0,-1,0], [-1,1,-1], [1,1,-1], [1,1,1], [-1,1,1]]
        edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
        return {'vertices': vertices, 'edges': edges, 'name': 'PYRAMID'}
    
    def generate_torus(self):
        vertices, edges = [], []
        R, r, segments, sides = 1, 0.4, 14, 10
        
        for i in range(segments):
            theta = (i * 2 * math.pi) / segments
            for j in range(sides):
                phi = (j * 2 * math.pi) / sides
                x = (R + r * math.cos(phi)) * math.cos(theta)
                y = (R + r * math.cos(phi)) * math.sin(theta)
                z = r * math.sin(phi)
                vertices.append([x, y, z])
        
        for i in range(segments):
            for j in range(sides):
                current = i * sides + j
                next_seg = ((i + 1) % segments) * sides + j
                next_side = i * sides + (j + 1) % sides
                edges.append((current, next_seg))
                edges.append((current, next_side))
        
        return {'vertices': vertices, 'edges': edges, 'name': 'TORUS'}
    
    def generate_cylinder(self):
        vertices, edges = [], []
        radius, height, segments = 1, 2, 14
        
        for i in range(segments):
            angle = (i * 2 * math.pi) / segments
            x, z = radius * math.cos(angle), radius * math.sin(angle)
            vertices.extend([[x, -height/2, z], [x, height/2, z]])
        
        for i in range(segments):
            bottom, top = i * 2, i * 2 + 1
            next_bottom, next_top = ((i + 1) % segments) * 2, ((i + 1) % segments) * 2 + 1
            edges.extend([(bottom, top), (bottom, next_bottom), (top, next_top)])
        
        return {'vertices': vertices, 'edges': edges, 'name': 'CYLINDER'}
    
    def generate_diamond(self):
        vertices = [
            [0, 1.5, 0],  # Top
            [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1],  # Middle
            [0, -1.5, 0]  # Bottom
        ]
        edges = [
            (0,1),(0,2),(0,3),(0,4),  # Top to middle
            (1,2),(2,3),(3,4),(4,1),  # Middle ring
            (5,1),(5,2),(5,3),(5,4)   # Bottom to middle
        ]
        return {'vertices': vertices, 'edges': edges, 'name': 'DIAMOND'}
    
    def generate_helix(self):
        vertices, edges = [], []
        turns, points_per_turn = 3, 12
        total_points = turns * points_per_turn
        
        for i in range(total_points):
            t = (i / points_per_turn) * 2 * math.pi
            x = math.cos(t)
            y = (i / total_points) * 3 - 1.5
            z = math.sin(t)
            vertices.append([x, y, z])
        
        for i in range(total_points - 1):
            edges.append((i, i + 1))
        
        return {'vertices': vertices, 'edges': edges, 'name': 'HELIX'}
    
    def generate_star(self):
        vertices, edges = [], []
        points = 5
        outer_radius, inner_radius = 1.2, 0.5
        
        for i in range(points * 2):
            angle = (i * math.pi) / points
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append([x, 0, z])
        
        for i in range(points * 2):
            edges.append((i, (i + 1) % (points * 2)))
        
        # Add center connections for 3D effect
        vertices.append([0, 1, 0])  # Top center
        vertices.append([0, -1, 0])  # Bottom center
        top_idx = len(vertices) - 2
        bottom_idx = len(vertices) - 1
        
        for i in range(0, points * 2, 2):
            edges.append((i, top_idx))
            edges.append((i, bottom_idx))
        
        return {'vertices': vertices, 'edges': edges, 'name': 'STAR'}
    
    def init_menu(self):
        y_start, spacing = 80, 70
        for idx, (name, obj) in enumerate(self.available_objects.items()):
            self.menu_items.append({
                'name': name,
                'object': obj,
                'position': (100, y_start + idx * spacing),
                'size': 35,
                'selected': False
            })
    
    # ============ 3D RENDERING ============
    def rotate_point_3d(self, point, angles):
        x, y, z = point
        rx, ry, rz = [math.radians(a) for a in angles]
        
        # X rotation
        y_rot = y * math.cos(rx) - z * math.sin(rx)
        z_rot = y * math.sin(rx) + z * math.cos(rx)
        y, z = y_rot, z_rot
        
        # Y rotation
        x_rot = x * math.cos(ry) + z * math.sin(ry)
        z_rot = -x * math.sin(ry) + z * math.cos(ry)
        x, z = x_rot, z_rot
        
        return [x, y, z]
    
    def project_3d_to_2d(self, point_3d, rotation, position, scale):
        rotated = self.rotate_point_3d(point_3d, rotation)
        x, y, z = [coord * scale for coord in rotated]
        
        perspective = 500
        scale_factor = perspective / (perspective + z)
        
        x_2d = int(x * scale_factor + position[0])
        y_2d = int(y * scale_factor + position[1])
        
        return (x_2d, y_2d, scale_factor)
    
    def draw_3d_object(self, frame, obj_data):
        obj = obj_data['obj']
        rotation = obj_data['rotation']
        position = obj_data['position']
        scale = obj_data['scale']
        is_selected = obj_data.get('selected', False)
        
        if not obj:
            return
        
        vertices, edges = obj['vertices'], obj['edges']
        projected = [self.project_3d_to_2d(v, rotation, position, scale) for v in vertices]
        
        # Choose color based on selection
        color = self.selected_color if is_selected else self.hologram_color
        
        # Draw edges with glow effect
        for edge in edges:
            p1_idx, p2_idx = edge
            if p1_idx < len(projected) and p2_idx < len(projected):
                p1, p2 = projected[p1_idx], projected[p2_idx]
                
                if (0 <= p1[0] < self.cam_width and 0 <= p1[1] < self.cam_height and
                    0 <= p2[0] < self.cam_width and 0 <= p2[1] < self.cam_height):
                    # Glow effect
                    cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), (50, 100, 150), 4)
                    # Main line
                    cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), color, 2)
        
        # Draw vertices
        for proj in projected:
            if 0 <= proj[0] < self.cam_width and 0 <= proj[1] < self.cam_height:
                cv2.circle(frame, (proj[0], proj[1]), 4, self.glow_color, -1)
        
        # Draw bounding box if selected
        if is_selected:
            xs = [p[0] for p in projected if 0 <= p[0] < self.cam_width]
            ys = [p[1] for p in projected if 0 <= p[1] < self.cam_height]
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                cv2.rectangle(frame, (min_x-10, min_y-10), (max_x+10, max_y+10), 
                            self.selected_color, 2)
    
    def draw_menu(self, frame):
        # Semi-transparent menu background
        menu_bg = frame.copy()
        cv2.rectangle(menu_bg, (0, 0), (220, self.cam_height), (20, 20, 30), -1)
        cv2.addWeighted(menu_bg, 0.7, frame, 0.3, 0, frame)
        
        # Menu title
        cv2.putText(frame, "3D MODELS", (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.hologram_color, 2)
        cv2.line(frame, (20, 50), (200, 50), self.hologram_color, 2)
        
        # Draw menu items
        for item in self.menu_items:
            pos, size = item['position'], item['size']
            
            color = self.hologram_color if item['selected'] else (0, 100, 150)
            thickness = 3 if item['selected'] else 2
            
            # Item box
            cv2.rectangle(frame, (pos[0] - size, pos[1] - size), 
                        (pos[0] + size, pos[1] + size), color, thickness)
            
            # Mini preview
            mini_rotation = [time.time() * 30 % 360, time.time() * 20 % 360, 0]
            mini_obj_data = {
                'obj': item['object'],
                'rotation': mini_rotation,
                'position': pos,
                'scale': 20,
                'selected': False
            }
            self.draw_3d_object(frame, mini_obj_data)
            
            # Name
            cv2.putText(frame, item['name'].upper(), (pos[0] - 35, pos[1] + size + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.hologram_color, 1)
    
    def draw_delete_zone(self, frame):
        dz = self.delete_zone
        color = (0, 0, 255) if dz['active'] else (100, 100, 100)
        
        # Trash bin icon
        cv2.rectangle(frame, 
                     (dz['x'] - dz['size']//2, dz['y'] - dz['size']//2),
                     (dz['x'] + dz['size']//2, dz['y'] + dz['size']//2),
                     color, 3)
        
        # Trash can lines
        cv2.line(frame, 
                (dz['x'] - 20, dz['y'] - 20),
                (dz['x'] + 20, dz['y'] - 20),
                color, 2)
        cv2.line(frame, 
                (dz['x'], dz['y'] - 10),
                (dz['x'], dz['y'] + 10),
                color, 2)
        
        if dz['active']:
            cv2.putText(frame, "DELETE", (dz['x'] - 35, dz['y'] + dz['size']//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ============ GESTURE DETECTION ============
    def check_menu_selection(self, hand_pos):
        for item in self.menu_items:
            pos, size = item['position'], item['size']
            if (pos[0] - size < hand_pos[0] < pos[0] + size and
                pos[1] - size < hand_pos[1] < pos[1] + size):
                return item
        return None
    
    def check_object_selection(self, hand_pos):
        # Check which spawned object is being touched
        for obj_data in reversed(self.spawned_objects):  # Check from top (last drawn)
            pos = obj_data['position']
            scale = obj_data['scale']
            
            # Simple circular hit detection
            dist = math.sqrt((hand_pos[0] - pos[0])**2 + (hand_pos[1] - pos[1])**2)
            if dist < scale * 1.5:  # Hit zone
                return obj_data['id']
        return None
    
    def check_delete_zone(self, pos):
        dz = self.delete_zone
        dist = math.sqrt((pos[0] - dz['x'])**2 + (pos[1] - dz['y'])**2)
        return dist < dz['size']
    
    def get_two_hand_distance(self, lm_list_1, lm_list_2):
        # Distance between two index fingers (for zoom gesture)
        p1 = lm_list_1[8]  # Right hand index
        p2 = lm_list_2[8]  # Left hand index
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # ============ ORIGINAL GESTURE FUNCTIONS ============
    def save_stats(self):
        stats = {
            'session_date': datetime.now().isoformat(),
            'duration': int(time.time() - self.session_start),
            'total_gestures': self.total_gestures,
            'gesture_breakdown': self.gesture_counts,
            'cursor_distance': round(self.cursor_distance, 2),
            'clicks': len(self.click_positions),
            'objects_spawned': len(self.spawned_objects)
        }
        with open(f'session_{int(time.time())}.json', 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"📊 Stats saved!")
    
    def notify(self, title, message):
        if self.show_notifications:
            try:
                os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
            except:
                pass
    
    def count_fingers(self, lm_list):
        fingers = []
        tips = [4, 8, 12, 16, 20]
        
        # Thumb
        if lm_list[tips[0]][0] < lm_list[tips[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for id in range(1, 5):
            if lm_list[tips[id]][1] < lm_list[tips[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_landmarks(self, hand_landmarks):
        lm_list = []
        for lm in hand_landmarks.landmark:
            cx = int(lm.x * self.cam_width)
            cy = int(lm.y * self.cam_height)
            lm_list.append([cx, cy])
        return lm_list
    
    def distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def detect_pinch(self, lm_list):
        return self.distance(lm_list[4], lm_list[8]) < self.pinch_threshold
    
    def detect_open_palm(self, fingers):
        return fingers.count(1) == 5
    
    def detect_swipe(self, lm_list):
        wrist = lm_list[0]
        
        if self.prev_wrist is None:
            self.prev_wrist = wrist
            return None
        
        delta_x = wrist[0] - self.prev_wrist[0]
        delta_y = wrist[1] - self.prev_wrist[1]
        
        self.prev_wrist = wrist
        
        threshold = 70
        if abs(delta_x) > threshold:
            return 'swipe_right' if delta_x > 0 else 'swipe_left'
        if abs(delta_y) > threshold:
            return 'swipe_down' if delta_y > 0 else 'swipe_up'
        
        return None
    
    def move_cursor(self, x, y):
        if self.prev_hand_x is None:
            self.prev_hand_x = x
            self.prev_hand_y = y
            return
        
        delta_x = (x - self.prev_hand_x) * self.cursor_speed
        delta_y = (y - self.prev_hand_y) * self.cursor_speed
        
        self.cursor_distance += abs(delta_x) + abs(delta_y)
        
        curr_x, curr_y = pyautogui.position()
        new_x = max(0, min(self.screen_width - 1, curr_x + delta_x))
        new_y = max(0, min(self.screen_height - 1, curr_y + delta_y))
        
        pyautogui.moveTo(new_x, new_y, _pause=False)
        
        self.prev_hand_x = x
        self.prev_hand_y = y
    
    def draw_on_canvas(self, pos):
        if len(self.drawing_points) > 0:
            cv2.line(self.drawing_canvas, self.drawing_points[-1], pos, 
                    self.drawing_color, self.drawing_thickness)
        self.drawing_points.append(pos)
    
    def detect_gesture_advanced(self, lm_lists, all_fingers):
        # lm_lists is a list of landmark lists (for multiple hands)
        # all_fingers is a list of finger counts for each hand
        
        if len(lm_lists) == 0:
            return 'none', None
        
        lm_list = lm_lists[0]
        fingers = all_fingers[0]
        finger_count = fingers.count(1)
        swipe = self.detect_swipe(lm_list)
        
        gesture = 'none'
        pos = None
        
        # HOLOGRAM MODE GESTURES
        if self.current_mode == 'hologram':
            # TWO HAND ZOOM GESTURE
            if len(lm_lists) == 2:
                gesture = 'hologram_zoom'
                return gesture, None
            
            # Single hand gestures - PRIORITY ORDER MATTERS!
            # Check pinch first (most specific)
            if self.detect_pinch(lm_list):
                gesture = 'hologram_pinch'
                pos = lm_list[8]
            # Check if ONLY index finger is up (for dragging)
            elif finger_count == 1 and fingers[1] == 1:
                gesture = 'hologram_drag'
                pos = lm_list[8]
            # Check open palm (all 5 fingers for rotation)
            elif self.detect_open_palm(fingers):
                gesture = 'hologram_rotate'
                pos = lm_list[8]
            # Check 2 fingers (peace sign - can also drag)
            elif finger_count == 2 and fingers[1] == 1 and fingers[2] == 1:
                gesture = 'hologram_drag'
                pos = lm_list[8]
            # Fallback to tracking
            else:
                gesture = 'hologram_tracking'
                pos = lm_list[8]
            
            # Debug print to see what's detected
            if gesture != 'hologram_tracking':
                print(f"🎯 Gesture: {gesture}, Fingers: {finger_count}, Pos: {pos}")
            
            return gesture, pos
        
        # NORMAL MODE GESTURES (original)
        if finger_count == 0:
            gesture = 'double_click'
        elif finger_count == 1 and fingers[1] == 1:
            if self.current_mode == 'drawing':
                gesture = 'draw'
                pos = lm_list[8]
            else:
                gesture = 'cursor'
                pos = lm_list[8]
        elif finger_count == 2:
            if fingers[0] == 1 and fingers[1] == 1:
                dist = self.distance(lm_list[4], lm_list[8])
                if dist < 40:
                    gesture = 'left_click'
                    self.click_positions.append(pyautogui.position())
                else:
                    gesture = 'cursor'
                    pos = lm_list[8]
            elif fingers[1] == 1 and fingers[2] == 1:
                gesture = 'right_click'
        elif finger_count == 3:
            if swipe == 'swipe_left':
                gesture = 'three_swipe_left'
            elif swipe == 'swipe_right':
                gesture = 'three_swipe_right'
            elif lm_list[8][1] < self.cam_height * 0.3:
                gesture = 'scroll_up'
            elif lm_list[8][1] > self.cam_height * 0.6:
                gesture = 'scroll_down'
            else:
                gesture = 'screenshot'
        elif finger_count == 4:
            if swipe == 'swipe_left':
                gesture = 'four_swipe_left'
            elif swipe == 'swipe_right':
                gesture = 'four_swipe_right'
            elif swipe == 'swipe_up':
                gesture = 'brightness_up'
            elif swipe == 'swipe_down':
                gesture = 'brightness_down'
            elif lm_list[8][1] < self.cam_height * 0.3:
                gesture = 'volume_up'
            elif lm_list[8][1] > self.cam_height * 0.6:
                gesture = 'volume_down'
            else:
                gesture = 'minimize_window'
        elif finger_count == 5:
            if swipe == 'swipe_left':
                gesture = 'five_swipe_left'
            elif swipe == 'swipe_right':
                gesture = 'five_swipe_right'
            elif swipe == 'swipe_up':
                gesture = 'show_desktop'
            elif swipe == 'swipe_down':
                gesture = 'mission_control'
            else:
                gesture = 'play_pause'
        
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > 6:
            self.gesture_history.popleft()
        
        if len(self.gesture_history) >= 3:
            recent = list(self.gesture_history)[-3:]
            if recent.count(recent[0]) >= 2:
                return recent[0], pos
        
        return gesture, pos
    
    def log_gesture(self, gesture):
        if gesture not in ['none', 'cursor', 'draw', 'hologram_tracking', 'hologram_drag', 'hologram_rotate']:
            self.total_gestures += 1
            self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
    
    def execute_all(self, gesture, pos, lm_lists):
        """Execute gesture actions."""
        t = time.time()

        # ========== HOLOGRAM GESTURES ==========
        if gesture == 'hologram_zoom' and len(lm_lists) == 2:
            # TWO HAND ZOOM!
            current_distance = self.get_two_hand_distance(lm_lists[0], lm_lists[1])

            if self.two_hand_prev_distance is not None and t - self.zoom_cooldown > 0.05:
                delta = current_distance - self.two_hand_prev_distance

                # Zoom selected object or all objects
                if self.selected_object_id is not None:
                    for obj_data in self.spawned_objects:
                        if obj_data['id'] == self.selected_object_id:
                            obj_data['scale'] = max(20, min(200, obj_data['scale'] + delta * 0.5))
                            break
                else:
                    # Zoom all objects
                    for obj_data in self.spawned_objects:
                        obj_data['scale'] = max(20, min(200, obj_data['scale'] + delta * 0.3))

                self.zoom_cooldown = t

            self.two_hand_prev_distance = current_distance
            return
        else:
            self.two_hand_prev_distance = None

        if gesture == 'hologram_pinch' and pos:
            # SELECT FROM MENU OR SELECT SPAWNED OBJECT
            menu_item = self.check_menu_selection(pos)
            if menu_item:
                # Spawn new object!
                for item in self.menu_items:
                    item['selected'] = False
                menu_item['selected'] = True

                new_obj = {
                    'obj': menu_item['object'],
                    'rotation': [0, 0, 0],
                    'position': [350, 240, 0],
                    'scale': 80,
                    'id': self.next_object_id,
                    'selected': True
                }
                self.spawned_objects.append(new_obj)
                self.selected_object_id = self.next_object_id
                self.next_object_id += 1
                self.log_gesture(gesture)
                print(f"✨ Spawned: {menu_item['name'].upper()} (ID: {new_obj['id']})")
                return

            # Select existing object
            obj_id = self.check_object_selection(pos)
            if obj_id is not None:
                # Deselect all, select this one
                for obj_data in self.spawned_objects:
                    obj_data['selected'] = (obj_data['id'] == obj_id)
                self.selected_object_id = obj_id
                print(f"🎯 Selected Object ID: {obj_id}")
                return
        
        if gesture == 'hologram_drag' and pos:
            # DRAG SELECTED OBJECT
            if self.selected_object_id is not None:
                # Check if in delete zone
                if self.check_delete_zone(pos):
                    self.delete_zone['active'] = True
                    # Delete object
                    self.spawned_objects = [obj for obj in self.spawned_objects 
                                           if obj['id'] != self.selected_object_id]
                    self.log_gesture('hologram_delete')
                    print(f"🗑️ Deleted Object ID: {self.selected_object_id}")
                    self.selected_object_id = None
                    self.delete_zone['active'] = False
                    return
                
                self.delete_zone['active'] = False
                
                # Move object
                for obj_data in self.spawned_objects:
                    if obj_data['id'] == self.selected_object_id:
                        obj_data['position'][0] = pos[0]
                        obj_data['position'][1] = pos[1]
                        break
            else:
                # If no object selected, try to select one
                obj_id = self.check_object_selection(pos)
                if obj_id is not None:
                    for obj_data in self.spawned_objects:
                        obj_data['selected'] = (obj_data['id'] == obj_id)
                    self.selected_object_id = obj_id
            return
        
        if gesture == 'hologram_rotate' and pos:
            # ROTATE SELECTED OBJECT (or all if none selected)
            if self.prev_hand_pos:
                delta_x = pos[0] - self.prev_hand_pos[0]
                delta_y = pos[1] - self.prev_hand_pos[1]
                
                if self.selected_object_id is not None:
                    for obj_data in self.spawned_objects:
                        if obj_data['id'] == self.selected_object_id:
                            obj_data['rotation'][1] += delta_x * 0.7
                            obj_data['rotation'][0] += delta_y * 0.7
                            break
                else:
                    # Rotate all objects
                    for obj_data in self.spawned_objects:
                        obj_data['rotation'][1] += delta_x * 0.5
                        obj_data['rotation'][0] += delta_y * 0.5
            return
        
        # ========== ORIGINAL GESTURES ==========
        if gesture == 'cursor' and pos:
            self.move_cursor(pos[0], pos[1])
        elif gesture == 'draw' and pos:
            self.draw_on_canvas(pos)
        elif gesture == 'left_click':
            if t - self.click_cooldown > 0.4:
                pyautogui.click()
                self.click_cooldown = t
                self.prev_hand_x = None
                self.log_gesture(gesture)
                print("✓ LEFT CLICK")
        elif gesture == 'right_click':
            if t - self.click_cooldown > 0.4:
                pyautogui.rightClick()
                self.click_cooldown = t
                self.prev_hand_x = None
                self.log_gesture(gesture)
                print("✓ RIGHT CLICK")
        elif gesture == 'double_click':
            if t - self.click_cooldown > 0.7:
                pyautogui.doubleClick()
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("✓✓ DOUBLE CLICK")
        elif gesture == 'scroll_up':
            pyautogui.scroll(12)
        elif gesture == 'scroll_down':
            pyautogui.scroll(-12)
        elif gesture == 'volume_up':
            if t - self.click_cooldown > 0.2:
                os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 6)'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🔊 VOL+")
        elif gesture == 'volume_down':
            if t - self.click_cooldown > 0.2:
                try:
                    vol = os.popen("osascript -e 'output volume of (get volume settings)'").read().strip()
                    new_vol = max(0, int(vol) - 6)
                    os.system(f"osascript -e 'set volume output volume {new_vol}'")
                    self.click_cooldown = t
                    self.log_gesture(gesture)
                    print("🔉 VOL-")
                except:
                    pass
        elif gesture == 'brightness_up':
            if t - self.click_cooldown > 0.3:
                os.system("osascript -e 'tell application \"System Events\" to key code 144'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("☀️ BRIGHT+")
        elif gesture == 'brightness_down':
            if t - self.click_cooldown > 0.3:
                os.system("osascript -e 'tell application \"System Events\" to key code 145'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🌙 BRIGHT-")
        elif gesture == 'screenshot':
            if t - self.click_cooldown > 1.0:
                filename = f"screenshot_{int(t)}.png"
                pyautogui.screenshot(filename)
                self.click_cooldown = t
                self.log_gesture(gesture)
                print(f"📸 {filename}")
                self.notify("Screenshot", f"Saved: {filename}")
        elif gesture == 'play_pause':
            if t - self.click_cooldown > 0.6:
                os.system("osascript -e 'tell application \"System Events\" to keystroke space'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("▶️ PLAY/PAUSE")
        elif gesture == 'four_swipe_left':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('command', 'shift', 'tab')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("◀️ PREV APP")
        elif gesture == 'four_swipe_right':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('command', 'tab')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("▶️ NEXT APP")
        elif gesture == 'three_swipe_left':
            if t - self.click_cooldown > 0.6:
                if self.current_mode == 'presentation':
                    pyautogui.press('left')
                    print("⏮ PREV SLIDE")
                else:
                    pyautogui.hotkey('command', '[')
                    print("⏪ BACK")
                self.click_cooldown = t
                self.log_gesture(gesture)
        elif gesture == 'three_swipe_right':
            if t - self.click_cooldown > 0.6:
                if self.current_mode == 'presentation':
                    pyautogui.press('right')
                    print("⏭ NEXT SLIDE")
                else:
                    pyautogui.hotkey('command', ']')
                    print("⏩ FORWARD")
                self.click_cooldown = t
                self.log_gesture(gesture)
        elif gesture == 'minimize_window':
            if t - self.click_cooldown > 1.0:
                pyautogui.hotkey('command', 'm')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("⬇️ MINIMIZE")
        elif gesture == 'five_swipe_left':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('ctrl', 'left')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("◀️ PREV DESKTOP")
        elif gesture == 'five_swipe_right':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('ctrl', 'right')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("▶️ NEXT DESKTOP")
        elif gesture == 'show_desktop':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('fn', 'f11')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🖥 SHOW DESKTOP")
        elif gesture == 'mission_control':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('ctrl', 'up')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🎛 MISSION CONTROL")
    
    def draw_ui(self, frame, gesture, finger_count, fps):
        h, w = frame.shape[:2]
        
        if self.current_mode == 'hologram':
            # IRON MAN STYLE SCANLINES
            for i in range(0, h, 3):
                cv2.line(frame, (0, i), (w, i), (0, 30, 60), 1)
            
            # Corner brackets (HUD style)
            corner_size = 30
            corner_color = self.hologram_color
            # Top-left
            cv2.line(frame, (10, 10), (10 + corner_size, 10), corner_color, 2)
            cv2.line(frame, (10, 10), (10, 10 + corner_size), corner_color, 2)
            # Top-right
            cv2.line(frame, (w - 10, 10), (w - 10 - corner_size, 10), corner_color, 2)
            cv2.line(frame, (w - 10, 10), (w - 10, 10 + corner_size), corner_color, 2)
            # Bottom-left
            cv2.line(frame, (10, h - 10), (10 + corner_size, h - 10), corner_color, 2)
            cv2.line(frame, (10, h - 10), (10, h - 10 - corner_size), corner_color, 2)
            # Bottom-right
            cv2.line(frame, (w - 10, h - 10), (w - 10 - corner_size, h - 10), corner_color, 2)
            cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - corner_size), corner_color, 2)
            
            # Title
            cv2.putText(frame, "🦾 HOLOGRAM MODE", (w - 280, 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, self.hologram_color, 2)
            
            # Object count
            cv2.putText(frame, f"Objects: {len(self.spawned_objects)}", 
                       (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.glow_color, 1)
            
            # Selected object info
            if self.selected_object_id is not None:
                for obj_data in self.spawned_objects:
                    if obj_data['id'] == self.selected_object_id:
                        cv2.putText(frame, f"Selected: {obj_data['obj']['name']}", 
                                   (w - 280, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                   self.selected_color, 1)
                        break
        
        elif self.current_mode == 'drawing':
            alpha = 0.4
            overlay = cv2.addWeighted(frame, 1-alpha, self.drawing_canvas, alpha, 0)
            frame[:] = overlay
        
        else:
            # Normal UI
            cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
            
            mode_colors = {
                'normal': (0, 255, 0),
                'drawing': (255, 0, 255),
                'gaming': (0, 128, 255),
                'presentation': (255, 165, 0),
                'hologram': (0, 200, 255)
            }
            mode_color = mode_colors.get(self.current_mode, (255, 255, 255))
            cv2.putText(frame, f"MODE: {self.current_mode.upper()}", 
                       (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, mode_color, 2)
            
            gesture_color = (0, 255, 0) if gesture not in ['none', 'cursor'] else (100, 100, 100)
            cv2.putText(frame, f"{gesture.upper()}", 
                       (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
            
            cv2.putText(frame, f"F:{finger_count} | G:{self.total_gestures} | FPS:{int(fps)}", 
                       (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controls
        cv2.rectangle(frame, (5, h-35), (w-5, h-5), (40, 40, 40), -1)
        cv2.putText(frame, "Q:Quit | H:Hologram | M:Mode | D:Draw | P:Present | C:Clear | S:Stats | X:Deselect", 
                   (12, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return frame
    
    def show_stats(self):
        print("\n" + "="*70)
        print("📊 SESSION STATISTICS")
        print("="*70)
        duration = int(time.time() - self.session_start)
        print(f"Duration: {duration//60}m {duration%60}s")
        print(f"Total Gestures: {self.total_gestures}")
        print(f"Clicks: {len(self.click_positions)}")
        print(f"Objects Spawned: {len(self.spawned_objects)}")
        print("\nTop Gestures:")
        for gesture, count in sorted(self.gesture_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {gesture:25s}: {count:3d}")
        print("="*70 + "\n")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*70)
        print("🦾 IRON MAN HOLOGRAM CONTROLLER 🦾")
        print("="*70)
        print("\n✨ ALL FEATURES:")
        print("  ✅ 30+ Gesture Controls")
        print("  ✅ 3D Hologram Mode with 8 Models")
        print("  ✅ Multiple Objects at Same Time")
        print("  ✅ Zoom with Two Hands (Pinch/Spread)")
        print("  ✅ Drag & Drop Objects")
        print("  ✅ Delete Zone (Trash Bin)")
        print("  ✅ Rotate with Open Palm")
        print("  ✅ Select Individual Objects")
        print("\n🎮 HOLOGRAM CONTROLS:")
        print("  👆 INDEX FINGER: Drag object anywhere!")
        print("  👌 PINCH (thumb+index): Select from menu/objects")
        print("  🖐️ OPEN PALM: Rotate selected object")
        print("  🤲 TWO HANDS: Zoom in/out (spread/pinch)")
        print("  🗑️ DRAG TO TOP-RIGHT: Delete object")
        print("\nPress 'H' for hologram mode!")
        print("="*70 + "\n")
        
        self.notify("Gesture Control", "Iron Man Edition Ready!")
        
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            gesture = 'none'
            finger_count = 0
            lm_lists = []
            all_fingers = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0,150,255), thickness=2)
                    )
                    
                    lm_list = self.get_landmarks(hand_landmarks)
                    fingers = self.count_fingers(lm_list)
                    lm_lists.append(lm_list)
                    all_fingers.append(fingers)
                
                if len(lm_lists) > 0:
                    finger_count = all_fingers[0].count(1)
                    gesture, pos = self.detect_gesture_advanced(lm_lists, all_fingers)
                    
                    if gesture != 'none':
                        self.execute_all(gesture, pos, lm_lists)
                    
                    if pos:
                        self.prev_hand_pos = pos
            else:
                self.prev_hand_x = None
                self.prev_hand_pos = None
                self.prev_wrist = None
                self.delete_zone['active'] = False
            
            # Draw hologram elements
            if self.current_mode == 'hologram':
                self.draw_menu(frame)
                self.draw_delete_zone(frame)
                
                # Draw all spawned objects
                for obj_data in self.spawned_objects:
                    self.draw_3d_object(frame, obj_data)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            frame = self.draw_ui(frame, gesture, finger_count, fps)
            
            cv2.imshow("Iron Man Hologram Controller", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                modes = ['normal', 'drawing', 'gaming', 'presentation', 'hologram']
                idx = modes.index(self.current_mode)
                self.current_mode = modes[(idx + 1) % len(modes)]
                print(f"🎯 Mode: {self.current_mode.upper()}")
                self.notify("Mode Change", self.current_mode.upper())
            elif key == ord('d'):
                self.current_mode = 'drawing'
                print("🎨 DRAWING MODE")
            elif key == ord('p'):
                self.current_mode = 'presentation'
                print("📊 PRESENTATION MODE")
            elif key == ord('h'):
                self.current_mode = 'hologram' if self.current_mode != 'hologram' else 'normal'
                print(f"🦾 HOLOGRAM: {self.current_mode.upper()}")
            elif key == ord('c'):
                if self.current_mode == 'drawing':
                    self.init_drawing_canvas()
                    self.drawing_points = []
                elif self.current_mode == 'hologram':
                    self.spawned_objects = []
                    self.selected_object_id = None
                    for item in self.menu_items:
                        item['selected'] = False
                print("🗑 Cleared")
            elif key == ord('x'):
                # Deselect all
                self.selected_object_id = None
                for obj_data in self.spawned_objects:
                    obj_data['selected'] = False
                print("❌ Deselected all")
            elif key == ord('s'):
                self.show_stats()
                self.save_stats()
        
        cap.release()
        cv2.destroyAllWindows()
        self.show_stats()
        self.save_stats()
        print("\n✅ Session Ended!\n")

if __name__ == "__main__":
    controller = HologramController()
    controller.run()
