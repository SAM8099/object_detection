import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import time

def counter(VIDEO_PATH):
    MODEL_PATH = "yolov8n.pt"    # or yolov8m.pt for better accuracy
    # Original door coordinates in the original video resolution:
    ORIG_DOOR = (85, 45, 220, 290)   # user's door coords in original video resolution
    RESIZE = (640, 480)              # set to None to keep original resolution
    GATE_INSET_PX = 20               # inset for inner gate box (pixels)
    MIN_FRAMES_BEFORE_COUNT = 2
    SIDE_CHANGE_COOLDOWN_FRAMES = 8  # cooldown after a counted crossing
    TRAIL_LEN = 20                   # how many past centroids to draw for each id
    # ====================

    model = YOLO(MODEL_PATH)

    # open video
    cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_PATH != 0 else 0)
    if not cap.isOpened():
        print("ERROR: cannot open video:", VIDEO_PATH)
        raise SystemExit

    # read a sample frame to compute scaling
    ret, sample = cap.read()
    if not ret:
        print("ERROR: cannot read frame from video.")
        cap.release()
        raise SystemExit

    orig_h, orig_w = sample.shape[:2]
    if RESIZE is not None:
        frame_w, frame_h = RESIZE
        scale_x = frame_w / orig_w
        scale_y = frame_h / orig_h
    else:
        frame_w, frame_h = orig_w, orig_h
        scale_x = scale_y = 1.0

    # scale door coords
    x1_o, y1_o, x2_o, y2_o = ORIG_DOOR
    door_x1 = int(round(x1_o * scale_x))
    door_y1 = int(round(y1_o * scale_y))
    door_x2 = int(round(x2_o * scale_x))
    door_y2 = int(round(y2_o * scale_y))

    # inner gate (inset)
    inset = GATE_INSET_PX
    # If resize scaling was applied, optionally scale inset (here we assume pixels are fine)
    # but we can scale inset too:
    inset_x = int(round(inset * scale_x))
    inset_y = int(round(inset * scale_y))

    gate_x1 = door_x1 + inset_x
    gate_y1 = door_y1 + inset_y
    gate_x2 = door_x2 - inset_x
    gate_y2 = door_y2 - inset_y

    # choose axis for side detection: if door is wider than tall -> people cross along y axis
    door_w = door_x2 - door_x1
    door_h = door_y2 - door_y1
    axis = "y" if door_w > door_h else "x"

    print("Using frame size:", frame_w, frame_h)
    print("Door box:", (door_x1,door_y1,door_x2,door_y2))
    print("Gate box:", (gate_x1,gate_y1,gate_x2,gate_y2))
    print("Axis:", axis)

    # per-ID state structure:
    # track_info[id] = {
    #   'last_pos': float,
    #   'seen_frames': int,
    #   'last_side': -1/0/1 or None,
    #   'stage': 'unknown'|'outside_A'|'inside_gate'|'outside_B',
    #   'cooldown': int,
    #   'trail': [(x,y), ...]
    # }
    track_info = {}

    total_count = 0
    total_count = 0
    frame_idx = 0
    t0 = time.time()

    def get_side(pos):
        """Return -1 if on side A, 0 if inside door/gate range, 1 if on side B (relative to door bounds)."""
        if axis == "y":
            if pos < door_y1:
                return -1
            elif pos > door_y2:
                return 1
            else:
                return 0
        else:
            if pos < door_x1:
                return -1
            elif pos > door_x2:
                return 1
            else:
                return 0

    def inside_gate(cx, cy):
        return (gate_x1 <= cx <= gate_x2) and (gate_y1 <= cy <= gate_y2)

    def stage_from_side(side):
        if side == -1:
            return "outside_A"
        elif side == 1:
            return "outside_B"
        else:
            return "inside_gate"

    # reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        frame_idx += 1

        if RESIZE is not None:
            frame = cv2.resize(frame, (frame_w, frame_h))

        # detect & track people
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        # Draw door and gate
        cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (200,200,0), 2)
        cv2.putText(frame, "Door", (door_x1, max(10, door_y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        cv2.rectangle(frame, (gate_x1, gate_y1), (gate_x2, gate_y2), (0,200,200), 2)
        cv2.putText(frame, "Gate", (gate_x1, max(10, gate_y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)

        # prepare boxes & ids
        boxes = []
        ids = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes_tensor = results[0].boxes.xyxy
            if boxes_tensor is not None and len(boxes_tensor) > 0:
                boxes = boxes_tensor.cpu().numpy()
            if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)

        present_ids = set()

        # iterate detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            pid = int(ids[i]) if i < len(ids) else None
            if pid is None:
                continue
            present_ids.add(pid)

            # draw bbox & centroid
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
            cv2.putText(frame, f"ID:{pid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # compute pos along axis
            pos = cy if axis == "y" else cx
            side = get_side(pos)
            in_gate = inside_gate(cx, cy)

            info = track_info.get(pid)
            if info is None:
                # initialize but keep last_side None to handle starting inside gate
                track_info[pid] = {
                    'last_pos': pos,
                    'seen_frames': 1,
                    'last_side': None,
                    'stage': None,
                    'cooldown': 0,
                    'trail': [(cx, cy)]
                }
                # don't count yet
                # set stage if we can detect a clear outside side
                if side == -1:
                    track_info[pid]['stage'] = "outside_A"
                    track_info[pid]['last_side'] = -1
                elif side == 1:
                    track_info[pid]['stage'] = "outside_B"
                    track_info[pid]['last_side'] = 1
                elif in_gate:
                    # started inside gate; stage unknown until they move out to a clear side
                    track_info[pid]['stage'] = "inside_gate"
                    track_info[pid]['last_side'] = 0
                continue

            # update info
            info['seen_frames'] += 1
            info['trail'].append((cx, cy))
            if len(info['trail']) > TRAIL_LEN:
                info['trail'].pop(0)
            if info.get('cooldown',0) > 0:
                info['cooldown'] -= 1

            prev_stage = info['stage']
            prev_side = info['last_side']

            # Determine current stage:
            if in_gate:
                curr_stage = "inside_gate"
                curr_side = 0
            else:
                # if outside, map side to stage
                if side == -1:
                    curr_stage = "outside_A"
                else:
                    curr_stage = "outside_B"
                curr_side = side

            # If we haven't set prev_stage, initialize conservatively
            if prev_stage is None:
                info['stage'] = curr_stage
                info['last_side'] = curr_side
                info['last_pos'] = pos
                continue

            # Stage machine and counting: require sequence A -> inside_gate -> B to count an entry
            # or B -> inside_gate -> A to count an exit. Also allow case where they start inside_gate and then exit to a side.
            counted = False
            if info['seen_frames'] >= MIN_FRAMES_BEFORE_COUNT and info.get('cooldown',0) == 0:
                # If previously outside_A and now inside_gate -> update stage
                if prev_stage == "outside_A" and curr_stage == "inside_gate":
                    info['stage'] = "inside_gate"
                # If previously inside_gate and now outside_B -> completed A->inside->B => entry
                elif prev_stage == "inside_gate" and curr_stage == "outside_B":
                    # completed A->inside->B OR started inside->outside_B (we count as entry)
                    total_count += 1
                    print(f"[{frame_idx}] Person {pid} ENTERED (inside->outside_B)")
                    counted = True
                # If previously outside_B and now inside_gate -> update stage
                elif prev_stage == "outside_B" and curr_stage == "inside_gate":
                    info['stage'] = "inside_gate"
                # If previously inside_gate and now outside_A -> completed B->inside->A => exit
                elif prev_stage == "inside_gate" and curr_stage == "outside_A":
                    total_count += 1
                    print(f"[{frame_idx}] Person {pid} EXITED (inside->outside_A)")
                    counted = True
                # If started outside_A and later directly went to outside_B skipping inside (fast frames)
                elif prev_stage == "outside_A" and curr_stage == "outside_B":
                    # more conservative: require they at least passed through gate region; so do not count immediately
                    # but if they are far away and crossing likely, you could count here. We'll ignore to avoid false positives.
                    pass
                elif prev_stage == "outside_B" and curr_stage == "outside_A":
                    pass
                # If started inside_gate (initially) and now outside_B => entry
                elif prev_stage == "inside_gate" and curr_stage == "outside_B":
                    total_count += 1
                    print(f"[{frame_idx}] Person {pid} ENTERED (started inside->outside_B)")
                    counted = True
                elif prev_stage == "inside_gate" and curr_stage == "outside_A":
                    total_count += 1
                    print(f"[{frame_idx}] Person {pid} EXITED (started inside->outside_A)")
                    counted = True

            # On count, set cooldown and update stage to the outside side
            if counted:
                info['cooldown'] = SIDE_CHANGE_COOLDOWN_FRAMES
                info['stage'] = curr_stage
                info['last_side'] = curr_side
            else:
                # no count, just update stage/side/pos
                info['stage'] = curr_stage
                info['last_side'] = curr_side
                info['last_pos'] = pos

            # draw trail
            for t_i in range(1, len(info['trail'])):
                cv2.line(frame, info['trail'][t_i-1], info['trail'][t_i], (0,255,0), 2)

        # prune track_info for ids not present (keeps memory small)
        to_delete = [pid for pid in list(track_info.keys()) if pid not in present_ids]
        for pid in to_delete:
            del track_info[pid]

        # draw big flash text when last frame had a count? (Simple approach: show counts always)
        cv2.putText(frame, f"Total: {total_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    

        cv2.imshow("People Counter (inner-gate)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print("Finished. total:",  total_count, f"Elapsed {elapsed:.1f}s")
    return total_count