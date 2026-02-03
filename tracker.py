from dataclasses import dataclass, field
from typing import Optional
import math

""" Return bbox corners for IoU calculations"""
def xywh_to_xyxy(x, y, w, h):
    return (x - w/2, y - h/2, x + w/2, y + h/2)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def dist(a: tuple[float,float], b: tuple[float,float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

@dataclass
class Track:
    track_id: int
    cls: str
    last_xy: tuple[float, float]
    last_t: float
    last_v: tuple[float, float] = (0.0, 0.0)
    last_a: tuple[float, float] = (0.0, 0.0)
    age: int = 0       # Successful frame matches
    missed: int = 0    # Frames not matched
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    last_box_xyxy: Optional[tuple[float, float, float, float]] = None

class SimpleTracker:
    def __init__(
        self,
        max_age: int = 10,
        conf_thres: float = 0.5,
        use_iou_gate: bool = False,
        min_iou: float = 0.1,
        max_dist_by_class: Optional[dict[str, float]] = None,
        vel_smooth_alpha: float = 0.6
    ):
        self.max_age = max_age
        self.conf_thres = conf_thres
        self.use_iou_gate = use_iou_gate
        self.min_iou = min_iou
        self.vel_smooth_alpha = vel_smooth_alpha
        
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        
        self.max_dist_by_class = max_dist_by_class or {
            "player": 80.0,
            "bullet": 60.0,
            "saw": 25.0,
            "rg": 60.0,
            "goal": 25.0,
        }
    
    def _max_dist(self, cls: str) -> float:
        return self.max_dist_by_class.get(cls, 80.0)
    
    def _new_track(self, det: dict, t: float) -> Track:
        tid = self.next_id
        self.next_id += 1
        x, y = float(det["x"]), float(det["y"])
        w, h = float(det["width"]), float(det["height"])
        tr = Track(
            track_id=tid,
            cls=det["class"],
            last_xy=(x, y),
            last_t=t,
            history=[(t, x, y)],
            last_box_xyxy=xywh_to_xyxy(x, y, w, h),
        )
        self.tracks[tid] = tr
        return tr
    
    def _update_track(self, tr: Track, det: dict, t: float):
        x, y = float(det["x"]), float(det["y"])
        dt = t - tr.last_t
        # print("DEBUG", tr.track_id, "dt=", dt, "last_t=", tr.last_t, "t=", t)
        
        if dt > 1e-6:
            vx = (x - tr.last_xy[0]) / dt
            vy = (y - tr.last_xy[1]) / dt
            
            # Smoothing to reduce bbox jitter
            alpha = self.vel_smooth_alpha
            svx = alpha * vx + (1 - alpha) * tr.last_v[0]
            svy = alpha * vy + (1 - alpha) * tr.last_v[1]
        
            ax = (svx - tr.last_v[0]) / dt
            ay = (svy - tr.last_v[1]) / dt
            
            tr.last_v = (svx, svy)
            tr.last_a = (ax, ay)
            
        tr.last_xy = (x, y)
        tr.last_t = t
        tr.missed = 0
        tr.age += 1
        tr.history.append((t, x, y))
        
        w, h = float(det["width"]), float(det["height"])
        tr.last_box_xyxy = xywh_to_xyxy(x, y, w, h)
        
    """ Call once per frame """
    def update(self, result: dict, t: float) -> list[Track]:
        t = float(t)
        
        detections = [
            d for d in result.get("predictions", ())
            if float(d.get("confidence", 0.0)) >= self.conf_thres
        ]
        
        for tr in self.tracks.values():
            tr.missed += 1
        
        dets_by_class: dict[str, list[dict]] = {}
        for d in detections:
            dets_by_class.setdefault(d["class"], []).append(d)
            
        used_track_ids = set()
        
        for cls, dets in dets_by_class.items():
            cand_tracks = [tr for tr in self.tracks.values() if tr.cls == cls and tr.track_id not in used_track_ids]
            
            dets = sorted(dets, key=lambda d: float(d["confidence"]), reverse=True)
            
            for det in dets:
                best_tr = None
                best_cost = float("inf")
                
                det_xy = (float(det["x"]), float(det["y"]))
                det_box = xywh_to_xyxy(det_xy[0], det_xy[1], float(det["width"]), float(det["height"]))
                max_d = self._max_dist(cls)
                
                for tr in cand_tracks:
                    if tr.track_id in used_track_ids:
                        continue
                    
                    d = dist(tr.last_xy, det_xy)
                    if d > max_d:
                        continue
                    
                    if self.use_iou_gate and tr.last_box_xyxy is not None:
                        if iou_xyxy(tr.last_box_xyxy, det_box) < self.min_iou:
                            continue
                    
                    if d < best_cost:
                        best_cost = d
                        best_tr = tr
                        
                if best_tr is None:
                    self._new_track(det, t)
                else:
                    self._update_track(best_tr, det, t)
                    used_track_ids.add(best_tr.track_id)
                    
        to_delete = [tid for tid, tr in self.tracks.items() if tr.missed > self.max_age]
        for tid in to_delete:
            del self.tracks[tid]

        return list(self.tracks.values())
    
__all__ = ["SimpleTracker", "Track"]