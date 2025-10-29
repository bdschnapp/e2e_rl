from DistributionCenter import *
from TractorTrailer import *


class DistributionCenterLatticePlannerEnv(DistributionCenterEnv):
    """
    Ring + Sweep planner:
      - Fixed rectangular ring (blue) inside map
      - Start -> ring: single forward circular arc
      - Along ring (shorter CW/CCW)
      - Ring -> goal: single reverse circular arc (tangent at both ends)
    """

    def __init__(self,
                 render_mode='human',
                 ring_margin_m: float = 6.0,  # distance from map edges
                 ring_step_m: float = 0.05,  # sampling along ring
                 arc_ds_m: float = 0.05,  # sampling along arcs
                 ang_tol_rad: float = np.deg2rad(20),  # tangent tolerance
                 robot_radius_m: float = 1.4):  # collision shell
        super().__init__(render_mode)
        self.ring_margin = float(ring_margin_m)
        self.ring_step = float(ring_step_m)
        self.arc_ds = float(arc_ds_m)
        self.ang_tol = float(ang_tol_rad)
        self.robot_r = float(robot_radius_m)

        self._circle_mask = None
        self._circle_radius_px = None

        # planned geometry (all in meters)
        self.ring_pts = []  # [(x,y,yaw_tangent), ...] CW order
        self.forward_segments = []  # list of [(x,y), ...]
        self.reverse_segments = []  # list of [(x,y), ...]

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prepare_collision_circle()
        self._build_ring()
        try:
            self._plan_ring_sweep()
        except Exception as e:
            print(f"[ring-planner] planning failed: {e}")
            self.forward_segments, self.reverse_segments = [], []
        return obs, info

    def _render_frame(self, surface=None):
        super()._render_frame(surface)
        surface = surface if surface else self.canvas

        # draw ring (thin blue outline)
        if self.ring_pts:
            self._draw_polyline_m(surface, [p[:2] for p in self.ring_pts] + [self.ring_pts[0][:2]], (40, 110, 255), 2)

        # forward in blue
        for seg in self.forward_segments:
            self._draw_polyline_m(surface, seg, (40, 110, 255), 5)

        # reverse in orange
        for seg in self.reverse_segments:
            self._draw_polyline_m(surface, seg, (240, 140, 30), 5)

        return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    # ---------- Planner ----------
    def _plan_ring_sweep(self):
        # poses
        sx, sy, syaw = self.vehicle.x, self.vehicle.y, self.vehicle.p
        gx, gy, gyaw = self.goal_pose

        # (1) choose docking tangent on ring: a single reverse circular arc ring->goal
        dock = self._best_ring_to_goal_arc((gx, gy, gyaw))
        if dock is None:
            raise RuntimeError("No reverse docking arc found")
        ring_idx_goal, rev_arc_pts = dock

        # (2) connect start to ring with a single forward arc (to any ring sample);
        #     bias towards something near the ring point chosen above, but fall back to nearest.
        entry = self._best_start_to_ring_arc((sx, sy, syaw), prefer_idx=ring_idx_goal)
        if entry is None:
            raise RuntimeError("No start->ring forward arc found")
        ring_idx_start, fwd_arc_pts = entry

        # (3) follow ring CW/CCW (shortest) from start_idx to goal_idx
        ring_path_pts = self._ring_segment(ring_idx_start, ring_idx_goal)

        # store
        self.forward_segments = []
        if fwd_arc_pts:
            self.forward_segments.append([(x, y) for (x, y, _) in fwd_arc_pts])
        if ring_path_pts:
            self.forward_segments.append(ring_path_pts)
        self.reverse_segments = []
        if rev_arc_pts:
            self.reverse_segments.append([(x, y) for (x, y, _) in rev_arc_pts])

    # ---------- Ring construction ----------
    def _build_ring(self):
        """Rectangular ring inset from map borders; samples include tangent yaw (CW)."""
        Wm = WINDOW_WIDTH * METERS_PER_PIXEL
        Hm = WINDOW_HEIGHT * METERS_PER_PIXEL
        m = self.ring_margin

        # rectangle corners (CW)
        p0 = (m, m)  # bottom-left
        p1 = (Wm - m, m)  # bottom-right
        p2 = (Wm - m, Hm - m)  # top-right
        p3 = (m, Hm - m)  # top-left
        corners = [p0, p1, p2, p3]

        # directions (CW): right, up, left, down
        tangents = [0.0, np.pi / 2, np.pi, -np.pi / 2]

        pts = []
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            yaw = tangents[i]
            seg_len = np.hypot(b[0] - a[0], b[1] - a[1])
            n = max(2, int(np.ceil(seg_len / self.ring_step)))
            for k in range(n):
                t = k / n
                x = a[0] + t * (b[0] - a[0])
                y = a[1] + t * (b[1] - a[1])
                pts.append((x, y, yaw))
        self.ring_pts = pts

    # ---------- Geometry ----------
    def _best_ring_to_goal_arc(self, goal_pose):
        """
        Quarter-circle template from ring to goal.
        Works for goal yaw ≈ 0, 90, 180, 270 deg (your generator uses 0 or 90).
        Returns (ring_idx_goal, pts_with_yaw) or None.
        """
        gx, gy, gyaw = goal_pose
        gyaw = self._wrap(gyaw)

        # ring edges in meters
        Wm = WINDOW_WIDTH * METERS_PER_PIXEL
        Hm = WINDOW_HEIGHT * METERS_PER_PIXEL
        m = self.ring_margin
        xL, xR = m, Wm - m
        yB, yT = m, Hm - m

        def sample_arc(center, R, side, phi_start, phi_end, ds):
            """Sample arc around 'center' from phi_start -> phi_end (signed), add yaw."""
            # total angle
            dphi = self._wrap(phi_end - phi_start)
            # enforce direction consistent with side: left(CCW) => dphi>0, right(CW)=>dphi<0
            if side > 0 and dphi < 0: dphi += 2 * np.pi
            if side < 0 and dphi > 0: dphi -= 2 * np.pi
            L = abs(R * dphi)
            n = max(2, int(np.ceil(L / self.arc_ds)))
            out = []
            for k in range(n + 1):
                t = k / n
                phi = phi_start + t * dphi
                x = center[0] + R * np.cos(phi)
                y = center[1] + R * np.sin(phi)
                yaw = self._wrap(phi + side * np.pi / 2)
                if not self._point_free(x, y):  # strict collision check
                    return None
                out.append((x, y, yaw))
            return out

        # Helpers for each yaw bucket (snap to axis)
        def try_horizontal(yaw_bucket):
            """
            Goal yaw ≈ 0 (east) or ≈ π (west).
            Use vertical ring sides (x = xL or xR).
            """
            nonlocal gx, gy
            # decide which side is 'left turn' wrt goal yaw
            if yaw_bucket == 0.0:  # facing +x
                # left edge candidate (CCW, side=+1)
                R = gx - xL
                if R > 0:
                    C = (gx, gy + R)  # center above goal
                    # end radial angle and start angle for 90° sweep
                    phi_end = gyaw - (+1) * np.pi / 2
                    phi_start = phi_end - (+1) * np.pi / 2
                    y_start = gy + R
                    if yB <= y_start <= yT:
                        pts = sample_arc(C, R, +1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((xL, y_start), -np.pi / 2)
                            return idx, pts
                # right edge candidate (CW, side=-1)
                R = xR - gx
                if R > 0:
                    C = (gx, gy - R)  # center below goal
                    phi_end = gyaw - (-1) * np.pi / 2
                    phi_start = phi_end - (-1) * np.pi / 2
                    y_start = gy - R
                    if yB <= y_start <= yT:
                        pts = sample_arc(C, R, -1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((xR, y_start), +np.pi / 2)
                            return idx, pts
            else:  # facing -x (≈ π)
                # right edge candidate (CCW, side=+1)
                R = xR - gx
                if R > 0:
                    C = (gx, gy - R)  # center below goal
                    phi_end = gyaw - (+1) * np.pi / 2
                    phi_start = phi_end - (+1) * np.pi / 2
                    y_start = gy - R
                    if yB <= y_start <= yT:
                        pts = sample_arc(C, R, +1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((xR, y_start), +np.pi / 2)
                            return idx, pts
                # left edge candidate (CW, side=-1)
                R = gx - xL
                if R > 0:
                    C = (gx, gy + R)  # center above goal
                    phi_end = gyaw - (-1) * np.pi / 2
                    phi_start = phi_end - (-1) * np.pi / 2
                    y_start = gy + R
                    if yB <= y_start <= yT:
                        pts = sample_arc(C, R, -1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((xL, y_start), -np.pi / 2)
                            return idx, pts
            return None

        def try_vertical(yaw_bucket):
            """
            Goal yaw ≈ +π/2 (north) or ≈ -π/2 (south).
            Use horizontal ring sides (y = yB or yT).
            """
            nonlocal gx, gy
            if yaw_bucket == +np.pi / 2:  # facing +y
                # bottom edge (CCW, side=+1)
                R = gy - yB
                if R > 0:
                    C = (gx - R, gy)  # center left of goal
                    phi_end = gyaw - (+1) * np.pi / 2
                    phi_start = phi_end - (+1) * np.pi / 2
                    x_start = gx - R
                    if xL <= x_start <= xR:
                        pts = sample_arc(C, R, +1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((x_start, yB), 0.0)
                            return idx, pts
                # top edge (CW, side=-1) — only valid if geometry allows (rare with quarter circle)
                R = yT - gy
                if R > 0:
                    C = (gx + R, gy)  # center right of goal
                    phi_end = gyaw - (-1) * np.pi / 2
                    phi_start = phi_end - (-1) * np.pi / 2
                    x_start = gx + R
                    if xL <= x_start <= xR:
                        pts = sample_arc(C, R, -1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((x_start, yT), np.pi)
                            return idx, pts
            else:  # facing -y (≈ -π/2)
                # top edge (CCW, side=+1)
                R = yT - gy
                if R > 0:
                    C = (gx + R, gy)  # center right of goal
                    phi_end = gyaw - (+1) * np.pi / 2
                    phi_start = phi_end - (+1) * np.pi / 2
                    x_start = gx + R
                    if xL <= x_start <= xR:
                        pts = sample_arc(C, R, +1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((x_start, yT), np.pi)
                            return idx, pts
                # bottom edge (CW, side=-1)
                R = gy - yB
                if R > 0:
                    C = (gx - R, gy)  # center left of goal
                    phi_end = gyaw - (-1) * np.pi / 2
                    phi_start = phi_end - (-1) * np.pi / 2
                    x_start = gx - R
                    if xL <= x_start <= xR:
                        pts = sample_arc(C, R, -1, phi_start, phi_end, self.arc_ds)
                        if pts is not None:
                            idx = self._closest_ring_idx_with_yaw((x_start, yB), 0.0)
                            return idx, pts
            return None

        # snap goal yaw to the closest axis
        a = self._wrap(gyaw)
        if abs(a) <= np.pi / 4 or abs(abs(a) - 2 * np.pi) <= np.pi / 4:
            out = try_horizontal(0.0)
        elif abs(a - np.pi / 2) <= np.pi / 4:
            out = try_vertical(+np.pi / 2)
        elif abs(a + np.pi / 2) <= np.pi / 4:
            out = try_vertical(-np.pi / 2)
        else:
            out = try_horizontal(np.pi)  # treat as ~π

        if out is not None:
            return out

        # Fallback to the generic solver if quarter-circle fails
        return super()._best_ring_to_goal_arc(goal_pose) if hasattr(super(), "_best_ring_to_goal_arc") else None

    def _closest_ring_idx_with_yaw(self, target_xy, yaw_target, yaw_tol=np.deg2rad(12)):
        """Pick the ring sample closest to target_xy whose stored tangent yaw matches yaw_target."""
        tx, ty = target_xy
        best = None
        best_d = np.inf
        for i, (x, y, yaw) in enumerate(self.ring_pts):
            if abs(self._wrap(yaw - yaw_target)) > yaw_tol:
                continue
            d = (x - tx) ** 2 + (y - ty) ** 2
            if d < best_d:
                best_d = d
                best = i
        return best if best is not None else int(
            np.argmin([(x - tx) ** 2 + (y - ty) ** 2 for (x, y, _) in self.ring_pts]))

    def _best_start_to_ring_arc(self, start_pose, prefer_idx=None):
        """Forward arc from start to ring. Prefer a neighborhood around prefer_idx first."""
        sx, sy, syaw = start_pose

        indices = list(range(len(self.ring_pts)))
        if prefer_idx is not None:
            # examine a window around preferred index first
            window = 40
            order = []
            for d in range(0, window + 1):
                if prefer_idx - d >= 0: order.append(prefer_idx - d)
                if prefer_idx + d < len(self.ring_pts): order.append(prefer_idx + d)
            # append the rest
            rest = [i for i in indices if i not in order]
            indices = order + rest

        best = None
        best_len = np.inf

        for idx in indices:
            rx, ry, ryaw = self.ring_pts[idx]
            for side in (+1, -1):
                arc = self._circle_arc_tangent_to_both((sx, sy, syaw), (rx, ry, ryaw),
                                                       side=side, ds=self.arc_ds)
                if arc is None:
                    continue
                pts, R, ang = arc
                if not self._polyline_free([(x, y) for (x, y, _) in pts]):
                    continue
                L = abs(R * ang)
                if L < best_len:
                    best_len = L
                    best = (idx, pts)
            # small speed-up: if we already found a very short arc near prefer_idx
            if best is not None and prefer_idx is not None and abs(idx - prefer_idx) < 5:
                break
        return best

    # single circular arc connecting two poses with given tangents (same turn side)
    def _circle_arc_tangent_to_both(self, pose1, pose2, side=+1, ds=0.25):
        (x1, y1, yaw1) = pose1
        (x2, y2, yaw2) = pose2

        # unit normals pointing to circle center for the chosen side
        n1 = np.array([np.cos(yaw1 + side * np.pi / 2), np.sin(yaw1 + side * np.pi / 2)])
        n2 = np.array([np.cos(yaw2 + side * np.pi / 2), np.sin(yaw2 + side * np.pi / 2)])

        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])

        # Solve p1 + a*n1 = p2 + b*n2  ->  [n1 | -n2] [a b]^T = (p2 - p1)
        A = np.array([[n1[0], -n2[0]], [n1[1], -n2[1]]], dtype=float)
        det = np.linalg.det(A)
        if abs(det) < 1e-9:
            return None  # nearly parallel tangents; skip
        a, b = np.linalg.solve(A, (p2 - p1))
        # center
        C = p1 + a * n1
        R1 = np.linalg.norm(C - p1)
        R2 = np.linalg.norm(C - p2)
        if R1 < 1e-6 or R2 < 1e-6 or abs(R1 - R2) > 1e-3:
            return None
        R = 0.5 * (R1 + R2)

        # tangent consistency check (loose)
        def wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        th1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
        th2 = np.arctan2(p2[1] - C[1], p2[0] - C[0])
        yaw1_chk = wrap(th1 + side * np.pi / 2)
        yaw2_chk = wrap(th2 + side * np.pi / 2)
        if abs(wrap(yaw1 - yaw1_chk)) > self.ang_tol: return None
        if abs(wrap(yaw2 - yaw2_chk)) > self.ang_tol: return None

        # direction: left => CCW, right => CW
        dtheta = wrap(th2 - th1)
        if side == +1 and dtheta < 0: dtheta += 2 * np.pi
        if side == -1 and dtheta > 0: dtheta -= 2 * np.pi

        # sample arc
        L = abs(R * dtheta)
        n = max(2, int(np.ceil(L / ds)))
        pts = []
        for k in range(n + 1):
            t = k / n
            th = th1 + t * dtheta
            x = C[0] + R * np.cos(th)
            y = C[1] + R * np.sin(th)
            yaw = wrap(th + side * np.pi / 2)
            if not self._point_free(x, y):
                return None
            pts.append((x, y, yaw))
        return pts, R, dtheta

    # ring traversal between two indices (CW/CCW shortest)
    def _ring_segment(self, i_start, i_goal):
        N = len(self.ring_pts)
        if N == 0:
            return []

        # CW path length in samples (ring list is CW)
        cw_len = (i_goal - i_start) % N
        ccw_len = (i_start - i_goal) % N

        if cw_len <= ccw_len:
            idxs = [(i_start + k) % N for k in range(cw_len + 1)]
        else:
            # go CCW by walking backwards
            idxs = [(i_start - k) % N for k in range(ccw_len + 1)]

        pts = []
        for idx in idxs:
            x, y, _ = self.ring_pts[idx]
            pts.append((x, y))
        # quick collision check along ring segment (should be free)
        if not self._polyline_free(pts):
            # if margin is too small, warn but still return (drawing aid)
            print("[ring-planner] WARNING: ring segment touches obstacles; increase ring_margin_m.")
        return pts

    # ---------- Collision helpers ----------
    def _prepare_collision_circle(self):
        r_px = max(2, int(round(self.robot_r / METERS_PER_PIXEL)))
        if self._circle_mask is None or r_px != getattr(self, "_circle_radius_px", None):
            surf = pygame.Surface((2 * r_px + 1, 2 * r_px + 1), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            pygame.draw.circle(surf, (255, 255, 255, 255), (r_px, r_px), r_px)
            self._circle_mask = pygame.mask.from_surface(surf)
            self._circle_radius_px = r_px

    def _point_free(self, x_m, y_m) -> bool:
        Wm = WINDOW_WIDTH * METERS_PER_PIXEL
        Hm = WINDOW_HEIGHT * METERS_PER_PIXEL
        if not (0.0 <= x_m <= Wm and 0.0 <= y_m <= Hm):
            return False
        if self.obstacle_mask is None:
            return True
        cx = int(round(x_m / METERS_PER_PIXEL))
        cy = int(round(WINDOW_HEIGHT - (y_m / METERS_PER_PIXEL)))
        off = (cx - self._circle_radius_px, cy - self._circle_radius_px)
        return self.obstacle_mask.overlap(self._circle_mask, off) is None

    def _polyline_free(self, pts_xy) -> bool:
        # dense resampling for safety
        if len(pts_xy) < 2:
            return True
        for i in range(len(pts_xy) - 1):
            if not self._segment_free(pts_xy[i], pts_xy[i + 1], step=self.arc_ds / 2):
                return False
        return True

    def _segment_free(self, a_xy, b_xy, step=0.15):
        ax, ay = a_xy
        bx, by = b_xy
        dist = max(1e-6, np.hypot(bx - ax, by - ay))
        n = int(np.ceil(dist / step))
        for k in range(n + 1):
            t = k / n
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            if not self._point_free(x, y): return False
        return True

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        return ((a - b + np.pi) % (2.0 * np.pi)) - np.pi

    # ---------- Drawing ----------
    def _draw_polyline_m(self, surface, pts_m, color, width=2):
        if len(pts_m) < 2: return
        pts_px = []
        for (x, y) in pts_m:
            px = x / METERS_PER_PIXEL
            py = WINDOW_HEIGHT - (y / METERS_PER_PIXEL)
            pts_px.append((px, py))
        pygame.draw.lines(surface, color, False, pts_px, width)


if __name__ == "__main__":
    env = DistributionCenterLatticePlannerEnv(
        render_mode='human',
        ring_margin_m=10.0,
        ring_step_m=0.05,
        arc_ds_m=0.05,
        ang_tol_rad=np.deg2rad(20),
        robot_radius_m=0.2
    )
    obs, info = env.reset()

    running = True
    while running:
        # you can drive freestyle OR later add a follower that tracks env.planned_segments
        action = np.array([0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                terminated = True
        if terminated or truncated:
            running = False
    env.close()
