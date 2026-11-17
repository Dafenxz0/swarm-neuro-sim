pub const WORLD_WIDTH: f32 = 1000.0;
pub const WORLD_HEIGHT: f32 = 700.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Segment {
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    #[inline]
    pub fn midpoint(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    #[inline]
    pub fn dist_sq_to_point(&self, px: f32, py: f32) -> f32 {
        let vx = self.x2 - self.x1;
        let vy = self.y2 - self.y1;
        let len_sq = vx * vx + vy * vy;
        if len_sq < 1e-8 {
            let dx = px - self.x1;
            let dy = py - self.y1;
            return dx * dx + dy * dy;
        }
        let t = (((px - self.x1) * vx + (py - self.y1) * vy) / len_sq).clamp(0.0, 1.0);
        let proj_x = self.x1 + t * vx;
        let proj_y = self.y1 + t * vy;
        let dx = px - proj_x;
        let dy = py - proj_y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    pub preset_id: u32,
    pub walls: Vec<Segment>,
    pub boundary_count: usize,
    pub checkpoints: Vec<Segment>,
    pub spawn_x: f32,
    pub spawn_y: f32,
    pub spawn_angle: f32,
}

impl Track {
    pub fn from_preset(preset_id: u32) -> Self {
        let mut walls = boundary_walls();
        let boundary_count = walls.len();

        let (track_walls, checkpoints, spawn_x, spawn_y, spawn_angle) = match preset_id {
            1 => technical_s_curve(),
            2 => spiral_gauntlet(),
            3 => open_slalom(),
            _ => circuit_oval(),
        };
        walls.extend(track_walls);

        Self {
            preset_id,
            walls,
            boundary_count,
            checkpoints,
            spawn_x,
            spawn_y,
            spawn_angle,
        }
    }

    pub fn add_wall(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        let dx = x2 - x1;
        let dy = y2 - y1;
        if dx * dx + dy * dy < 9.0 {
            return;
        }
        self.walls.push(Segment::new(
            x1.clamp(4.0, WORLD_WIDTH - 4.0),
            y1.clamp(4.0, WORLD_HEIGHT - 4.0),
            x2.clamp(4.0, WORLD_WIDTH - 4.0),
            y2.clamp(4.0, WORLD_HEIGHT - 4.0),
        ));
    }

    pub fn erase_walls_near(&mut self, x: f32, y: f32, radius: f32) -> usize {
        let r_sq = radius * radius;
        let before = self.walls.len();
        let boundary_count = self.boundary_count;
        let mut idx = 0;
        self.walls.retain(|seg| {
            let keep = idx < boundary_count || seg.dist_sq_to_point(x, y) > r_sq;
            idx += 1;
            keep
        });
        before - self.walls.len()
    }

    pub fn clear_custom_walls(&mut self) {
        self.walls.truncate(self.boundary_count);
    }
}

fn boundary_walls() -> Vec<Segment> {
    let m = 12.0;
    let w = WORLD_WIDTH - m;
    let h = WORLD_HEIGHT - m;
    vec![
        Segment::new(m, m, w, m),
        Segment::new(w, m, w, h),
        Segment::new(w, h, m, h),
        Segment::new(m, h, m, m),
    ]
}

fn polygon_walls(pts: &[(f32, f32)], closed: bool) -> Vec<Segment> {
    if pts.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pts.len());
    for i in 0..pts.len() - 1 {
        out.push(Segment::new(pts[i].0, pts[i].1, pts[i + 1].0, pts[i + 1].1));
    }
    if closed {
        let last = pts[pts.len() - 1];
        out.push(Segment::new(last.0, last.1, pts[0].0, pts[0].1));
    }
    out
}

fn checkpoints_between(outer: &[(f32, f32)], inner: &[(f32, f32)]) -> Vec<Segment> {
    outer
        .iter()
        .zip(inner.iter())
        .map(|(o, i)| Segment::new(o.0, o.1, i.0, i.1))
        .collect()
}

fn circuit_oval() -> (Vec<Segment>, Vec<Segment>, f32, f32, f32) {
    // Clockwise oval circuit with a lower-right chicane.
    let outer = [
        (180.0, 95.0),
        (500.0, 75.0),
        (820.0, 95.0),
        (925.0, 210.0),
        (925.0, 410.0),
        (820.0, 560.0),
        (650.0, 505.0),
        (500.0, 610.0),
        (210.0, 600.0),
        (80.0, 470.0),
        (80.0, 220.0),
    ];
    let inner = [
        (235.0, 205.0),
        (500.0, 190.0),
        (760.0, 210.0),
        (805.0, 275.0),
        (805.0, 395.0),
        (755.0, 455.0),
        (640.0, 395.0),
        (500.0, 485.0),
        (255.0, 485.0),
        (200.0, 420.0),
        (200.0, 265.0),
    ];

    let mut walls = polygon_walls(&outer, true);
    walls.extend(polygon_walls(&inner, true));

    // Rotate checkpoint order so checkpoint[0] is just ahead of spawn (top straight heading right).
    let mut cps = checkpoints_between(&outer, &inner);
    cps.rotate_left(1);
    (walls, cps, 270.0, 145.0, 0.0)
}

fn technical_s_curve() -> (Vec<Segment>, Vec<Segment>, f32, f32, f32) {
    let outer = [
        (110.0, 90.0),
        (420.0, 90.0),
        (480.0, 220.0),
        (610.0, 220.0),
        (680.0, 90.0),
        (910.0, 90.0),
        (920.0, 350.0),
        (810.0, 610.0),
        (510.0, 610.0),
        (430.0, 460.0),
        (260.0, 460.0),
        (190.0, 610.0),
        (80.0, 560.0),
        (80.0, 250.0),
    ];
    let inner = [
        (195.0, 190.0),
        (350.0, 190.0),
        (410.0, 330.0),
        (675.0, 330.0),
        (740.0, 200.0),
        (815.0, 200.0),
        (820.0, 345.0),
        (745.0, 505.0),
        (575.0, 505.0),
        (495.0, 355.0),
        (205.0, 355.0),
        (170.0, 480.0),
        (170.0, 450.0),
        (170.0, 265.0),
    ];

    let mut walls = polygon_walls(&outer, true);
    walls.extend(polygon_walls(&inner, true));
    let mut cps = checkpoints_between(&outer, &inner);
    cps.rotate_left(1);
    (walls, cps, 225.0, 140.0, 0.0)
}

fn spiral_gauntlet() -> (Vec<Segment>, Vec<Segment>, f32, f32, f32) {
    // Open-ended inward spiral corridor from outer top-left to inner center.
    let outer = [
        (75.0, 75.0),
        (925.0, 75.0),
        (925.0, 625.0),
        (75.0, 625.0),
        (75.0, 190.0),
        (790.0, 190.0),
        (790.0, 510.0),
        (215.0, 510.0),
        (215.0, 300.0),
        (650.0, 300.0),
        (650.0, 410.0),
        (375.0, 410.0),
    ];
    let inner = [
        (190.0, 185.0),
        (815.0, 185.0),
        (815.0, 515.0),
        (190.0, 515.0),
        (190.0, 295.0),
        (675.0, 295.0),
        (675.0, 415.0),
        (335.0, 415.0),
        (335.0, 395.0),
        (540.0, 395.0),
        (540.0, 355.0),
        (430.0, 355.0),
    ];

    let mut walls = Vec::new();
    // Outer and inner spiral walls.
    walls.extend(polygon_walls(&outer, false));
    walls.extend(polygon_walls(
        &[(75.0, 75.0), (75.0, 190.0), (190.0, 185.0)],
        false,
    ));
    walls.extend(polygon_walls(&inner, false));
    // Cap the center goal end.
    walls.push(Segment::new(375.0, 410.0, 430.0, 355.0));

    let mut cps = checkpoints_between(&outer, &inner);
    cps.rotate_left(1);
    (walls, cps, 150.0, 130.0, 0.0)
}

fn open_slalom() -> (Vec<Segment>, Vec<Segment>, f32, f32, f32) {
    // Wide rectangular arena with staggered obstacles and horizontal progression gates.
    let mut walls = vec![
        Segment::new(60.0, 100.0, 940.0, 100.0),
        Segment::new(940.0, 100.0, 940.0, 600.0),
        Segment::new(940.0, 600.0, 60.0, 600.0),
        Segment::new(60.0, 600.0, 60.0, 100.0),
    ];

    // Staggered slalom barriers
    let barriers = [
        (240.0, 100.0, 240.0, 390.0),
        (420.0, 310.0, 420.0, 600.0),
        (600.0, 100.0, 600.0, 390.0),
        (780.0, 310.0, 780.0, 600.0),
    ];
    for (x1, y1, x2, y2) in barriers {
        walls.push(Segment::new(x1, y1, x2, y2));
    }

    let cps = vec![
        Segment::new(240.0, 390.0, 240.0, 600.0),
        Segment::new(420.0, 100.0, 420.0, 310.0),
        Segment::new(600.0, 390.0, 600.0, 600.0),
        Segment::new(780.0, 100.0, 780.0, 310.0),
        Segment::new(910.0, 100.0, 910.0, 600.0),
    ];

    (walls, cps, 125.0, 350.0, 0.25)
}

/// Returns closest hit distance along ray `(ox, oy) + t * (dx, dy)` in `[0, max_dist]`.
#[inline]
pub fn ray_cast(ox: f32, oy: f32, dx: f32, dy: f32, max_dist: f32, walls: &[Segment]) -> f32 {
    let mut best = max_dist;
    for seg in walls {
        let sx = seg.x2 - seg.x1;
        let sy = seg.y2 - seg.y1;
        let denom = dx * sy - dy * sx;
        if denom.abs() < 1e-7 {
            continue;
        }
        let qpx = seg.x1 - ox;
        let qpy = seg.y1 - oy;
        let t = (qpx * sy - qpy * sx) / denom;
        if t < 0.0 || t >= best {
            continue;
        }
        let u = (qpx * dy - qpy * dx) / denom;
        if (0.0..=1.0).contains(&u) {
            best = t;
        }
    }
    best
}

/// True if closed segments `a` and `b` intersect.
#[inline]
pub fn segments_intersect(a: Segment, b: Segment) -> bool {
    let r_x = a.x2 - a.x1;
    let r_y = a.y2 - a.y1;
    let s_x = b.x2 - b.x1;
    let s_y = b.y2 - b.y1;

    let denom = r_x * s_y - r_y * s_x;
    if denom.abs() < 1e-7 {
        return false;
    }

    let qpx = b.x1 - a.x1;
    let qpy = b.y1 - a.y1;
    let t = (qpx * s_y - qpy * s_x) / denom;
    let u = (qpx * r_y - qpy * r_x) / denom;

    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raycast_hits_closest_wall_and_ignores_parallel_or_behind() {
        let walls = vec![
            Segment::new(150.0, 0.0, 150.0, 200.0), // hit at t=150
            Segment::new(90.0, 0.0, 90.0, 200.0),   // closer hit at t=90
            Segment::new(-40.0, 0.0, -40.0, 200.0), // behind origin
            Segment::new(0.0, 10.0, 200.0, 10.0),   // parallel to +X ray
        ];
        let hit = ray_cast(0.0, 100.0, 1.0, 0.0, 250.0, &walls);
        assert!((hit - 90.0).abs() < 1e-4, "hit={hit}");

        let miss = ray_cast(0.0, 100.0, 0.0, 1.0, 250.0, &walls);
        assert!((miss - 250.0).abs() < 1e-4, "miss={miss}");
    }

    #[test]
    fn segment_intersection_detects_checkpoint_crossing() {
        let move_seg = Segment::new(90.0, 50.0, 110.0, 50.0);
        let gate = Segment::new(100.0, 10.0, 100.0, 90.0);
        let far_gate = Segment::new(200.0, 10.0, 200.0, 90.0);

        assert!(segments_intersect(move_seg, gate));
        assert!(!segments_intersect(move_seg, far_gate));
    }

    #[test]
    fn track_custom_walls_can_be_added_and_erased() {
        let mut track = Track::from_preset(0);
        let initial = track.walls.len();
        track.add_wall(300.0, 300.0, 360.0, 320.0);
        assert_eq!(track.walls.len(), initial + 1);

        let erased = track.erase_walls_near(330.0, 310.0, 15.0);
        assert_eq!(erased, 1);
        assert_eq!(track.walls.len(), initial);
    }
}
