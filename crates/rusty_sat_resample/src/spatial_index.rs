//! Small 2D spatial index foundations for nearest-neighbour resampling.
//!
//! This is intentionally dependency-free for the first accelerated slice. It
//! provides exact nearest-point lookup with a KD-tree over finite 2D points.
//! Later Pyresample parity work can extend this with geocentric coordinates,
//! multiple neighbours, and chunked/parallel query execution.

use rusty_sat_core::{Result, RustySatError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    index: usize,
    x: f64,
    y: f64,
}

impl Point2D {
    pub fn new(index: usize, x: f64, y: f64) -> Result<Self> {
        if !x.is_finite() || !y.is_finite() {
            return Err(RustySatError::invalid_input(
                "spatial index points must have finite coordinates",
            ));
        }
        Ok(Self { index, x, y })
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestPoint {
    index: usize,
    distance: f64,
}

impl NearestPoint {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn distance(&self) -> f64 {
        self.distance
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KdPointIndex2D {
    nodes: Vec<KdNode>,
    root: Option<usize>,
    point_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct KdNode {
    point: Point2D,
    axis: Axis,
    left: Option<usize>,
    right: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

impl Axis {
    fn for_depth(depth: usize) -> Self {
        if depth.is_multiple_of(2) {
            Self::X
        } else {
            Self::Y
        }
    }

    fn value(self, point: Point2D) -> f64 {
        match self {
            Self::X => point.x,
            Self::Y => point.y,
        }
    }

    fn query_value(self, x: f64, y: f64) -> f64 {
        match self {
            Self::X => x,
            Self::Y => y,
        }
    }
}

impl KdPointIndex2D {
    pub fn from_points(points: impl IntoIterator<Item = Point2D>) -> Self {
        let points: Vec<_> = points.into_iter().collect();
        let point_count = points.len();
        let mut nodes = Vec::with_capacity(point_count);
        let root = build_kd_tree(points, 0, &mut nodes);
        Self {
            nodes,
            root,
            point_count,
        }
    }

    pub fn from_xy(xs: &[f64], ys: &[f64]) -> Result<Self> {
        if xs.len() != ys.len() {
            return Err(RustySatError::invalid_input(format!(
                "x/y coordinate lengths differ: {} != {}",
                xs.len(),
                ys.len()
            )));
        }
        let points = xs.iter().zip(ys).enumerate().filter_map(|(index, (x, y))| {
            if x.is_finite() && y.is_finite() {
                Some(Point2D {
                    index,
                    x: *x,
                    y: *y,
                })
            } else {
                None
            }
        });
        Ok(Self::from_points(points))
    }

    pub fn len(&self) -> usize {
        self.point_count
    }

    pub fn is_empty(&self) -> bool {
        self.point_count == 0
    }

    pub fn nearest(
        &self,
        x: f64,
        y: f64,
        radius_of_influence: Option<f64>,
    ) -> Result<Option<NearestPoint>> {
        if !x.is_finite() || !y.is_finite() {
            return Err(RustySatError::invalid_input(
                "nearest-point query coordinates must be finite",
            ));
        }
        if radius_of_influence.is_some_and(|radius| radius < 0.0) {
            return Err(RustySatError::invalid_input(
                "radius_of_influence must be non-negative",
            ));
        }
        let max_distance_squared = radius_of_influence
            .map(|radius| radius * radius)
            .unwrap_or(f64::INFINITY);
        let mut best = None;
        if let Some(root) = self.root {
            self.search(root, x, y, max_distance_squared, &mut best);
        }
        Ok(best.map(|best| NearestPoint {
            index: best.index,
            distance: best.distance_squared.sqrt(),
        }))
    }

    fn search(
        &self,
        node_index: usize,
        x: f64,
        y: f64,
        max_distance_squared: f64,
        best: &mut Option<BestPoint>,
    ) {
        let node = &self.nodes[node_index];
        let distance_squared = squared_distance(node.point.x, node.point.y, x, y);
        let current_limit = best
            .map(|best| best.distance_squared)
            .unwrap_or(max_distance_squared);
        if distance_squared <= current_limit
            && best_is_better(
                node.point.index,
                distance_squared,
                best.map(|best| (best.index, best.distance_squared)),
            )
        {
            *best = Some(BestPoint {
                index: node.point.index,
                distance_squared,
            });
        }

        let query_axis_value = node.axis.query_value(x, y);
        let node_axis_value = node.axis.value(node.point);
        let diff = query_axis_value - node_axis_value;
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(child) = near {
            self.search(child, x, y, max_distance_squared, best);
        }
        let updated_limit = best
            .map(|best| best.distance_squared)
            .unwrap_or(max_distance_squared);
        if diff * diff <= updated_limit {
            if let Some(child) = far {
                self.search(child, x, y, max_distance_squared, best);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BestPoint {
    index: usize,
    distance_squared: f64,
}

fn build_kd_tree(mut points: Vec<Point2D>, depth: usize, nodes: &mut Vec<KdNode>) -> Option<usize> {
    if points.is_empty() {
        return None;
    }
    let axis = Axis::for_depth(depth);
    points.sort_by(|left, right| {
        axis.value(*left)
            .total_cmp(&axis.value(*right))
            .then_with(|| left.index.cmp(&right.index))
    });
    let median = points.len() / 2;
    let right_points = points.split_off(median + 1);
    let point = points.pop().expect("median point exists after split_off");
    let node_index = nodes.len();
    nodes.push(KdNode {
        point,
        axis,
        left: None,
        right: None,
    });
    let left = build_kd_tree(points, depth + 1, nodes);
    let right = build_kd_tree(right_points, depth + 1, nodes);
    nodes[node_index].left = left;
    nodes[node_index].right = right;
    Some(node_index)
}

fn squared_distance(left_x: f64, left_y: f64, right_x: f64, right_y: f64) -> f64 {
    let dx = left_x - right_x;
    let dy = left_y - right_y;
    dx * dx + dy * dy
}

fn best_is_better(
    candidate_index: usize,
    candidate_distance_squared: f64,
    best: Option<(usize, f64)>,
) -> bool {
    match best {
        None => true,
        Some((best_index, best_distance_squared)) => {
            candidate_distance_squared < best_distance_squared
                || (candidate_distance_squared == best_distance_squared
                    && candidate_index < best_index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kd_index_finds_nearest_point() {
        let index = KdPointIndex2D::from_points([
            Point2D::new(0, -1.0, 0.0).unwrap(),
            Point2D::new(1, 2.0, 0.0).unwrap(),
            Point2D::new(2, 0.25, 0.25).unwrap(),
        ]);

        let nearest = index.nearest(0.0, 0.0, None).unwrap().unwrap();

        assert_eq!(nearest.index(), 2);
        assert!((nearest.distance() - (0.125_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn kd_index_respects_radius_of_influence() {
        let index = KdPointIndex2D::from_xy(&[0.0, 10.0], &[0.0, 10.0]).unwrap();

        assert_eq!(
            index
                .nearest(0.2, 0.0, Some(0.25))
                .unwrap()
                .unwrap()
                .index(),
            0
        );
        assert_eq!(index.nearest(1.0, 1.0, Some(0.25)).unwrap(), None);
    }

    #[test]
    fn kd_index_skips_non_finite_source_points() {
        let index = KdPointIndex2D::from_xy(&[f64::NAN, 3.0], &[1.0, 4.0]).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(index.nearest(3.1, 4.0, None).unwrap().unwrap().index(), 1);
    }

    #[test]
    fn kd_index_prefers_lowest_source_index_for_ties() {
        let index = KdPointIndex2D::from_xy(&[-1.0, 1.0], &[0.0, 0.0]).unwrap();

        assert_eq!(index.nearest(0.0, 0.0, None).unwrap().unwrap().index(), 0);
    }

    #[test]
    fn kd_index_returns_none_for_empty_tree() {
        let index = KdPointIndex2D::from_xy(&[], &[]).unwrap();

        assert!(index.is_empty());
        assert_eq!(index.nearest(0.0, 0.0, None).unwrap(), None);
    }

    #[test]
    fn kd_index_rejects_mismatched_xy_lengths() {
        let err = KdPointIndex2D::from_xy(&[0.0, 1.0], &[0.0]).unwrap_err();
        assert!(err.to_string().contains("lengths differ"));
    }

    #[test]
    fn kd_index_rejects_non_finite_query_coordinates() {
        let index = KdPointIndex2D::from_xy(&[0.0], &[0.0]).unwrap();

        assert!(index.nearest(f64::NAN, 0.0, None).is_err());
        assert!(index.nearest(0.0, f64::INFINITY, None).is_err());
    }

    #[test]
    fn kd_index_rejects_negative_radius() {
        let index = KdPointIndex2D::from_xy(&[0.0], &[0.0]).unwrap();

        assert!(index.nearest(0.0, 0.0, Some(-0.5)).is_err());
    }
}
