//! Fixed-capacity node arena: never grows past `MAX_NODES` (CLAUDE.md Law 2).

use cozy_chess::Move;

/// Hard cap on nodes per search (~1.8 MB at 28 bytes per node).
pub const MAX_NODES: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeState {
    /// Not yet evaluated (or evaluated but the arena had no room for children).
    Unexpanded,
    /// Children allocated at `first_child..first_child + num_children`.
    Expanded,
    /// Game over; value from the perspective of the side to move here.
    Terminal(f32),
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// Move leading into this node (`None` for the root).
    pub mv: Option<Move>,
    pub prior: f32,
    pub visits: u32,
    /// Sum of backed-up values from the perspective of the side that played `mv`
    /// (i.e. the parent's side to move), so the parent maximizes `Q` directly.
    pub value_sum: f32,
    pub first_child: u32,
    pub num_children: u8,
    pub state: NodeState,
}

impl Node {
    fn new(mv: Option<Move>, prior: f32) -> Self {
        Self {
            mv,
            prior,
            visits: 0,
            value_sum: 0.0,
            first_child: 0,
            num_children: 0,
            state: NodeState::Unexpanded,
        }
    }

    /// Mean value from the parent's perspective; 0 when unvisited. Selection
    /// applies its root/non-root FPU rule separately.
    pub fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    pub fn children(&self) -> std::ops::Range<usize> {
        let start = self.first_child as usize;
        start..start + self.num_children as usize
    }
}

pub struct Arena {
    nodes: Vec<Node>,
    capacity: usize,
}

impl Arena {
    /// New arena holding just the root. `capacity` is clamped to `MAX_NODES`.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, MAX_NODES);
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(Node::new(None, 1.0));
        Self { nodes, capacity }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Allocate children for `parent`; returns `false` (and changes nothing) if full.
    pub fn expand(&mut self, parent: usize, priors: &[(Move, f32)]) -> bool {
        assert!(priors.len() <= u8::MAX as usize, "too many children");
        if self.nodes.len() + priors.len() > self.capacity {
            return false;
        }
        let first = self.nodes.len() as u32;
        self.nodes
            .extend(priors.iter().map(|&(mv, p)| Node::new(Some(mv), p)));
        let node = &mut self.nodes[parent];
        node.first_child = first;
        node.num_children = priors.len() as u8;
        node.state = NodeState::Expanded;
        true
    }
}

impl std::ops::Index<usize> for Arena {
    type Output = Node;
    fn index(&self, i: usize) -> &Node {
        &self.nodes[i]
    }
}

impl std::ops::IndexMut<usize> for Arena {
    fn index_mut(&mut self, i: usize) -> &mut Node {
        &mut self.nodes[i]
    }
}
